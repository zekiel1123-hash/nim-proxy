// server.js
// OpenAI-compatible NVIDIA NIM proxy
// STREAMING ONLY
// Models:
// - glm-5.1 (default fallback)
// - kimi-k2.6
// - deepseek-v4

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());

app.use(express.json({
  limit: '100mb'
}));

app.use(express.urlencoded({
  extended: true,
  limit: '100mb'
}));

// ============================================
// CONFIG
// ============================================

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY =
  process.env.NIM_API_KEY;

// Display reasoning in <think> blocks
const SHOW_REASONING = true;

// Enable thinking mode
const ENABLE_THINKING_MODE = true;

// DeepSeek reasoning effort
const REASONING_EFFORT = 'low';

// Retry behavior for transient upstream failures (cold starts, gateway
// timeouts). Only status codes in RETRYABLE_STATUS get retried — anything
// else (e.g. 400/401/403) fails fast since retrying won't help.
const MAX_RETRIES = 2;
const RETRY_BASE_DELAY_MS = 500;
const REQUEST_TIMEOUT_MS = 60000;
const RETRYABLE_STATUS = new Set([504, 502, 503]);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ============================================
// MODEL MAPPING
// ============================================

const MODEL_MAPPING = {
  'glm-5.1': 'z-ai/glm-5.2',

  'kimi-k2.6':
    'moonshotai/kimi-k2.6',

  'deepseek-v4':
    'deepseek-ai/deepseek-v4-pro'
};

// ============================================
// MODEL CONFIG
// ============================================

function buildThinkingConfig(model) {
  // GLM
  if (model.includes('glm')) {
    return {
      chat_template_kwargs: {
        enable_thinking: true,
        clear_thinking: true
      }
    };
  }

  // Kimi
  if (model.includes('kimi')) {
    return {
      chat_template_kwargs: {
        thinking: true
      }
    };
  }

  // DeepSeek
  if (model.includes('deepseek')) {
    return {
      reasoning_effort:
        REASONING_EFFORT
    };
  }

  return {};
}

function extractReasoning(delta) {
  return (
    delta?.reasoning ||
    delta?.reasoning_content ||
    ''
  );
}

// ============================================
// ERROR HELPERS
// ============================================

// When axios is configured with responseType: 'stream', a non-2xx response
// still comes back as a readable stream on error.response.data (NOT parsed
// JSON). Trying to JSON.stringify that stream directly blows up with
// "Converting circular structure to JSON" because it holds a reference back
// to the underlying socket. This drains the stream into text (and tries to
// parse it as JSON) so we can actually see/report what NVIDIA sent back.
function readStreamToString(stream) {
  return new Promise((resolve, reject) => {
    let data = '';

    stream.on('data', (chunk) => {
      data += chunk.toString();
    });

    stream.on('end', () => resolve(data));
    stream.on('error', reject);
  });
}

// Wraps the NVIDIA request with retry-on-transient-failure logic. Cold
// starts on NVCF (NVIDIA's serverless backend) commonly surface as a 504
// on the *first* call after idle time and succeed immediately on retry, so
// this only retries 502/503/504 with a short exponential backoff. Anything
// else (auth errors, bad request shape, etc.) is thrown immediately since
// retrying won't fix it.
async function postToNimWithRetry(url, payload, headers) {
  let lastError;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await axios.post(url, payload, {
        headers,
        responseType: 'stream',
        timeout: REQUEST_TIMEOUT_MS
      });
    } catch (error) {
      lastError = error;

      const status = error?.response?.status;
      const isRetryable = status
        ? RETRYABLE_STATUS.has(status)
        : error.code === 'ECONNABORTED'; // client-side timeout

      if (!isRetryable || attempt === MAX_RETRIES) {
        throw error;
      }

      const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt);

      console.warn(
        `NIM request failed (status ${status || error.code}), retrying in ${delay}ms ` +
          `[attempt ${attempt + 1}/${MAX_RETRIES}]`
      );

      await sleep(delay);
    }
  }

  throw lastError;
}

async function getSafeErrorPayload(error) {
  const responseData = error?.response?.data;

  // Stream case (responseType: 'stream') — drain it instead of stringifying.
  if (responseData && typeof responseData.on === 'function') {
    try {
      const raw = await readStreamToString(responseData);

      try {
        return JSON.parse(raw);
      } catch {
        return raw || error.message;
      }
    } catch (drainErr) {
      console.error('Failed to read error stream:', drainErr);
      return error.message;
    }
  }

  // Already a plain object/string (non-streamed error) — safe to use as-is.
  if (responseData) {
    return responseData;
  }

  return error.message;
}

// ============================================
// HEALTH
// ============================================

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    streaming_only: true,
    reasoning_display:
      SHOW_REASONING,
    thinking_mode:
      ENABLE_THINKING_MODE,
    fallback_model:
      'z-ai/glm-5.1'
  });
});

// ============================================
// MODELS
// ============================================

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',

    data: Object.keys(
      MODEL_MAPPING
    ).map((model) => ({
      id: model,
      object: 'model',
      created: Date.now(),
      owned_by:
        'nvidia-nim-proxy'
    }))
  });
});

// ============================================
// CHAT COMPLETIONS
// STREAMING ONLY
// ============================================

app.post(
  '/v1/chat/completions',
  async (req, res) => {
    try {
      const {
        model,
        messages,
        temperature,
        max_tokens
      } = req.body;

      // ============================================
      // FALLBACK TO GLM-5.1
      // ============================================

      const nimModel =
        MODEL_MAPPING[model] ||
        'z-ai/glm-5.1';

      // ============================================
      // BUILD REQUEST
      // ============================================

      const nimRequest = {
        model: nimModel,

        messages,

        temperature:
          temperature ?? 1.0,

        max_tokens:
          max_tokens ?? 4096,

        stream: true,

        ...(ENABLE_THINKING_MODE
          ? buildThinkingConfig(
              nimModel
            )
          : {})
      };

      // ============================================
      // NVIDIA REQUEST
      // ============================================

      const response =
        await postToNimWithRetry(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            Authorization:
              `Bearer ${NIM_API_KEY}`,

            'Content-Type':
              'application/json',

            Accept:
              'text/event-stream'
          }
        );

      // ============================================
      // SSE HEADERS
      // ============================================

      res.setHeader(
        'Content-Type',
        'text/event-stream'
      );

      res.setHeader(
        'Cache-Control',
        'no-cache'
      );

      res.setHeader(
        'Connection',
        'keep-alive'
      );

      // ============================================
      // STREAM HANDLING
      // ============================================

      let buffer = '';

      let reasoningOpen =
        false;

      response.data.on(
        'data',
        (chunk) => {
          buffer +=
            chunk.toString();

          const lines =
            buffer.split('\n');

          buffer =
            lines.pop() || '';

          for (const line of lines) {
            if (
              !line.startsWith(
                'data: '
              )
            ) {
              continue;
            }

            // ============================================
            // DONE
            // ============================================

            if (
              line.includes(
                '[DONE]'
              )
            ) {
              // close think block if needed
              if (
                SHOW_REASONING &&
                reasoningOpen
              ) {
                const closeChunk =
                  {
                    choices: [
                      {
                        delta: {
                          content:
                            '\n</think>\n'
                        }
                      }
                    ]
                  };

                res.write(
                  `data: ${JSON.stringify(
                    closeChunk
                  )}\n\n`
                );

                reasoningOpen =
                  false;
              }

              res.write(
                'data: [DONE]\n\n'
              );

              res.end();

              return;
            }

            try {
              const data =
                JSON.parse(
                  line.slice(6)
                );

              const delta =
                data?.choices?.[0]
                  ?.delta || {};

              const reasoning =
                extractReasoning(
                  delta
                );

              const content =
                delta.content || '';

              let output = '';

              // ============================================
              // REASONING
              // ============================================

              if (
                SHOW_REASONING &&
                reasoning
              ) {
                if (
                  !reasoningOpen
                ) {
                  output +=
                    '<think>\n';

                  reasoningOpen =
                    true;
                }

                output += reasoning;
              }

              // ============================================
              // CONTENT
              // ============================================

              if (content) {
                if (
                  SHOW_REASONING &&
                  reasoningOpen
                ) {
                  output +=
                    '\n</think>\n\n';

                  reasoningOpen =
                    false;
                }

                output += content;
              }

              // ============================================
              // REPLACE CONTENT
              // ============================================

              if (output) {
                data.choices[0].delta.content =
                  output;
              }

              // remove raw reasoning
              delete data
                .choices[0].delta
                .reasoning;

              delete data
                .choices[0].delta
                .reasoning_content;

              res.write(
                `data: ${JSON.stringify(
                  data
                )}\n\n`
              );
            } catch (err) {
              console.error(
                'Parse error:',
                err
              );
            }
          }
        }
      );

      response.data.on(
        'end',
        () => {
          res.end();
        }
      );

      response.data.on(
        'error',
        (err) => {
          console.error(
            'Stream error:',
            err
          );

          res.end();
        }
      );
    } catch (error) {
      const safePayload =
        await getSafeErrorPayload(
          error
        );

      console.error(
        'Proxy error:',
        safePayload
      );

      // If SSE headers were already flushed to the client before this
      // failed, we can no longer send a fresh status/JSON body — just
      // close the connection instead of throwing a second error.
      if (res.headersSent) {
        res.end();
        return;
      }

      res.status(
        error.response?.status ||
          500
      );

      res.json({
        error: {
          message: safePayload,

          type:
            'invalid_request_error',

          code:
            error.response
              ?.status || 500
        }
      });
    }
  }
);

// ============================================
// 404
// ============================================

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message:
        `Endpoint ${req.path} not found`,

      type:
        'invalid_request_error',

      code: 404
    }
  });
});

// ============================================
// START
// ============================================

app.listen(PORT, () => {
  console.log(
    `NVIDIA NIM proxy running on port ${PORT}`
  );

  console.log(
    `Fallback model: z-ai/glm-5.1`
  );

  console.log(
    `Reasoning display: ${
      SHOW_REASONING
        ? 'ENABLED'
        : 'DISABLED'
    }`
  );

  console.log(
    `Thinking mode: ${
      ENABLE_THINKING_MODE
        ? 'ENABLED'
        : 'DISABLED'
    }`
  );
});
