// server.js
// OpenAI-compatible NVIDIA NIM proxy
// STREAMING ONLY
//
// Models:
// - glm-5.1       -> z-ai/glm-5.2
// - kimi-k2.6     -> moonshotai/kimi-k2.6
// - deepseek-v4   -> deepseek-ai/deepseek-v4-pro
// - step-3.7-flash -> stepfun-ai/step-3.7-flash
//
// Step 3.7 Flash:
// - Thinking/reasoning output is intentionally DISABLED.
// - No chat_template_kwargs are sent.
// - No reasoning_effort is sent.
// - No <think> tags are generated.
// - Stray </think> emitted by upstream is removed.
//
// Other models:
// - Reasoning output remains enabled when SHOW_REASONING = true.

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ============================================
// MIDDLEWARE
// ============================================

app.use(cors());

app.use(
  express.json({
    limit: '100mb'
  })
);

app.use(
  express.urlencoded({
    extended: true,
    limit: '100mb'
  })
);

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

// Enable model-specific thinking parameters
const ENABLE_THINKING_MODE = true;

// DeepSeek reasoning effort
const REASONING_EFFORT = 'low';

// ============================================
// MODEL MAPPING
// ============================================

const MODEL_MAPPING = {
  'glm-5.1':
    'z-ai/glm-5.2',

  'kimi-k2.6':
    'moonshotai/kimi-k2.6',

  'deepseek-v4':
    'deepseek-ai/deepseek-v4-pro',

  'step-3.7-flash':
    'stepfun-ai/step-3.7-flash'
};

// ============================================
// DEFAULT FALLBACK
// ============================================

const FALLBACK_MODEL =
  'z-ai/glm-5.2';

// ============================================
// MODEL HELPERS
// ============================================

function isStep37Flash(model) {
  return (
    typeof model === 'string' &&
    (
      model.includes('step-3.7-flash') ||
      model.includes('stepfun-ai/step-3.7-flash')
    )
  );
}

// ============================================
// THINKING CONFIG
// ============================================
//
// IMPORTANT:
//
// Step 3.7 Flash intentionally gets NO thinking
// parameters.
//
// NVIDIA's published Step 3.7 Flash API example
// does not send chat_template_kwargs,
// reasoning_effort, or another thinking flag.
//
// This prevents the proxy from trying to force
// reasoning output from the model.
//

function buildThinkingConfig(model) {
  // ============================================
  // STEP 3.7 FLASH
  // REASONING DISABLED
  // ============================================

  if (isStep37Flash(model)) {
    return {};
  }

  // ============================================
  // GLM
  // ============================================

  if (
    typeof model === 'string' &&
    model.includes('glm')
  ) {
    return {
      chat_template_kwargs: {
        enable_thinking: true,
        clear_thinking: true
      }
    };
  }

  // ============================================
  // KIMI
  // ============================================

  if (
    typeof model === 'string' &&
    model.includes('kimi')
  ) {
    return {
      chat_template_kwargs: {
        thinking: true
      }
    };
  }

  // ============================================
  // DEEPSEEK
  // ============================================

  if (
    typeof model === 'string' &&
    model.includes('deepseek')
  ) {
    return {
      reasoning_effort:
        REASONING_EFFORT
    };
  }

  return {};
}

// ============================================
// REASONING EXTRACTION
// ============================================

function extractReasoning(delta) {
  if (!delta) {
    return '';
  }

  return (
    delta.reasoning ||
    delta.reasoning_content ||
    ''
  );
}

// ============================================
// REMOVE STRAY STEP THINK TAGS
// ============================================

function cleanStepContent(content) {
  if (
    typeof content !== 'string' ||
    !content
  ) {
    return content || '';
  }

  return content
    .replace(/<think>\s*/gi, '')
    .replace(/\s*<\/think>/gi, '');
}

// ============================================
// SAFE ERROR SERIALIZATION
// ============================================

function getSafeErrorMessage(error) {
  if (error?.response) {
    const status =
      error.response.status;

    const data =
      error.response.data;

    if (
      typeof data === 'string'
    ) {
      return data;
    }

    if (
      data &&
      typeof data === 'object'
    ) {
      try {
        if (data.error) {
          if (
            typeof data.error === 'string'
          ) {
            return data.error;
          }

          if (
            data.error.message
          ) {
            return data.error.message;
          }
        }

        if (data.message) {
          return data.message;
        }

        return JSON.stringify(data);
      } catch {
        return `NVIDIA API returned HTTP ${status}`;
      }
    }

    return `NVIDIA API returned HTTP ${status}`;
  }

  if (error?.message) {
    return error.message;
  }

  return 'Unknown proxy error';
}

// ============================================
// SAFE ERROR LOGGING
// ============================================

function logProxyError(error) {
  const status =
    error?.response?.status;

  const message =
    getSafeErrorMessage(error);

  console.error(
    '[Proxy Error]',
    status
      ? `HTTP ${status}`
      : 'No HTTP status',
    message
  );
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
      FALLBACK_MODEL,

    models: {
      'glm-5.1': {
        reasoning: true
      },

      'kimi-k2.6': {
        reasoning: true
      },

      'deepseek-v4': {
        reasoning: true
      },

      'step-3.7-flash': {
        reasoning: false
      }
    }
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

      created:
        Math.floor(
          Date.now() / 1000
        ),

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
    let response = null;

    try {
      const {
        model,
        messages,
        temperature,
        max_tokens,
        top_p,
        seed
      } = req.body || {};

      // ============================================
      // VALIDATE MESSAGES
      // ============================================

      if (
        !Array.isArray(messages)
      ) {
        return res.status(400).json({
          error: {
            message:
              'messages must be an array',

            type:
              'invalid_request_error',

            code: 400
          }
        });
      }

      // ============================================
      // RESOLVE MODEL
      // ============================================

      const nimModel =
        MODEL_MAPPING[model] ||
        FALLBACK_MODEL;

      const step37 =
        isStep37Flash(nimModel);

      // ============================================
      // BUILD BASE REQUEST
      // ============================================

      const nimRequest = {
        model: nimModel,

        messages,

        temperature:
          temperature ?? 1.0,

        max_tokens:
          max_tokens ?? (
            step37
              ? 16384
              : 4096
          ),

        stream: true
      };

      // ============================================
      // OPTIONAL TOP_P
      // ============================================

      if (
        top_p !== undefined &&
        top_p !== null
      ) {
        nimRequest.top_p =
          top_p;
      } else if (step37) {
        // NVIDIA's Step 3.7 example uses 0.95
        nimRequest.top_p = 0.95;
      }

      // ============================================
      // OPTIONAL SEED
      // ============================================

      if (
        seed !== undefined &&
        seed !== null
      ) {
        nimRequest.seed =
          seed;
      }

      // ============================================
      // THINKING CONFIG
      // ============================================

      if (
        ENABLE_THINKING_MODE &&
        !step37
      ) {
        Object.assign(
          nimRequest,
          buildThinkingConfig(
            nimModel
          )
        );
      }

      // ============================================
      // DEBUG INFORMATION
      // ============================================

      console.log(
        `[Request] ${model || 'unknown'} -> ${nimModel}` +
        `${step37 ? ' [REASONING DISABLED]' : ''}`
      );

      // ============================================
      // NVIDIA REQUEST START
      // ============================================

      console.log(
        `[NVIDIA Request Starting] ${nimModel}`
      );

      // ============================================
      // NVIDIA REQUEST
      // ============================================

      response =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization:
                `Bearer ${NIM_API_KEY}`,

              'Content-Type':
                'application/json',

              Accept:
                'text/event-stream'
            },

            responseType:
              'stream',

            validateStatus:
              () => true
          }
        );

      // ============================================
      // NVIDIA RESPONSE RECEIVED
      // ============================================

      console.log(
        `[NVIDIA Response Received] HTTP ${response.status}`
      );

      // ============================================
      // HANDLE NVIDIA HTTP ERROR
      // ============================================

      if (
        response.status < 200 ||
        response.status >= 300
      ) {
        let errorBody = '';

        try {
          for await (
            const chunk
            of response.data
          ) {
            errorBody +=
              chunk.toString();

            if (
              errorBody.length >
              100000
            ) {
              break;
            }
          }
        } catch {
          // Ignore stream-read error.
        }

        let parsedError =
          errorBody;

        try {
          parsedError =
            JSON.parse(
              errorBody
            );
        } catch {
          // Keep string.
        }

        console.error(
          `[NVIDIA Error] HTTP ${response.status}:`,
          typeof parsedError ===
            'string'
            ? parsedError
            : JSON.stringify(
                parsedError
              )
        );

        return res.status(
          response.status
        ).json({
          error: {
            message:
              typeof parsedError ===
              'string'
                ? parsedError
                : (
                    parsedError?.error
                      ?.message ||
                    parsedError?.message ||
                    `NVIDIA API returned HTTP ${response.status}`
                  ),

            type:
              'nvidia_api_error',

            code:
              response.status
          }
        });
      }

      // ============================================
      // SSE HEADERS
      // ============================================

      res.status(200);

      res.setHeader(
        'Content-Type',
        'text/event-stream; charset=utf-8'
      );

      res.setHeader(
        'Cache-Control',
        'no-cache, no-transform'
      );

      res.setHeader(
        'Connection',
        'keep-alive'
      );

      res.setHeader(
        'X-Accel-Buffering',
        'no'
      );

      // ============================================
      // STREAM STATE
      // ============================================

      let buffer = '';

      let reasoningOpen =
        false;

      let finished =
        false;

      // ============================================
      // SEND DONE
      // ============================================

      function sendDone() {
        if (finished) {
          return;
        }

        finished = true;

        if (
          SHOW_REASONING &&
          reasoningOpen
        ) {
          const closeChunk = {
            choices: [
              {
                delta: {
                  content:
                    '\n</think>\n'
                }
              }
            ]
          };

          try {
            res.write(
              `data: ${JSON.stringify(
                closeChunk
              )}\n\n`
            );
          } catch {
            // Client may already be gone.
          }

          reasoningOpen =
            false;
        }

        try {
          res.write(
            'data: [DONE]\n\n'
          );
        } catch {
          // Client may already be gone.
        }

        if (!res.writableEnded) {
          res.end();
        }
      }

      // ============================================
      // WRITE SSE DATA
      // ============================================

      function writeSSE(data) {
        if (
          finished ||
          res.writableEnded
        ) {
          return;
        }

        try {
          res.write(
            `data: ${JSON.stringify(
              data
            )}\n\n`
          );
        } catch (error) {
          console.error(
            '[SSE Write Error]',
            error.message
          );
        }
      }

      // ============================================
      // PROCESS SSE LINE
      // ============================================

      function processLine(line) {
        line =
          line.replace(
            /\r$/,
            ''
          );

        if (!line.trim()) {
          return;
        }

        if (
          line.startsWith(':')
        ) {
          return;
        }

        if (
          !line.startsWith(
            'data:'
          )
        ) {
          return;
        }

        const raw =
          line
            .slice(5)
            .trim();

        // ============================================
        // DONE
        // ============================================

        if (
          raw === '[DONE]'
        ) {
          console.log(
            '[NVIDIA Stream] Received [DONE]'
          );

          sendDone();
          return;
        }

        // ============================================
        // PARSE JSON
        // ============================================

        let data;

        try {
          data =
            JSON.parse(raw);
        } catch (error) {
          console.error(
            '[SSE Parse Error]',
            error.message
          );

          console.error(
            '[SSE Raw Data]',
            raw
          );

          return;
        }

        // ============================================
        // DEBUG STREAM DATA
        // ============================================

        console.log(
          '[NVIDIA Stream Data]',
          JSON.stringify(data)
        );

        // ============================================
        // GET DELTA
        // ============================================

        const choice =
          data?.choices?.[0];

        const delta =
          choice?.delta;

        if (!delta) {
          writeSSE(data);
          return;
        }

        // ============================================
        // STEP 3.7 FLASH
        // REASONING DISABLED
        // ============================================

        if (step37) {
          if (
            typeof delta.content ===
            'string'
          ) {
            delta.content =
              cleanStepContent(
                delta.content
              );
          }

          delete delta.reasoning;

          delete delta.reasoning_content;

          writeSSE(data);

          return;
        }

        // ============================================
        // OTHER MODELS
        // REASONING PROCESSING
        // ============================================

        const reasoning =
          extractReasoning(
            delta
          );

        const content =
          typeof delta.content ===
          'string'
            ? delta.content
            : '';

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

          output +=
            reasoning;
        }

        // ============================================
        // NORMAL CONTENT
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

          output +=
            content;
        }

        // ============================================
        // REPLACE CONTENT
        // ============================================

        if (output) {
          delta.content =
            output;
        }

        // ============================================
        // REMOVE RAW REASONING
        // ============================================

        delete delta.reasoning;

        delete delta.reasoning_content;

        // ============================================
        // SEND TO CLIENT
        // ============================================

        writeSSE(data);
      }

      // ============================================
      // STREAM DATA
      // ============================================

      console.log(
        `[NVIDIA Stream] Connected for ${nimModel}`
      );

      response.data.on(
        'data',
        (chunk) => {
          if (
            finished ||
            res.writableEnded
          ) {
            return;
          }

          console.log(
            `[NVIDIA Stream Chunk] ${chunk.length} bytes`
          );

          buffer +=
            chunk.toString(
              'utf8'
            );

          const lines =
            buffer.split(
              '\n'
            );

          buffer =
            lines.pop() || '';

          for (
            const line of lines
          ) {
            if (
              finished
            ) {
              break;
            }

            processLine(
              line
            );
          }
        }
      );

      // ============================================
      // STREAM END
      // ============================================

      response.data.on(
        'end',
        () => {
          console.log(
            '[NVIDIA Stream] Ended'
          );

          if (
            buffer.trim()
          ) {
            processLine(
              buffer
            );
          }

          sendDone();
        }
      );

      // ============================================
      // STREAM ERROR
      // ============================================

      response.data.on(
        'error',
        (error) => {
          console.error(
            '[NVIDIA Stream Error]',
            error.message
          );

          if (
            !finished &&
            !res.writableEnded
          ) {
            try {
              writeSSE({
                error: {
                  message:
                    error.message ||
                    'NVIDIA stream error',

                  type:
                    'stream_error'
                }
              });
            } catch {
              // Ignore write failure.
            }

            if (
              !res.writableEnded
            ) {
              res.end();
            }

            finished = true;
          }
        }
      );

      // ============================================
      // CLIENT DISCONNECT
      // ============================================

      req.on(
        'close',
        () => {
          console.log(
            `[Client] Disconnected from ${nimModel}`
          );

          if (
            !finished &&
            response?.data &&
            typeof response.data.destroy ===
              'function'
          ) {
            response.data.destroy();
          }

          finished = true;
        }
      );
    } catch (error) {
      // ============================================
      // SAFE PROXY ERROR
      // ============================================

      logProxyError(error);

      if (
        res.headersSent ||
        res.writableEnded
      ) {
        if (
          !res.writableEnded
        ) {
          res.end();
        }

        return;
      }

      const status =
        error?.response?.status ||
        500;

      const message =
        getSafeErrorMessage(
          error
        );

      return res.status(
        status
      ).json({
        error: {
          message,

          type:
            'invalid_request_error',

          code:
            status
        }
      });
    }
  }
);

// ============================================
// 404
// ============================================

app.all(
  '*',
  (req, res) => {
    if (
      res.headersSent
    ) {
      return res.end();
    }

    res.status(404).json({
      error: {
        message:
          `Endpoint ${req.path} not found`,

        type:
          'invalid_request_error',

        code: 404
      }
    });
  }
);

// ============================================
// START SERVER
// ============================================

app.listen(
  PORT,
  () => {
    console.log(
      '============================================'
    );

    console.log(
      'NVIDIA NIM proxy running'
    );

    console.log(
      `Port: ${PORT}`
    );

    console.log(
      `API base: ${NIM_API_BASE}`
    );

    console.log(
      `Fallback model: ${FALLBACK_MODEL}`
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

    console.log(
      'Step 3.7 Flash reasoning: DISABLED'
    );

    console.log(
      'Streaming only: ENABLED'
    );

    console.log(
      '============================================'
    );
  }
);
