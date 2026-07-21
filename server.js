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
const REASONING_EFFORT = 'med';

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
        clear_thinking: false
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
          temperature ?? 0.7,

        max_tokens:
          max_tokens ?? 8192,

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
              'stream'
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
      console.error(
        'Proxy error:',
        error?.response?.data ||
          error.message
      );

      res.status(
        error.response?.status ||
          500
      );

      res.json({
        error: {
          message:
            error?.response?.data ||
            error.message,

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
