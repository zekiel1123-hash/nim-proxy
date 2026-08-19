// server.js
// OpenAI-compatible NVIDIA NIM proxy
// STREAMING ONLY
//
// Client models:
// - glm-5.2
// - kimi-k2.6
// - deepseek-v4
// - step-3.7-flash
//
// NVIDIA NIM models:
// - glm-5.2       -> z-ai/glm-5.2
// - kimi-k2.6     -> moonshotai/kimi-k2.6
// - deepseek-v4   -> deepseek-ai/deepseek-v4-pro
// - step-3.7-flash -> stepfun-ai/step-3.7-flash
//
// Step 3.7 Flash:
// - Thinking/reasoning intentionally disabled.
// - No thinking parameters sent.
// - No reasoning_effort sent.
// - Stray <think> / </think> tags removed.
//
// RETRY / BACKOFF:
// - Retry on HTTP 429 and transient 5xx errors.
// - 10 seconds
// - 30 seconds
// - 60 seconds
// - 60 seconds
// - 60 seconds
// - Maximum 5 retries after the initial request.
//
// IMPORTANT:
// The retry cycle is NOT cancelled by req.on('close').
// We use res.on('close') instead so a normal request-body
// close does not incorrectly terminate the retry cycle.

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
// RETRY CONFIGURATION
// ============================================
//
// Initial request:
//   Attempt 1
//
// If NVIDIA returns 429 / transient 5xx:
//
//   Retry 1 -> 10 seconds
//   Retry 2 -> 30 seconds
//   Retry 3 -> 60 seconds
//   Retry 4 -> 60 seconds
//   Retry 5 -> 60 seconds
//
// Total possible attempts = 6
//

const MAX_RETRIES = 5;

const RETRY_DELAYS = [
  10_000,
  30_000,
  60_000,
  60_000,
  60_000
];

// ============================================
// MODEL MAPPING
// ============================================
//
// These are the model names exposed to the client.
//

const MODEL_MAPPING = {
  'glm-5.2':
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
      model.includes(
        'step-3.7-flash'
      ) ||
      model.includes(
        'stepfun-ai/step-3.7-flash'
      )
    )
  );
}

// ============================================
// THINKING CONFIG
// ============================================

function buildThinkingConfig(model) {

  // ==========================================
  // STEP 3.7 FLASH
  // REASONING DISABLED
  // ==========================================

  if (
    isStep37Flash(model)
  ) {
    return {};
  }

  // ==========================================
  // GLM
  // ==========================================

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

  // ==========================================
  // KIMI
  // ==========================================

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

  // ==========================================
  // DEEPSEEK
  // ==========================================

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
// CLEAN STEP 3.7 CONTENT
// ============================================

function cleanStepContent(content) {

  if (
    typeof content !== 'string' ||
    !content
  ) {
    return content || '';
  }

  return content
    .replace(
      /<think>\s*/gi,
      ''
    )
    .replace(
      /\s*<\/think>/gi,
      '');
}

// ============================================
// SAFE ERROR MESSAGE
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
            typeof data.error ===
            'string'
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

        return JSON.stringify(
          data
        );

      } catch {
        return (
          `NVIDIA API returned HTTP ${status}`
        );
      }
    }

    return (
      `NVIDIA API returned HTTP ${status}`
    );
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
// RETRYABLE STATUS
// ============================================

function isRetryableStatus(status) {

  // Rate limit
  if (status === 429) {
    return true;
  }

  // Temporary server-side errors
  if (
    status === 500 ||
    status === 502 ||
    status === 503 ||
    status === 504
  ) {
    return true;
  }

  return false;
}

// ============================================
// SLEEP
// ============================================

function sleep(ms) {

  return new Promise(
    (resolve) => {
      setTimeout(
        resolve,
        ms
      );
    }
  );
}

// ============================================
// HEALTH
// ============================================

app.get(
  '/health',
  (req, res) => {

    res.json({

      status: 'ok',

      streaming_only: true,

      reasoning_display:
        SHOW_REASONING,

      thinking_mode:
        ENABLE_THINKING_MODE,

      fallback_model:
        FALLBACK_MODEL,

      retry: {
        enabled: true,
        max_retries:
          MAX_RETRIES,
        delays: [
          '10s',
          '30s',
          '60s',
          '60s',
          '60s'
        ]
      },

      models: {

        'glm-5.2': {
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
  }
);

// ============================================
// MODELS
// ============================================

app.get(
  '/v1/models',
  (req, res) => {

    res.json({

      object: 'list',

      data:
        Object.keys(
          MODEL_MAPPING
        ).map(
          (model) => ({

            id: model,

            object: 'model',

            created:
              Math.floor(
                Date.now() / 1000
              ),

            owned_by:
              'nvidia-nim-proxy'

          })
        )

    });
  }
);

// ============================================
// CHAT COMPLETIONS
// STREAMING ONLY
// ============================================

app.post(
  '/v1/chat/completions',
  async (req, res) => {

    let response = null;

    // ==========================================
    // IMPORTANT CLIENT CONNECTION STATE
    // ==========================================
    //
    // DO NOT use req.on('close') here.
    //
    // The incoming request can close normally after
    // Express has received the request body.
    //
    // Cancelling retries from req.close caused:
    //
    // [Proxy] Client disconnected; stopping retry cycle
    //
    // even though the client was still waiting.
    //
    // We instead monitor the RESPONSE connection.
    //

    let clientDisconnected =
      false;

    let responseFinished =
      false;

    res.on(
      'finish',
      () => {
        responseFinished =
          true;
      }
    );

    res.on(
      'close',
      () => {

        // If the response did not finish normally,
        // the downstream client really disconnected.

        if (
          !responseFinished
        ) {
          clientDisconnected =
            true;

          console.log(
            '[Proxy] Client response connection closed'
          );

          // If an NVIDIA stream is already active,
          // destroy it.
          //
          // This does NOT interfere with retries that
          // have not started streaming yet.
          //

          if (
            response?.data &&
            typeof response.data.destroy ===
              'function'
          ) {
            try {
              response.data.destroy();
            } catch {
              // Ignore.
            }
          }
        }
      }
    );

    try {

      const {
        model,
        messages,
        temperature,
        max_tokens,
        top_p,
        seed
      } = req.body || {};

      // ==========================================
      // VALIDATE MESSAGES
      // ==========================================

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

      // ==========================================
      // RESOLVE MODEL
      // ==========================================

      const nimModel =
        MODEL_MAPPING[model] ||
        FALLBACK_MODEL;

      const step37 =
        isStep37Flash(
          nimModel
        );

      // ==========================================
      // BUILD REQUEST
      // ==========================================

      const nimRequest = {

        model:
          nimModel,

        messages,

        temperature:
          temperature ?? 1.0,

        max_tokens:
          max_tokens ??
          (
            step37
              ? 16384
              : 4096
          ),

        stream: true

      };

      // ==========================================
      // TOP P
      // ==========================================

      if (
        top_p !== undefined &&
        top_p !== null
      ) {

        nimRequest.top_p =
          top_p;

      } else if (step37) {

        nimRequest.top_p =
          0.95;
      }

      // ==========================================
      // SEED
      // ==========================================

      if (
        seed !== undefined &&
        seed !== null
      ) {

        nimRequest.seed =
          seed;
      }

      // ==========================================
      // THINKING CONFIG
      // ==========================================

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

      // ==========================================
      // REQUEST LOG
      // ==========================================

      console.log(
        `[Proxy] ${model || 'unknown'} -> ${nimModel}` +
        (
          step37
            ? ' [REASONING DISABLED]'
            : ''
        )
      );

      // ==========================================
      // NVIDIA REQUEST + RETRY LOOP
      // ==========================================

      let lastStatus = null;

      let lastError = null;

      for (
        let attempt = 0;
        attempt <= MAX_RETRIES;
        attempt++
      ) {

        // ========================================
        // DO NOT STOP JUST BECAUSE req.close
        // FIRED.
        //
        // Only stop if the response connection
        // actually closed.
        // ========================================

        if (
          clientDisconnected
        ) {

          console.log(
            '[Proxy] Response connection closed; stopping retry cycle'
          );

          return;
        }

        console.log(
          `[NVIDIA Request] Attempt ${attempt + 1}/${MAX_RETRIES + 1}`
        );

        try {

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

                timeout: 0,

                validateStatus:
                  () => true

              }
            );

          lastStatus =
            response.status;

          // ======================================
          // SUCCESS
          // ======================================

          if (
            response.status >= 200 &&
            response.status < 300
          ) {

            console.log(
              `[NVIDIA Request] Success HTTP ${response.status}`
            );

            break;
          }

          // ======================================
          // NON-RETRYABLE ERROR
          // ======================================

          if (
            !isRetryableStatus(
              response.status
            )
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
              // Ignore.
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

            const message =
              typeof parsedError ===
              'string'
                ? parsedError
                : (
                    parsedError?.error
                      ?.message ||
                    parsedError?.message ||
                    `NVIDIA API returned HTTP ${response.status}`
                  );

            console.error(
              `[NVIDIA Error] HTTP ${response.status}: ${message}`
            );

            if (
              !res.headersSent
            ) {

              return res
                .status(
                  response.status
                )
                .json({

                  error: {

                    message,

                    type:
                      'nvidia_api_error',

                    code:
                      response.status

                  }

                });
            }

            return;
          }

          // ======================================
          // RETRYABLE ERROR
          // ======================================

          let retryBody = '';

          try {

            for await (
              const chunk
              of response.data
            ) {

              retryBody +=
                chunk.toString();

              if (
                retryBody.length >
                100000
              ) {
                break;
              }
            }

          } catch {
            // Ignore.
          }

          let retryMessage =
            retryBody;

          try {

            const parsed =
              JSON.parse(
                retryBody
              );

            retryMessage =
              parsed?.error
                ?.message ||
              parsed?.message ||
              retryBody;

          } catch {
            // Keep string.
          }

          lastError =
            retryMessage ||
            `NVIDIA API returned HTTP ${response.status}`;

          // ======================================
          // NO RETRIES LEFT
          // ======================================

          if (
            attempt >=
            MAX_RETRIES
          ) {

            console.error(
              `[NVIDIA Retry] Exhausted retries after HTTP ${response.status}`
            );

            break;
          }

          // ======================================
          // BACKOFF
          // ======================================

          const delay =
            RETRY_DELAYS[
              attempt
            ] ??
            60_000;

          console.log(
            `[NVIDIA Retry] HTTP ${response.status}. ` +
            `Retry ${attempt + 1}/${MAX_RETRIES} ` +
            `in ${delay / 1000}s`
          );

          console.log(
            `[NVIDIA Retry] ${lastError}`
          );

          await sleep(
            delay
          );

          // ======================================
          // CHECK AFTER SLEEP
          // ======================================

          if (
            clientDisconnected
          ) {

            console.log(
              '[Proxy] Response connection closed during backoff; stopping retry cycle'
            );

            return;
          }

        } catch (error) {

          lastError =
            error;

          // ======================================
          // NETWORK / AXIOS ERROR
          // ======================================

          console.error(
            '[NVIDIA Request Error]',
            error?.message ||
              'Unknown error'
          );

          // ======================================
          // RETRY NETWORK ERRORS
          // ======================================

          if (
            attempt >=
            MAX_RETRIES
          ) {
            break;
          }

          const delay =
            RETRY_DELAYS[
              attempt
            ] ??
            60_000;

          console.log(
            `[NVIDIA Retry] Network error. ` +
            `Retry ${attempt + 1}/${MAX_RETRIES} ` +
            `in ${delay / 1000}s`
          );

          await sleep(
            delay
          );

          if (
            clientDisconnected
          ) {

            console.log(
              '[Proxy] Response connection closed during backoff; stopping retry cycle'
            );

            return;
          }
        }
      }

      // ==========================================
      // NO SUCCESSFUL RESPONSE
      // ==========================================

      if (
        !response ||
        response.status < 200 ||
        response.status >= 300
      ) {

        const message =
          typeof lastError ===
          'string'
            ? lastError
            : getSafeErrorMessage(
                lastError
              );

        console.error(
          '[Proxy] NVIDIA request failed after retries:',
          message
        );

        if (
          !res.headersSent &&
          !res.writableEnded
        ) {

          return res
            .status(
              lastStatus || 503
            )
            .json({

              error: {

                message:
                  message ||
                  'NVIDIA API request failed after retries',

                type:
                  'nvidia_api_error',

                code:
                  lastStatus || 503

              }

            });
        }

        return;
      }

      // ==========================================
      // SSE HEADERS
      // ==========================================

      if (
        clientDisconnected
      ) {
        return;
      }

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

      // ==========================================
      // STREAM STATE
      // ==========================================

      let buffer = '';

      let reasoningOpen =
        false;

      let finished =
        false;

      // ==========================================
      // SEND DONE
      // ==========================================

      function sendDone() {

        if (
          finished
        ) {
          return;
        }

        finished = true;

        // Close unfinished reasoning block.
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

            if (
              !res.writableEnded
            ) {

              res.write(
                `data: ${JSON.stringify(
                  closeChunk
                )}\n\n`
              );
            }

          } catch {
            // Ignore.
          }

          reasoningOpen =
            false;
        }

        try {

          if (
            !res.writableEnded
          ) {

            res.write(
              'data: [DONE]\n\n'
            );

          }

        } catch {
          // Ignore.
        }

        if (
          !res.writableEnded
        ) {
          res.end();
        }
      }

      // ==========================================
      // WRITE SSE
      // ==========================================

      function writeSSE(data) {

        if (
          finished ||
          res.writableEnded ||
          clientDisconnected
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

      // ==========================================
      // PROCESS SSE LINE
      // ==========================================

      function processLine(line) {

        line =
          line.replace(
            /\r$/,
            ''
          );

        if (
          !line.trim()
        ) {
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

        // ========================================
        // DONE
        // ========================================

        if (
          raw === '[DONE]'
        ) {

          sendDone();

          return;
        }

        // ========================================
        // JSON
        // ========================================

        let data;

        try {

          data =
            JSON.parse(
              raw
            );

        } catch (error) {

          console.error(
            '[SSE Parse Error]',
            error.message
          );

          return;
        }

        // ========================================
        // DELTA
        // ========================================

        const choice =
          data?.choices?.[0];

        const delta =
          choice?.delta;

        if (
          !delta
        ) {

          writeSSE(data);

          return;
        }

        // ========================================
        // STEP 3.7 FLASH
        // REASONING OFF
        // ========================================

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

        // ========================================
        // OTHER MODELS
        // REASONING
        // ========================================

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

        // ========================================
        // REASONING
        // ========================================

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

        // ========================================
        // CONTENT
        // ========================================

        if (
          content
        ) {

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

        // ========================================
        // REPLACE CONTENT
        // ========================================

        if (
          output
        ) {

          delta.content =
            output;
        }

        // ========================================
        // REMOVE RAW REASONING
        // ========================================

        delete delta.reasoning;

        delete delta.reasoning_content;

        // ========================================
        // SEND
        // ========================================

        writeSSE(data);
      }

      // ==========================================
      // STREAM DATA
      // ==========================================

      response.data.on(
        'data',
        (chunk) => {

          if (
            finished ||
            res.writableEnded ||
            clientDisconnected
          ) {
            return;
          }

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
              finished ||
              clientDisconnected
            ) {
              break;
            }

            processLine(
              line
            );
          }
        }
      );

      // ==========================================
      // STREAM END
      // ==========================================

      response.data.on(
        'end',
        () => {

          if (
            clientDisconnected
          ) {
            return;
          }

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

      // ==========================================
      // STREAM ERROR
      // ==========================================

      response.data.on(
        'error',
        (error) => {

          console.error(
            '[NVIDIA Stream Error]',
            error.message
          );

          if (
            !finished &&
            !res.writableEnded &&
            !clientDisconnected
          ) {

            writeSSE({

              error: {

                message:
                  error.message ||
                  'NVIDIA stream error',

                type:
                  'stream_error'

              }

            });

            finished = true;

            if (
              !res.writableEnded
            ) {
              res.end();
            }
          }
        }
      );

    } catch (error) {

      // ==========================================
      // SAFE PROXY ERROR
      // ==========================================

      logProxyError(
        error
      );

      // ==========================================
      // IMPORTANT:
      // If the downstream response has already
      // started, NEVER call res.json().
      // ==========================================

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

      // If the client really disconnected,
      // there is nothing left to send.
      if (
        clientDisconnected
      ) {
        return;
      }

      const status =
        error?.response?.status ||
        500;

      const message =
        getSafeErrorMessage(
          error
        );

      return res
        .status(status)
        .json({

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
      'Client model: glm-5.2'
    );

    console.log(
      'NVIDIA model: z-ai/glm-5.2'
    );

    console.log(
      'Step 3.7 Flash reasoning: DISABLED'
    );

    console.log(
      'Retry delays: 10s -> 30s -> 60s -> 60s -> 60s'
    );

    console.log(
      `Maximum retries: ${MAX_RETRIES}`
    );

    console.log(
      'Streaming only: ENABLED'
    );

    console.log(
      '============================================'
    );
  }
);
