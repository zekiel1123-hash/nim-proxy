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
// Retry behavior:
// - Attempt 1
// - 429/5xx/network error -> wait 10s
// - Retry -> wait 30s
// - Retry -> wait 60s
// - Further retries -> wait 60s
//
// IMPORTANT:
// - HTTP 200 immediately stops retrying.
// - Once HTTP 200 is received, the NVIDIA SSE stream
//   is forwarded directly to the client.
// - Step 3.7 Flash reasoning is intentionally disabled.
// - Step 3.7 Flash stray </think> tags are removed.

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();

const PORT =
  process.env.PORT || 3000;

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
// 6 total attempts:
//
// Attempt 1
//   ↓ 429
// wait 10 seconds
//   ↓
// Attempt 2
//   ↓ 429
// wait 30 seconds
//   ↓
// Attempt 3
//   ↓ 429
// wait 60 seconds
//   ↓
// Attempt 4
//   ↓
// wait 60 seconds
//   ↓
// Attempt 5
//   ↓
// wait 60 seconds
//   ↓
// Attempt 6
//
// HTTP 200 NEVER retries.
//

const MAX_ATTEMPTS = 6;

const RETRY_DELAYS = [
  10_000,
  30_000,
  60_000,
  60_000,
  60_000
];

// Retry these HTTP statuses.
const RETRYABLE_STATUS_CODES = new Set([
  429,
  500,
  502,
  503,
  504
]);

// ============================================
// MODEL MAPPING
// ============================================

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
  if (
    typeof model !== 'string'
  ) {
    return false;
  }

  return (
    model === 'step-3.7-flash' ||
    model === 'stepfun-ai/step-3.7-flash' ||
    model.includes('step-3.7-flash')
  );
}

// ============================================
// THINKING CONFIG
// ============================================

function buildThinkingConfig(model) {
  // ============================================
  // STEP 3.7 FLASH
  // REASONING DISABLED
  // ============================================

  if (
    isStep37Flash(model)
  ) {
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
// STEP 3.7 CONTENT CLEANING
// ============================================
//
// Step 3.7 reasoning is disabled.
//
// If upstream nevertheless inserts a stray
// <think> or </think> into content, remove it.
//

function cleanStepContent(content) {
  if (
    typeof content !== 'string'
  ) {
    return '';
  }

  return content
    .replace(
      /<think>\s*/gi,
      ''
    )
    .replace(
      /\s*<\/think>/gi,
      ''
    );
}

// ============================================
// SAFE ERROR MESSAGE
// ============================================

function getSafeErrorMessage(error) {
  // ============================================
  // AXIOS RESPONSE
  // ============================================

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
        if (
          typeof data.error ===
          'string'
        ) {
          return data.error;
        }

        if (
          data.error?.message
        ) {
          return data.error.message;
        }

        if (
          data.message
        ) {
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

  // ============================================
  // NETWORK / AXIOS ERROR
  // ============================================

  if (
    error?.message
  ) {
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
    getSafeErrorMessage(
      error
    );

  console.error(
    '[Proxy Error]',
    status
      ? `HTTP ${status}`
      : 'No HTTP status',
    message
  );
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
        max_attempts:
          MAX_ATTEMPTS,

        delays_seconds:
          RETRY_DELAYS.map(
            (ms) =>
              ms / 1000
          ),

        retryable_statuses:
          Array.from(
            RETRYABLE_STATUS_CODES
          )
      },

      models: {
        'glm-5.2': {
          reasoning: true,
          nim_model:
            'z-ai/glm-5.2'
        },

        'kimi-k2.6': {
          reasoning: true,
          nim_model:
            'moonshotai/kimi-k2.6'
        },

        'deepseek-v4': {
          reasoning: true,
          nim_model:
            'deepseek-ai/deepseek-v4-pro'
        },

        'step-3.7-flash': {
          reasoning: false,
          nim_model:
            'stepfun-ai/step-3.7-flash'
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

      data: Object.keys(
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

    let clientDisconnected =
      false;

    let streamStarted =
      false;

    let finished =
      false;

    let abortController =
      null;

    // ============================================
    // CLIENT DISCONNECT
    // ============================================
    //
    // IMPORTANT:
    //
    // Do NOT use req.on('close') here.
    //
    // The incoming HTTP request can finish while
    // the outgoing SSE response is still active.
    //
    // We monitor the RESPONSE instead.
    //

    res.on(
      'close',
      () => {
        if (
          !res.writableEnded
        ) {
          clientDisconnected =
            true;

          console.log(
            '[Proxy] Client disconnected'
          );

          if (
            abortController
          ) {
            try {
              abortController.abort();
            } catch {
              // Ignore abort error.
            }
          }

          if (
            response?.data &&
            typeof response.data.destroy ===
              'function'
          ) {
            try {
              response.data.destroy();
            } catch {
              // Ignore destroy error.
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

      // ============================================
      // VALIDATE MESSAGES
      // ============================================

      if (
        !Array.isArray(
          messages
        )
      ) {
        return res
          .status(400)
          .json({
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
      // RESOLVE CLIENT MODEL
      // ============================================

      const clientModel =
        typeof model === 'string'
          ? model
          : '';

      const nimModel =
        MODEL_MAPPING[
          clientModel
        ] ||
        FALLBACK_MODEL;

      const step37 =
        isStep37Flash(
          nimModel
        ) ||
        isStep37Flash(
          clientModel
        );

      // ============================================
      // BUILD NIM REQUEST
      // ============================================

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

      // ============================================
      // TOP P
      // ============================================

      if (
        top_p !== undefined &&
        top_p !== null
      ) {
        nimRequest.top_p =
          top_p;
      } else if (
        step37
      ) {
        nimRequest.top_p =
          0.95;
      }

      // ============================================
      // SEED
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
      // LOG REQUEST
      // ============================================

      console.log(
        `[Proxy] ${
          clientModel ||
          'unknown'
        } -> ${nimModel}` +
        (
          step37
            ? ' [REASONING DISABLED]'
            : ''
        )
      );

      // ============================================
      // RETRY LOOP
      // ============================================

      let successfulResponse =
        false;

      let lastError =
        null;

      for (
        let attempt = 1;
        attempt <= MAX_ATTEMPTS;
        attempt++
      ) {
        // ==========================================
        // CLIENT ALREADY GONE
        // ==========================================

        if (
          clientDisconnected
        ) {
          console.log(
            '[Proxy] Client disconnected before NVIDIA request; stopping retry cycle'
          );

          return;
        }

        console.log(
          `[NVIDIA Request] Attempt ${attempt}/${MAX_ATTEMPTS}`
        );

        // New abort controller for every attempt.
        abortController =
          new AbortController();

        try {
          // ========================================
          // NVIDIA REQUEST
          // ========================================

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

                signal:
                  abortController.signal,

                validateStatus:
                  () => true
              }
            );

          console.log(
            `[NVIDIA Response] HTTP ${response.status}`
          );

          // ========================================
          // SUCCESS
          // ========================================
          //
          // THIS IS CRITICAL:
          //
          // Once NVIDIA gives us 200, we NEVER
          // retry. We immediately start consuming
          // the SSE stream.
          //

          if (
            response.status >= 200 &&
            response.status < 300
          ) {
            successfulResponse =
              true;

            break;
          }

          // ========================================
          // NON-RETRYABLE HTTP ERROR
          // ========================================

          if (
            !RETRYABLE_STATUS_CODES.has(
              response.status
            )
          ) {
            let errorBody =
              '';

            try {
              for await (
                const chunk
                of response.data
              ) {
                errorBody +=
                  chunk.toString(
                    'utf8'
                  );

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

            let parsed =
              errorBody;

            try {
              parsed =
                JSON.parse(
                  errorBody
                );
            } catch {
              // Keep string.
            }

            const message =
              typeof parsed ===
              'string'
                ? parsed
                : (
                    parsed?.error
                      ?.message ||
                    parsed?.message ||
                    `NVIDIA API returned HTTP ${response.status}`
                  );

            console.error(
              `[NVIDIA Error] HTTP ${response.status}: ${message}`
            );

            if (
              !res.headersSent &&
              !res.writableEnded
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

          // ========================================
          // RETRYABLE HTTP ERROR
          // ========================================

          let retryBody =
            '';

          try {
            for await (
              const chunk
              of response.data
            ) {
              retryBody +=
                chunk.toString(
                  'utf8'
                );

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
            `NVIDIA API returned HTTP ${response.status}`;

          if (
            retryBody
          ) {
            try {
              const parsed =
                JSON.parse(
                  retryBody
                );

              retryMessage =
                parsed?.error
                  ?.message ||
                parsed?.message ||
                retryMessage;
            } catch {
              retryMessage =
                retryBody;
            }
          }

          lastError =
            new Error(
              retryMessage
            );

          // ========================================
          // LAST ATTEMPT
          // ========================================

          if (
            attempt >=
            MAX_ATTEMPTS
          ) {
            console.error(
              `[NVIDIA Retry] Exhausted all ${MAX_ATTEMPTS} attempts`
            );

            if (
              !res.headersSent &&
              !res.writableEnded
            ) {
              return res
                .status(
                  response.status
                )
                .json({
                  error: {
                    message:
                      retryMessage,

                    type:
                      'nvidia_api_error',

                    code:
                      response.status
                  }
                });
            }

            return;
          }

          // ========================================
          // WAIT BEFORE RETRY
          // ========================================

          const delay =
            RETRY_DELAYS[
              attempt - 1
            ] ??
            60_000;

          console.log(
            `[NVIDIA Retry] HTTP ${response.status}. ` +
            `Retry ${attempt}/${MAX_ATTEMPTS - 1} ` +
            `in ${delay / 1000}s`
          );

          console.log(
            `[NVIDIA Retry] ${retryMessage}`
          );

          // ========================================
          // ABORTABLE WAIT
          // ========================================

          let remaining =
            delay;

          while (
            remaining > 0
          ) {
            if (
              clientDisconnected
            ) {
              console.log(
                '[Proxy] Client disconnected during backoff; stopping retry cycle'
              );

              return;
            }

            const wait =
              Math.min(
                remaining,
                1000
              );

            await sleep(
              wait
            );

            remaining -=
              wait;
          }
        } catch (error) {
          // ========================================
          // ABORT / CLIENT DISCONNECT
          // ========================================

          if (
            clientDisconnected ||
            error?.code ===
              'ERR_CANCELED' ||
            error?.name ===
              'CanceledError'
          ) {
            console.log(
              '[Proxy] Request aborted because client disconnected'
            );

            return;
          }

          lastError =
            error;

          console.error(
            `[NVIDIA Request Error] Attempt ${attempt}:`,
            getSafeErrorMessage(
              error
            )
          );

          // ========================================
          // LAST ATTEMPT
          // ========================================

          if (
            attempt >=
            MAX_ATTEMPTS
          ) {
            break;
          }

          // ========================================
          // NETWORK RETRY
          // ========================================

          const delay =
            RETRY_DELAYS[
              attempt - 1
            ] ??
            60_000;

          console.log(
            `[NVIDIA Retry] Network error. ` +
            `Retry ${attempt}/${MAX_ATTEMPTS - 1} ` +
            `in ${delay / 1000}s`
          );

          // ========================================
          // ABORTABLE WAIT
          // ========================================

          let remaining =
            delay;

          while (
            remaining > 0
          ) {
            if (
              clientDisconnected
            ) {
              console.log(
                '[Proxy] Client disconnected during backoff; stopping retry cycle'
              );

              return;
            }

            const wait =
              Math.min(
                remaining,
                1000
              );

            await sleep(
              wait
            );

            remaining -=
              wait;
          }
        }
      }

      // ============================================
      // RETRY EXHAUSTED
      // ============================================

      if (
        !successfulResponse
      ) {
        if (
          clientDisconnected
        ) {
          return;
        }

        const message =
          lastError?.message ||
          'Unable to obtain a successful response from NVIDIA NIM';

        console.error(
          '[Proxy] Retry cycle exhausted:',
          message
        );

        if (
          !res.headersSent &&
          !res.writableEnded
        ) {
          return res
            .status(503)
            .json({
              error: {
                message,

                type:
                  'nvidia_retry_exhausted',

                code: 503
              }
            });
        }

        return;
      }

      // ============================================
      // HTTP 200 RECEIVED
      // ============================================
      //
      // NO MORE RETRIES.
      //
      // From this point forward, this is purely
      // an SSE streaming operation.
      //

      console.log(
        '[NVIDIA Stream] HTTP 200 received; starting SSE stream'
      );

      streamStarted =
        true;

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

      // Flush headers immediately.
      if (
        typeof res.flushHeaders ===
        'function'
      ) {
        res.flushHeaders();
      }

      // Send an initial SSE comment to establish
      // the connection immediately.
      try {
        res.write(
          ': connected\n\n'
        );
      } catch {
        return;
      }

      console.log(
        '[NVIDIA Stream] Connected'
      );

      // ============================================
      // STREAM STATE
      // ============================================

      let buffer =
        '';

      let reasoningOpen =
        false;

      // ============================================
      // SEND DONE
      // ============================================

      function sendDone() {
        if (
          finished
        ) {
          return;
        }

        finished =
          true;

        // ==========================================
        // CLOSE OPEN THINK BLOCK
        // ==========================================

        if (
          SHOW_REASONING &&
          reasoningOpen &&
          !step37
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

        // ==========================================
        // DONE
        // ==========================================

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

        console.log(
          '[NVIDIA Stream] Complete'
        );
      }

      // ============================================
      // WRITE SSE
      // ============================================

      function writeSSE(data) {
        if (
          finished ||
          clientDisconnected ||
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
      // PROCESS SSE EVENT
      // ============================================

      function processEvent(
        eventText
      ) {
        if (
          finished ||
          clientDisconnected
        ) {
          return;
        }

        const lines =
          eventText.split(
            '\n'
          );

        const dataLines =
          [];

        for (
          let lineIndex = 0;
          lineIndex <
            lines.length;
          lineIndex++
        ) {
          let line =
            lines[
              lineIndex
            ];

          line =
            line.replace(
              /\r$/,
              ''
            );

          if (
            !line ||
            line.startsWith(':')
          ) {
            continue;
          }

          if (
            line.startsWith(
              'data:'
            )
          ) {
            dataLines.push(
              line
                .slice(5)
                .trim()
            );
          }
        }

        if (
          dataLines.length ===
          0
        ) {
          return;
        }

        const raw =
          dataLines.join(
            '\n'
          );

        // ==========================================
        // DONE
        // ==========================================

        if (
          raw === '[DONE]'
        ) {
          sendDone();
          return;
        }

        // ==========================================
        // PARSE JSON
        // ==========================================

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

          console.error(
            '[SSE Raw Data]',
            raw.slice(
              0,
              1000
            )
          );

          return;
        }

        // ==========================================
        // GET CHOICE
        // ==========================================

        const choice =
          data?.choices?.[0];

        const delta =
          choice?.delta;

        // Some OpenAI-compatible SSE events contain
        // usage or other metadata without delta.
        //
        // Forward those untouched.
        //

        if (
          !delta
        ) {
          writeSSE(
            data
          );

          return;
        }

        // ==========================================
        // STEP 3.7 FLASH
        // REASONING DISABLED
        // ==========================================

        if (
          step37
        ) {
          if (
            typeof delta.content ===
            'string'
          ) {
            delta.content =
              cleanStepContent(
                delta.content
              );
          }

          // Never expose upstream reasoning fields.
          delete delta.reasoning;

          delete delta.reasoning_content;

          writeSSE(
            data
          );

          return;
        }

        // ==========================================
        // OTHER MODELS
        // REASONING
        // ==========================================

        const reasoning =
          extractReasoning(
            delta
          );

        const content =
          typeof delta.content ===
          'string'
            ? delta.content
            : '';

        let output =
          '';

        // ==========================================
        // REASONING
        // ==========================================

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

        // ==========================================
        // NORMAL CONTENT
        // ==========================================

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

        // ==========================================
        // CONTENT REPLACEMENT
        // ==========================================

        if (
          output
        ) {
          delta.content =
            output;
        }

        // ==========================================
        // REMOVE RAW REASONING
        // ==========================================

        delete delta.reasoning;

        delete delta.reasoning_content;

        // ==========================================
        // FORWARD EVENT
        // ==========================================

        writeSSE(
          data
        );
      }

      // ============================================
      // NVIDIA STREAM DATA
      // ============================================

      response.data.on(
        'data',
        (chunk) => {
          if (
            finished ||
            clientDisconnected ||
            res.writableEnded
          ) {
            return;
          }

          const text =
            chunk.toString(
              'utf8'
            );

          buffer +=
            text;

          // SSE events are separated by a blank
          // line. Handle both LF and CRLF.
          //
          // Normalize CRLF first.
          buffer =
            buffer.replace(
              /\r\n/g,
              '\n'
            );

          let separatorIndex;

          while (
            (
              separatorIndex =
                buffer.indexOf(
                  '\n\n'
                )
            ) !== -1
          ) {
            const eventText =
              buffer.slice(
                0,
                separatorIndex
              );

            buffer =
              buffer.slice(
                separatorIndex + 2
              );

            processEvent(
              eventText
            );

            if (
              finished ||
              clientDisconnected
            ) {
              break;
            }
          }
        }
      );

      // ============================================
      // NVIDIA STREAM END
      // ============================================

      response.data.on(
        'end',
        () => {
          if (
            finished ||
            clientDisconnected
          ) {
            return;
          }

          // Process any remaining complete/partial
          // event if one exists.
          if (
            buffer.trim()
          ) {
            processEvent(
              buffer
            );
          }

          if (
            !finished
          ) {
            sendDone();
          }
        }
      );

      // ============================================
      // NVIDIA STREAM ERROR
      // ============================================

      response.data.on(
        'error',
        (error) => {
          if (
            clientDisconnected
          ) {
            return;
          }

          console.error(
            '[NVIDIA Stream Error]',
            error.message
          );

          if (
            finished ||
            res.writableEnded
          ) {
            return;
          }

          // Once 200 has been received, retries are
          // intentionally NOT attempted. Send an SSE
          // error to the client instead.
          //

          writeSSE({
            error: {
              message:
                error.message ||
                'NVIDIA stream error',

              type:
                'stream_error'
            }
          });

          finished =
            true;

          if (
            !res.writableEnded
          ) {
            res.end();
          }
        }
      );
    } catch (error) {
      // ============================================
      // SAFE PROXY ERROR
      // ============================================

      if (
        clientDisconnected
      ) {
        return;
      }

      logProxyError(
        error
      );

      // ==========================================
      // STREAM ALREADY STARTED
      // ==========================================

      if (
        streamStarted ||
        res.headersSent ||
        res.writableEnded
      ) {
        if (
          !res.writableEnded
        ) {
          try {
            res.end();
          } catch {
            // Ignore.
          }
        }

        return;
      }

      // ==========================================
      // NORMAL HTTP ERROR
      // ==========================================

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

    res
      .status(404)
      .json({
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
      'Client models:'
    );

    console.log(
      '  glm-5.2 -> z-ai/glm-5.2'
    );

    console.log(
      '  kimi-k2.6 -> moonshotai/kimi-k2.6'
    );

    console.log(
      '  deepseek-v4 -> deepseek-ai/deepseek-v4-pro'
    );

    console.log(
      '  step-3.7-flash -> stepfun-ai/step-3.7-flash'
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
      'Retry delays: 10s -> 30s -> 60s -> 60s -> 60s'
    );

    console.log(
      `Maximum attempts: ${MAX_ATTEMPTS}`
    );

    console.log(
      'Streaming only: ENABLED'
    );

    console.log(
      '============================================'
    );
  }
);
