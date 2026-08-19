// server.js
// OpenAI-compatible NVIDIA NIM proxy
// STREAMING ONLY
//
// Client-facing models:
// - glm-5.2
// - kimi-k2.6
// - deepseek-v4
// - step-3.7-flash
//
// NVIDIA models:
// - glm-5.2       -> z-ai/glm-5.2
// - kimi-k2.6     -> moonshotai/kimi-k2.6
// - deepseek-v4   -> deepseek-ai/deepseek-v4-pro
// - step-3.7-flash -> stepfun-ai/step-3.7-flash
//
// Step 3.7 Flash:
// - Thinking/reasoning intentionally DISABLED.
// - No thinking parameters are sent.
// - No <think> tags are generated.
// - Stray <think> / </think> tags are removed.
//
// Retry behavior:
// - HTTP 429 / transient NVIDIA errors are retried.
// - Retry delays: 10s -> 30s -> 60s -> 60s -> 60s
// - Maximum 6 total NVIDIA attempts.
// - Client disconnects do not get reported as the primary
//   NVIDIA error.
// - API keys and sensitive headers are never logged.
//

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
// RETRY CONFIG
// ============================================
//
// Total attempts = 6:
//
// Attempt 1
//   ↓ 429
// wait 10 seconds
//
// Attempt 2
//   ↓ 429
// wait 30 seconds
//
// Attempt 3
//   ↓ 429
// wait 60 seconds
//
// Attempt 4
//   ↓ 429
// wait 60 seconds
//
// Attempt 5
//   ↓ 429
// wait 60 seconds
//
// Attempt 6
//   ↓
// final failure
//
// The first three delays are intentionally
// 10s, 30s, and 60s. Additional retries stay
// at 60s rather than becoming increasingly long.
//

const MAX_ATTEMPTS = 6;

const RETRY_DELAYS_MS = [
  10 * 1000,
  30 * 1000,
  60 * 1000,
  60 * 1000,
  60 * 1000
];

// Retry these NVIDIA HTTP statuses.
const RETRYABLE_STATUS_CODES = new Set([
  408, // Request Timeout
  409, // Conflict
  425, // Too Early
  429, // Too Many Requests
  500, // Internal Server Error
  502, // Bad Gateway
  503, // Service Unavailable
  504  // Gateway Timeout
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
  return (
    typeof model === 'string' &&
    (
      model === 'step-3.7-flash' ||
      model === 'stepfun-ai/step-3.7-flash' ||
      model.includes('step-3.7-flash')
    )
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
// CLEAN STEP CONTENT
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
  // Explicit client cancellation
  if (
    error?.code === 'CLIENT_DISCONNECTED'
  ) {
    return 'Client disconnected';
  }

  // Axios response
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
// SLEEP
// ============================================
//
// Promise-based delay used by retry logic.
//
// The promise can be cancelled when the client
// disconnects so the proxy doesn't sit around
// unnecessarily for the entire retry period.
//

function sleep(ms, shouldCancel) {
  return new Promise(
    (resolve, reject) => {
      let settled = false;

      const timer =
        setTimeout(() => {
          if (settled) {
            return;
          }

          settled = true;
          resolve();
        }, ms);

      if (
        typeof shouldCancel !==
        'function'
      ) {
        return;
      }

      const checkCancellation =
        () => {
          if (
            settled
          ) {
            return;
          }

          if (
            shouldCancel()
          ) {
            settled = true;

            clearTimeout(
              timer
            );

            const error =
              new Error(
                'Client disconnected'
              );

            error.code =
              'CLIENT_DISCONNECTED';

            reject(error);
          }
        };

      // Check periodically while waiting.
      //
      // This keeps the retry delay slow while
      // still allowing a disconnected client to
      // cancel the request promptly.

      const interval =
        setInterval(
          () => {
            if (
              settled
            ) {
              clearInterval(
                interval
              );

              return;
            }

            checkCancellation();
          },
          1000
        );

      // Clean up the interval when the timer
      // completes.
      const originalResolve =
        resolve;

      void originalResolve;
    }
  );
}

// ============================================
// BETTER CANCELLABLE DELAY
// ============================================

function waitForRetry(
  ms,
  isClientDisconnected
) {
  return new Promise(
    (resolve, reject) => {
      let finished = false;

      const timer =
        setTimeout(() => {
          if (finished) {
            return;
          }

          finished = true;

          clearInterval(
            interval
          );

          resolve();
        }, ms);

      const interval =
        setInterval(() => {
          if (
            isClientDisconnected()
          ) {
            if (finished) {
              return;
            }

            finished = true;

            clearTimeout(
              timer
            );

            clearInterval(
              interval
            );

            const error =
              new Error(
                'Client disconnected'
              );

            error.code =
              'CLIENT_DISCONNECTED';

            reject(error);
          }
        }, 1000);
    }
  );
}

// ============================================
// READ NVIDIA ERROR STREAM
// ============================================

async function readErrorStream(stream) {
  let body = '';

  try {
    for await (
      const chunk of stream
    ) {
      body += chunk.toString();

      if (
        body.length >= 100000
      ) {
        break;
      }
    }
  } catch {
    // Ignore secondary stream-read errors.
  }

  return body;
}

// ============================================
// PARSE NVIDIA ERROR
// ============================================

function parseNvidiaErrorBody(
  errorBody
) {
  if (!errorBody) {
    return {
      raw: '',
      message: ''
    };
  }

  try {
    const parsed =
      JSON.parse(errorBody);

    const message =
      parsed?.error?.message ||
      (
        typeof parsed?.error ===
        'string'
          ? parsed.error
          : ''
      ) ||
      parsed?.message ||
      '';

    return {
      raw: parsed,
      message
    };
  } catch {
    return {
      raw: errorBody,
      message: errorBody
    };
  }
}

// ============================================
// CREATE NVIDIA REQUEST
// ============================================

async function requestNvidiaWithRetry(
  nimRequest,
  isClientDisconnected
) {
  let lastError = null;

  for (
    let attempt = 1;
    attempt <= MAX_ATTEMPTS;
    attempt++
  ) {
    // ==========================================
    // CLIENT DISCONNECTED
    // ==========================================

    if (
      isClientDisconnected()
    ) {
      const error =
        new Error(
          'Client disconnected'
        );

      error.code =
        'CLIENT_DISCONNECTED';

      throw error;
    }

    console.log(
      `[NVIDIA Request] Attempt ${attempt}/${MAX_ATTEMPTS}`
    );

    try {
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
              'stream',

            timeout: 0,

            validateStatus:
              () => true
          }
        );

      // ==========================================
      // SUCCESS
      // ==========================================

      if (
        response.status >= 200 &&
        response.status < 300
      ) {
        console.log(
          `[NVIDIA Request] Success on attempt ${attempt}/${MAX_ATTEMPTS}`
        );

        return response;
      }

      // ==========================================
      // READ ERROR RESPONSE
      // ==========================================

      const errorBody =
        await readErrorStream(
          response.data
        );

      const parsed =
        parseNvidiaErrorBody(
          errorBody
        );

      const status =
        response.status;

      const message =
        parsed.message ||
        `NVIDIA API returned HTTP ${status}`;

      // Create a normal Error object.
      // Do NOT attach the stream to it.
      //
      // This avoids circular JSON errors.

      const error =
        new Error(message);

      error.name =
        'NvidiaApiError';

      error.status =
        status;

      error.response = {
        status,
        data:
          parsed.raw
      };

      lastError =
        error;

      // ==========================================
      // NON-RETRYABLE ERROR
      // ==========================================

      if (
        !RETRYABLE_STATUS_CODES.has(
          status
        )
      ) {
        console.error(
          `[NVIDIA Error] HTTP ${status}: ${message}`
        );

        throw error;
      }

      // ==========================================
      // FINAL ATTEMPT
      // ==========================================

      if (
        attempt >= MAX_ATTEMPTS
      ) {
        console.error(
          `[NVIDIA Retry] Exhausted ${MAX_ATTEMPTS} attempts. ` +
          `Final HTTP ${status}: ${message}`
        );

        throw error;
      }

      // ==========================================
      // RETRY DELAY
      // ==========================================

      const delay =
        RETRY_DELAYS_MS[
          attempt - 1
        ] ??
        60 * 1000;

      const seconds =
        Math.round(
          delay / 1000
        );

      console.warn(
        `[NVIDIA Retry] HTTP ${status}. ` +
        `Retry ${attempt}/${MAX_ATTEMPTS - 1} in ${seconds}s`
      );

      console.warn(
        `[NVIDIA Retry] ${message}`
      );

      await waitForRetry(
        delay,
        isClientDisconnected
      );
    } catch (error) {
      // ==========================================
      // CLIENT DISCONNECT
      // ==========================================

      if (
        error?.code ===
        'CLIENT_DISCONNECTED'
      ) {
        throw error;
      }

      // ==========================================
      // OUR NVIDIA HTTP ERROR
      // ==========================================
      //
      // We deliberately throw this above after
      // exhausting/non-retryable responses.
      //

      if (
        error?.name ===
        'NvidiaApiError'
      ) {
        throw error;
      }

      // ==========================================
      // NETWORK / AXIOS ERROR
      // ==========================================

      lastError =
        error;

      const retryableNetworkError =
        !error?.response ||
        RETRYABLE_STATUS_CODES.has(
          error?.response?.status
        );

      if (
        !retryableNetworkError
      ) {
        throw error;
      }

      // ==========================================
      // FINAL NETWORK ATTEMPT
      // ==========================================

      if (
        attempt >= MAX_ATTEMPTS
      ) {
        throw error;
      }

      const delay =
        RETRY_DELAYS_MS[
          attempt - 1
        ] ??
        60 * 1000;

      const seconds =
        Math.round(
          delay / 1000
        );

      console.warn(
        `[NVIDIA Retry] Network error. ` +
        `Retry ${attempt}/${MAX_ATTEMPTS - 1} in ${seconds}s`
      );

      console.warn(
        `[NVIDIA Retry] ${
          error?.message ||
          'Network error'
        }`
      );

      await waitForRetry(
        delay,
        isClientDisconnected
      );
    }
  }

  throw (
    lastError ||
    new Error(
      'NVIDIA request failed'
    )
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
        max_attempts:
          MAX_ATTEMPTS,

        delays_seconds:
          RETRY_DELAYS_MS.map(
            (ms) =>
              Math.round(
                ms / 1000
              )
          )
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

    // ==========================================
    // CLIENT DISCONNECT TRACKING
    // ==========================================

    const onClientClose =
      () => {
        // IMPORTANT:
        //
        // req.close can also occur during normal
        // request lifecycle handling. We only mark
        // the request disconnected when the
        // response has not completed.
        //
        // This flag is checked by the retry loop
        // and streaming code.

        if (
          !res.writableEnded
        ) {
          clientDisconnected =
            true;
        }
      };

    req.on(
      'close',
      onClientClose
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
      // BUILD NVIDIA REQUEST
      // ==========================================

      const nimRequest = {
        model: nimModel,

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
      } else if (
        step37
      ) {
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
        `[Request] ${model || 'unknown'} -> ${nimModel}` +
        (
          step37
            ? ' [REASONING DISABLED]'
            : ''
        )
      );

      // ==========================================
      // NVIDIA REQUEST + RETRY
      // ==========================================

      response =
        await requestNvidiaWithRetry(
          nimRequest,
          () =>
            clientDisconnected
        );

      // ==========================================
      // IF CLIENT DISCONNECTED DURING RETRIES
      // ==========================================

      if (
        clientDisconnected
      ) {
        console.warn(
          '[Proxy] Client disconnected during NVIDIA retry cycle'
        );

        // Destroy the NVIDIA stream if one
        // somehow became available.
        if (
          response?.data &&
          typeof response.data.destroy ===
            'function'
        ) {
          response.data.destroy();
        }

        return;
      }

      // ==========================================
      // SSE HEADERS
      // ==========================================

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

        // Close an unfinished reasoning block
        // for reasoning-enabled models.

        if (
          SHOW_REASONING &&
          reasoningOpen &&
          !step37
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
            // Client may be gone.
          }

          reasoningOpen =
            false;
        }

        try {
          res.write(
            'data: [DONE]\n\n'
          );
        } catch {
          // Client may be gone.
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
        // PARSE
        // ==========================================

        let data;

        try {
          data =
            JSON.parse(raw);
        } catch (error) {
          console.error(
            '[SSE Parse Error]',
            error.message
          );

          return;
        }

        // ==========================================
        // CHOICE / DELTA
        // ==========================================

        const choice =
          data?.choices?.[0];

        const delta =
          choice?.delta;

        if (!delta) {
          writeSSE(data);
          return;
        }

        // ==========================================
        // STEP 3.7 FLASH
        // REASONING DISABLED
        // ==========================================

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

        let output = '';

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
        // CONTENT
        // ==========================================

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

        // ==========================================
        // REPLACE CONTENT
        // ==========================================

        if (output) {
          delta.content =
            output;
        }

        // ==========================================
        // REMOVE RAW REASONING
        // ==========================================

        delete delta.reasoning;

        delete delta.reasoning_content;

        // ==========================================
        // SEND
        // ==========================================

        writeSSE(data);
      }

      // ==========================================
      // NVIDIA STREAM DATA
      // ==========================================

      response.data.on(
        'data',
        (chunk) => {
          if (
            finished ||
            res.writableEnded
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

      // ==========================================
      // STREAM END
      // ==========================================

      response.data.on(
        'end',
        () => {
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
            !res.writableEnded
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

            if (
              !res.writableEnded
            ) {
              res.end();
            }

            finished = true;
          }
        }
      );

      // ==========================================
      // CLIENT DISCONNECT DURING STREAM
      // ==========================================

      req.on(
        'close',
        () => {
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
      // ==========================================
      // CLIENT DISCONNECTED
      // ==========================================
      //
      // Do NOT turn this into:
      //
      // [Proxy Error] No HTTP status
      // Client disconnected
      //
      // when it happened during a 429 retry.
      //

      if (
        error?.code ===
        'CLIENT_DISCONNECTED'
      ) {
        console.warn(
          '[Proxy] Client disconnected; stopping retry cycle'
        );

        return;
      }

      // ==========================================
      // SAFE PROXY ERROR
      // ==========================================

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
        error?.status ||
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
            error?.name ===
            'NvidiaApiError'
              ? 'nvidia_api_error'
              : 'invalid_request_error',

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
      'Retry delays: 10s -> 30s -> 60s -> 60s -> 60s'
    );

    console.log(
      `Maximum NVIDIA attempts: ${MAX_ATTEMPTS}`
    );

    console.log(
      'Streaming only: ENABLED'
    );

    console.log(
      '============================================'
    );
  }
);
