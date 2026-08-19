// server.js
// OpenAI-compatible NVIDIA NIM proxy
// STREAMING ONLY
//
// Models:
// - glm-5.1          -> z-ai/glm-5.2
// - kimi-k2.6        -> moonshotai/kimi-k2.6
// - deepseek-v4     -> deepseek-ai/deepseek-v4-pro
// - step-3.7-flash  -> stepfun-ai/step-3.7-flash
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
//
// RETRY / BACKOFF:
// - Retries transient failures before streaming starts.
// - 10 seconds -> 30 seconds -> 1 minute -> 2 minutes -> 4 minutes.
// - Honors Retry-After when NVIDIA provides it.
// - Does NOT retry 400/401/403/404/etc.
// - Once streaming has started, the request is never retried.

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
// RETRY / EXPONENTIAL BACKOFF CONFIG
// ============================================
//
// Retry schedule:
//
// Attempt 1 -> 10 seconds
// Attempt 2 -> 30 seconds
// Attempt 3 -> 1 minute
// Attempt 4 -> 2 minutes
// Attempt 5 -> 4 minutes
//
// MAX_RETRIES means retries AFTER the initial
// request.
//
// Therefore MAX_RETRIES = 5 means:
// initial request + 5 retries = 6 total attempts.
//

const MAX_RETRIES = 5;

const BACKOFF_DELAYS_MS = [
  10 * 1000,       // 10 seconds
  30 * 1000,       // 30 seconds
  60 * 1000,       // 1 minute
  2 * 60 * 1000,   // 2 minutes
  4 * 60 * 1000    // 4 minutes
];

// ============================================
// RETRYABLE HTTP STATUS CODES
// ============================================

const RETRYABLE_STATUS_CODES = new Set([
  408,
  409,
  425,
  429,
  500,
  502,
  503,
  504
]);

// ============================================
// RETRYABLE NETWORK ERRORS
// ============================================

const RETRYABLE_NETWORK_CODES = new Set([
  'ECONNRESET',
  'ECONNABORTED',
  'ETIMEDOUT',
  'EPIPE',
  'ENETRESET',
  'ENETUNREACH',
  'EAI_AGAIN',
  'ECONNREFUSED',
  'ERR_NETWORK'
]);

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
      model.includes(
        'stepfun-ai/step-3.7-flash'
      )
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
// No chat_template_kwargs.
// No reasoning_effort.
// No thinking flag.
//
// This prevents the proxy from attempting to
// force reasoning output from Step 3.7 Flash.
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
// BACKOFF HELPERS
// ============================================

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

// ============================================
// RETRY DELAY
// ============================================
//
// retryAttempt:
//   0 = first retry -> 10 seconds
//   1 = second retry -> 30 seconds
//   2 = third retry -> 1 minute
//   3 = fourth retry -> 2 minutes
//   4 = fifth retry -> 4 minutes
//

function getBackoffDelay(retryAttempt) {
  const index = Math.min(
    retryAttempt,
    BACKOFF_DELAYS_MS.length - 1
  );

  return BACKOFF_DELAYS_MS[index];
}

// ============================================
// RETRY-AFTER PARSER
// ============================================
//
// NVIDIA may send:
//
// Retry-After: 60
//
// or:
//
// Retry-After: <HTTP date>
//
// Returns milliseconds or null.
//

function getRetryAfterMs(response) {
  const retryAfter =
    response?.headers?.['retry-after'];

  if (!retryAfter) {
    return null;
  }

  // Numeric seconds
  const seconds =
    Number(retryAfter);

  if (
    Number.isFinite(seconds) &&
    seconds >= 0
  ) {
    return seconds * 1000;
  }

  // HTTP date
  const date =
    Date.parse(retryAfter);

  if (
    Number.isFinite(date)
  ) {
    const delay =
      date - Date.now();

    return Math.max(
      0,
      delay
    );
  }

  return null;
}

// ============================================
// RETRY DECISION
// ============================================

function isRetryableError(error) {
  const status =
    error?.response?.status;

  if (
    status &&
    RETRYABLE_STATUS_CODES.has(
      status
    )
  ) {
    return true;
  }

  const code =
    error?.code;

  if (
    code &&
    RETRYABLE_NETWORK_CODES.has(
      code
    )
  ) {
    return true;
  }

  // Axios timeout
  if (
    code === 'ETIMEDOUT' ||
    code === 'ECONNABORTED'
  ) {
    return true;
  }

  return false;
}

// ============================================
// SAFE ERROR SERIALIZATION
// ============================================
//
// NEVER send Axios response.data directly into
// res.json().
//
// When responseType = "stream", response.data
// can be a stream object and may contain circular
// references.
//

function getSafeErrorMessage(error) {
  // Axios response exists
  if (error?.response) {
    const status =
      error.response.status;

    const data =
      error.response.data;

    // String response
    if (
      typeof data === 'string'
    ) {
      return data;
    }

    // Normal JSON object
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

  // Axios/network error
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
// READ NVIDIA ERROR STREAM
// ============================================

async function readErrorStream(stream) {
  let errorBody = '';

  try {
    for await (
      const chunk of stream
    ) {
      errorBody +=
        chunk.toString();

      // Prevent pathological
      // error responses.
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

  return errorBody;
}

// ============================================
// PARSE ERROR BODY
// ============================================

function parseErrorBody(errorBody) {
  if (!errorBody) {
    return '';
  }

  try {
    return JSON.parse(
      errorBody
    );
  } catch {
    return errorBody;
  }
}

// ============================================
// GET ERROR MESSAGE FROM BODY
// ============================================

function getErrorBodyMessage(
  parsedError,
  status
) {
  if (
    typeof parsedError === 'string'
  ) {
    return parsedError;
  }

  if (
    parsedError &&
    typeof parsedError === 'object'
  ) {
    return (
      parsedError?.error?.message ||
      (
        typeof parsedError.error ===
        'string'
          ? parsedError.error
          : null
      ) ||
      parsedError?.message ||
      `NVIDIA API returned HTTP ${status}`
    );
  }

  return (
    `NVIDIA API returned HTTP ${status}`
  );
}

// ============================================
// MAKE NVIDIA REQUEST WITH RETRIES
// ============================================
//
// IMPORTANT:
//
// This function only retries BEFORE a successful
// streaming response is returned.
//
// Once a 2xx streaming response is returned,
// streaming begins and there are NO retries.
//

async function makeNvidiaRequest(
  nimRequest,
  req
) {
  let retryCount = 0;

  while (true) {
    // If the client disconnected while we were
    // waiting during backoff, stop immediately.
    if (req.destroyed) {
      const error =
        new Error(
          'Client disconnected'
        );

      error.code =
        'CLIENT_DISCONNECTED';

      throw error;
    }

    try {
      console.log(
        `[NVIDIA Request] Attempt ${
          retryCount + 1
        }/${MAX_RETRIES + 1}`
      );

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

            // We inspect the status ourselves
            // so retry logic can make the decision.
            validateStatus:
              () => true,

            // This controls connection/response
            // timeout before streaming starts.
            timeout: 120000
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
          `[NVIDIA Request] Success HTTP ${response.status}`
        );

        // IMPORTANT:
        // Streaming has not been processed yet,
        // but this is now a successful stream.
        // Do not retry after returning it.
        return response;
      }

      // ==========================================
      // NVIDIA HTTP ERROR
      // ==========================================

      const status =
        response.status;

      const retryable =
        RETRYABLE_STATUS_CODES.has(
          status
        );

      // Honor Retry-After if provided.
      const retryAfterMs =
        getRetryAfterMs(
          response
        );

      const errorBody =
        await readErrorStream(
          response.data
        );

      const parsedError =
        parseErrorBody(
          errorBody
        );

      const message =
        getErrorBodyMessage(
          parsedError,
          status
        );

      // ==========================================
      // NO RETRIES LEFT
      // ==========================================

      if (
        !retryable ||
        retryCount >= MAX_RETRIES
      ) {
        const error =
          new Error(
            message
          );

        error.response = {
          status,
          data: parsedError,
          headers:
            response.headers
        };

        error.isNvidiaError =
          true;

        throw error;
      }

      // ==========================================
      // CALCULATE BACKOFF
      // ==========================================

      const configuredDelay =
        getBackoffDelay(
          retryCount
        );

      const delay =
        retryAfterMs !== null
          ? Math.max(
              configuredDelay,
              retryAfterMs
            )
          : configuredDelay;

      console.warn(
        `[NVIDIA Retry] HTTP ${status}. ` +
        `Retry ${retryCount + 1}/${MAX_RETRIES} ` +
        `in ${formatDuration(delay)}`
      );

      if (message) {
        console.warn(
          `[NVIDIA Retry] ${message}`
        );
      }

      // Destroy the consumed error stream
      // before sleeping/retrying.
      try {
        if (
          response.data &&
          typeof response.data.destroy ===
            'function'
        ) {
          response.data.destroy();
        }
      } catch {
        // Ignore.
      }

      await sleepWithDisconnectCheck(
        delay,
        req
      );

      retryCount++;
    } catch (error) {
      // ==========================================
      // REQUEST / NETWORK ERROR
      // ==========================================

      if (
        error?.code ===
        'CLIENT_DISCONNECTED'
      ) {
        throw error;
      }

      const retryable =
        isRetryableError(
          error
        );

      // If this is one of our already-processed
      // NVIDIA HTTP errors, don't accidentally
      // treat it as a network retry separately.
      const status =
        error?.response?.status;

      // ==========================================
      // NO RETRY
      // ==========================================

      if (
        !retryable ||
        retryCount >= MAX_RETRIES
      ) {
        throw error;
      }

      // ==========================================
      // CALCULATE NETWORK BACKOFF
      // ==========================================

      const configuredDelay =
        getBackoffDelay(
          retryCount
        );

      const retryAfterMs =
        getRetryAfterMs(
          error?.response
        );

      const delay =
        retryAfterMs !== null
          ? Math.max(
              configuredDelay,
              retryAfterMs
            )
          : configuredDelay;

      console.warn(
        `[NVIDIA Retry] ` +
        `${status ? `HTTP ${status}` : error.code || 'network error'}. ` +
        `Retry ${retryCount + 1}/${MAX_RETRIES} ` +
        `in ${formatDuration(delay)}`
      );

      await sleepWithDisconnectCheck(
        delay,
        req
      );

      retryCount++;
    }
  }
}

// ============================================
// FORMAT DURATION
// ============================================

function formatDuration(ms) {
  const totalSeconds =
    Math.ceil(ms / 1000);

  if (
    totalSeconds < 60
  ) {
    return `${totalSeconds}s`;
  }

  const minutes =
    Math.floor(
      totalSeconds / 60
    );

  const seconds =
    totalSeconds % 60;

  if (seconds === 0) {
    return `${minutes}m`;
  }

  return `${minutes}m ${seconds}s`;
}

// ============================================
// SLEEP WITH CLIENT DISCONNECT CHECK
// ============================================

function sleepWithDisconnectCheck(
  ms,
  req
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

          req.removeListener(
            'close',
            onClose
          );

          resolve();
        }, ms);

      function onClose() {
        if (finished) {
          return;
        }

        finished = true;

        clearTimeout(timer);

        const error =
          new Error(
            'Client disconnected during retry backoff'
          );

        error.code =
          'CLIENT_DISCONNECTED';

        reject(error);
      }

      req.once(
        'close',
        onClose
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
          '1m',
          '2m',
          '4m'
        ],

        retryable_status_codes:
          Array.from(
            RETRYABLE_STATUS_CODES
          )
      },

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
        isStep37Flash(nimModel);

      // ==========================================
      // BUILD BASE REQUEST
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
      // OPTIONAL TOP_P
      // ==========================================

      if (
        top_p !== undefined &&
        top_p !== null
      ) {
        nimRequest.top_p =
          top_p;
      } else if (step37) {
        // Step 3.7 Flash default from the
        // configured NVIDIA example.
        nimRequest.top_p =
          0.95;
      }

      // ==========================================
      // OPTIONAL SEED
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
      // DEBUG INFORMATION
      // ==========================================

      console.log(
        `[Request] ${
          model || 'unknown'
        } -> ${nimModel}` +
        `${
          step37
            ? ' [REASONING DISABLED]'
            : ''
        }`
      );

      // ==========================================
      // NVIDIA REQUEST WITH RETRIES
      // ==========================================

      response =
        await makeNvidiaRequest(
          nimRequest,
          req
        );

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
        if (finished) {
          return;
        }

        finished = true;

        // Only models with reasoning enabled
        // can have an open <think> block.
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

        if (
          !res.writableEnded
        ) {
          res.end();
        }
      }

      // ==========================================
      // WRITE SSE DATA
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

        // Ignore blank lines.
        if (!line.trim()) {
          return;
        }

        // Ignore SSE comments.
        if (
          line.startsWith(':')
        ) {
          return;
        }

        // NVIDIA sends SSE data lines.
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
        // PARSE JSON
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
        // GET DELTA
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
        //
        // Do NOT create <think>.
        // Do NOT close/open reasoning.
        // Do NOT expose reasoning fields.
        //
        // If Step 3.7 Flash puts a stray
        // </think> into content, remove it.
        //

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
        // REASONING PROCESSING
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
        // NORMAL CONTENT
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
        // SEND TO CLIENT
        // ==========================================

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
            if (finished) {
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

      // ==========================================
      // CLIENT DISCONNECT
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
      // SAFE PROXY ERROR
      // ==========================================

      logProxyError(error);

      // If streaming has already begun,
      // do NOT attempt res.json().
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

      // ==========================================
      // CLIENT DISCONNECTED
      // ==========================================

      if (
        error?.code ===
        'CLIENT_DISCONNECTED'
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

      return res.status(
        status
      ).json({
        error: {
          message,

          type:
            error?.isNvidiaError
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
      'Streaming only: ENABLED'
    );

    console.log(
      'Retry/backoff: 10s -> 30s -> 1m -> 2m -> 4m'
    );

    console.log(
      `Maximum retries: ${MAX_RETRIES}`
    );

    console.log(
      '============================================'
    );
  }
);
