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
// NVIDIA models:
// - glm-5.2       -> z-ai/glm-5.2
// - kimi-k2.6     -> moonshotai/kimi-k2.6
// - deepseek-v4   -> deepseek-ai/deepseek-v4-pro
// - step-3.7-flash -> stepfun-ai/step-3.7-flash
//
// RETRY BEHAVIOR:
// - 200: stream immediately
// - 429: internally retry without sending anything to client
// - Retry delays:
//     10 seconds
//     30 seconds
//     60 seconds
//     120 seconds
//     240 seconds
// - Maximum 6 total attempts
//
// STEP 3.7 FLASH:
// - Thinking/reasoning intentionally disabled
// - No chat_template_kwargs
// - No reasoning_effort
// - Stray </think> removed
//
// OTHER MODELS:
// - Reasoning remains enabled when SHOW_REASONING = true

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

// ============================================
// REASONING CONFIG
// ============================================

// Display reasoning in <think> blocks
const SHOW_REASONING = true;

// Enable model-specific thinking parameters
const ENABLE_THINKING_MODE = true;

// DeepSeek reasoning effort
const REASONING_EFFORT = 'low';

// ============================================
// RETRY CONFIG
// ============================================

// Maximum number of total NVIDIA attempts.
//
// Attempt 1 = original request
// Attempts 2-6 = retries
//
const MAX_ATTEMPTS = 6;

// Slow exponential-style backoff.
//
// Retry 1: 10 seconds
// Retry 2: 30 seconds
// Retry 3: 60 seconds
// Retry 4: 120 seconds
// Retry 5: 240 seconds
//
const RETRY_DELAYS = [
  10_000,
  30_000,
  60_000,
  120_000,
  240_000
];

// ============================================
// MODEL MAPPING
// ============================================
//
// IMPORTANT:
// The client-facing name is also glm-5.2.
//
// There is NO glm-5.1 client model anymore.
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
      ''
    );
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
// READ NVIDIA ERROR STREAM
// ============================================
//
// When NVIDIA returns 429, Axios gives us a
// response stream.
//
// We MUST consume the stream before retrying.
//
// Importantly, this error response is NEVER
// sent to the client.
//

async function readErrorStream(stream) {

  let body = '';

  if (!stream) {
    return body;
  }

  try {

    for await (
      const chunk of stream
    ) {

      body +=
        chunk.toString(
          'utf8'
        );

      // Prevent enormous error bodies.
      if (
        body.length >
        100000
      ) {
        break;
      }
    }

  } catch {
    // Ignore errors while consuming
    // an already-failed response.
  }

  return body;
}

// ============================================
// PARSE NVIDIA ERROR BODY
// ============================================

function parseErrorBody(body) {

  if (
    !body
  ) {
    return '';
  }

  try {

    const parsed =
      JSON.parse(body);

    if (
      parsed?.error?.message
    ) {
      return parsed.error.message;
    }

    if (
      parsed?.message
    ) {
      return parsed.message;
    }

    return JSON.stringify(
      parsed
    );

  } catch {

    return body;
  }
}

// ============================================
// NVIDIA REQUEST WITH INTERNAL RETRIES
// ============================================
//
// THIS IS THE IMPORTANT PART.
//
// The client does not see the retry.
//
// We do NOT:
// - set SSE headers
// - write data
// - write [DONE]
// - call res.end()
//
// until NVIDIA gives us a successful 2xx
// response.
//
// Only HTTP 429 is retried.
//

async function requestNvidiaWithRetry(
  nimRequest
) {

  let lastError = null;

  for (
    let attempt = 1;
    attempt <= MAX_ATTEMPTS;
    attempt++
  ) {

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

            // IMPORTANT:
            //
            // Do not let Axios throw automatically
            // on 429.
            //
            // We need to inspect the status and
            // retry internally.
            validateStatus:
              () => true
          }
        );

      // ============================================
      // SUCCESS
      // ============================================

      if (
        response.status >= 200 &&
        response.status < 300
      ) {

        console.log(
          `[NVIDIA Request] HTTP ${response.status} - streaming`
        );

        return response;
      }

      // ============================================
      // RATE LIMITED
      // ============================================

      if (
        response.status === 429
      ) {

        const errorBody =
          await readErrorStream(
            response.data
          );

        const errorMessage =
          parseErrorBody(
            errorBody
          );

        // No more retries.
        if (
          attempt >=
          MAX_ATTEMPTS
        ) {

          console.error(
            `[NVIDIA Retry] HTTP 429. ` +
            `Maximum attempts (${MAX_ATTEMPTS}) reached.`
          );

          const finalError =
            new Error(
              errorMessage ||
              'NVIDIA API rate limit exceeded'
            );

          finalError.response = {
            status: 429,
            data:
              errorMessage
          };

          throw finalError;
        }

        const delay =
          RETRY_DELAYS[
            attempt - 1
          ];

        console.warn(
          `[NVIDIA Retry] HTTP 429. ` +
          `Retry ${attempt}/${MAX_ATTEMPTS - 1} ` +
          `in ${delay / 1000}s`
        );

        if (
          errorMessage
        ) {
          console.warn(
            `[NVIDIA Retry] ${errorMessage}`
          );
        }

        // ==========================================
        // INTERNAL WAIT
        // ==========================================
        //
        // NOTHING is sent to the client here.
        //
        // No SSE headers.
        // No JSON.
        // No [DONE].
        //
        await sleep(
          delay
        );

        continue;
      }

      // ============================================
      // OTHER NVIDIA HTTP ERROR
      // ============================================
      //
      // Do NOT retry these.
      //

      const errorBody =
        await readErrorStream(
          response.data
        );

      const errorMessage =
        parseErrorBody(
          errorBody
        );

      console.error(
        `[NVIDIA Error] HTTP ${response.status}:`,
        errorMessage ||
          'Unknown NVIDIA error'
      );

      const httpError =
        new Error(
          errorMessage ||
          `NVIDIA API returned HTTP ${response.status}`
        );

      httpError.response = {
        status:
          response.status,

        data:
          errorMessage
      };

      throw httpError;

    } catch (error) {

      lastError =
        error;

      // ==========================================
      // IMPORTANT:
      // Axios/network errors are NOT retried here.
      //
      // We only retry an actual HTTP 429 response.
      // ==========================================

      if (
        error?.response?.status ===
        429
      ) {

        // This should normally already have
        // been handled above, but keep this
        // protection in place.
        continue;
      }

      throw error;
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

      streaming_only:
        true,

      reasoning_display:
        SHOW_REASONING,

      thinking_mode:
        ENABLE_THINKING_MODE,

      fallback_model:
        FALLBACK_MODEL,

      retry: {
        enabled: true,

        retries:
          MAX_ATTEMPTS - 1,

        delays:
          RETRY_DELAYS.map(
            (ms) =>
              `${ms / 1000}s`
          ),

        only_status:
          429
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

    // This is ONLY used after a successful
    // NVIDIA response has started streaming.
    let streamingStarted =
      false;

    let clientAborted =
      false;

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

        return res.status(
          400
        ).json({

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

        model:
          nimModel,

        messages,

        temperature:
          temperature ??
          1.0,

        max_tokens:
          max_tokens ??
          (
            step37
              ? 16384
              : 4096
          ),

        stream:
          true
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
      // LOG REQUEST
      // ==========================================

      console.log(
        `[Proxy] ${model || 'unknown'} -> ${nimModel}` +
        `${step37
          ? ' [REASONING DISABLED]'
          : ''}`
      );

      // ==========================================
      // CLIENT ABORT HANDLING
      // ==========================================
      //
      // IMPORTANT:
      //
      // DO NOT use req.on('close') here.
      //
      // A request can emit "close" after its
      // request body has completed, even though
      // the client is still waiting for the
      // response.
      //
      // That was causing the retry cycle to be
      // incorrectly cancelled.
      //
      // "aborted" is the event we care about
      // for the incoming request being aborted.
      //

      req.on(
        'aborted',
        () => {

          clientAborted =
            true;

          console.warn(
            '[Proxy] Client aborted request'
          );

          // If NVIDIA is already streaming,
          // stop the upstream stream.
          if (
            streamingStarted &&
            response?.data &&
            typeof response.data.destroy ===
              'function'
          ) {

            response.data.destroy();
          }
        }
      );

      // ==========================================
      // INTERNAL NVIDIA REQUEST
      // ==========================================
      //
      // CRITICAL:
      //
      // If NVIDIA returns 429, this function
      // waits and retries.
      //
      // Nothing has been sent to the client yet.
      //
      // If NVIDIA returns 200, it returns the
      // successful stream and we proceed exactly
      // like the old behavior.
      //

      response =
        await requestNvidiaWithRetry(
          nimRequest
        );

      // ==========================================
      // CLIENT ABORTED DURING RETRY
      // ==========================================

      if (
        clientAborted
      ) {

        console.warn(
          '[Proxy] Client aborted before successful NVIDIA response'
        );

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
      // NOW STREAMING STARTS
      // ==========================================
      //
      // This is deliberately AFTER the retry
      // mechanism has succeeded.
      //
      // Therefore a 429 never causes a partial
      // response to be sent to the bot.
      //

      streamingStarted =
        true;

      // ==========================================
      // SSE HEADERS
      // ==========================================

      res.status(
        200
      );

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
      // FLUSH HEADERS
      // ==========================================

      if (
        typeof res.flushHeaders ===
        'function'
      ) {
        res.flushHeaders();
      }

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

        finished =
          true;

        // Close unfinished reasoning block.
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
            // Client may have gone away.
          }

          reasoningOpen =
            false;
        }

        // ========================================
        // OPENAI-COMPATIBLE DONE
        // ========================================

        try {

          if (
            !res.writableEnded
          ) {

            res.write(
              'data: [DONE]\n\n'
            );
          }

        } catch {
          // Client may have gone away.
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

      // ==========================================
      // WRITE SSE
      // ==========================================

      function writeSSE(data) {

        if (
          finished ||
          res.writableEnded ||
          clientAborted
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
        // PARSE JSON
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
        // GET DELTA
        // ========================================

        const choice =
          data?.choices?.[0];

        const delta =
          choice?.delta;

        if (
          !delta
        ) {

          writeSSE(
            data
          );

          return;
        }

        // ========================================
        // STEP 3.7 FLASH
        // NO REASONING
        // ========================================

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

          delete delta.reasoning;

          delete delta.reasoning_content;

          writeSSE(
            data
          );

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

        let output =
          '';

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
        // NORMAL CONTENT
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

        writeSSE(
          data
        );
      }

      // ==========================================
      // NVIDIA STREAM DATA
      // ==========================================

      response.data.on(
        'data',
        (chunk) => {

          if (
            finished ||
            res.writableEnded ||
            clientAborted
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
      // NVIDIA STREAM END
      // ==========================================

      response.data.on(
        'end',
        () => {

          if (
            finished
          ) {
            return;
          }

          // Process final buffered line.
          if (
            buffer.trim()
          ) {

            processLine(
              buffer
            );
          }

          // If NVIDIA did not send [DONE],
          // send it ourselves.
          sendDone();
        }
      );

      // ==========================================
      // NVIDIA STREAM ERROR
      // ==========================================

      response.data.on(
        'error',
        (error) => {

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

          // ========================================
          // STREAMING ERROR
          // ========================================
          //
          // At this point we already received 200.
          //
          // DO NOT retry.
          //
          // The retry window is only before
          // streaming begins.
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

          if (
            !res.writableEnded
          ) {

            res.end();
          }

          finished =
            true;
        }
      );

      // ==========================================
      // RESPONSE CLOSE
      // ==========================================
      //
      // ONLY use response close AFTER streaming
      // has started.
      //
      // This does NOT affect the retry loop because
      // the retry loop has already completed.
      //

      res.on(
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

          finished =
            true;
        }
      );

    } catch (error) {

      // ==========================================
      // SAFE ERROR
      // ==========================================

      logProxyError(
        error
      );

      // ==========================================
      // STREAM ALREADY STARTED
      // ==========================================

      if (
        streamingStarted ||
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
      // NORMAL ERROR RESPONSE
      // ==========================================

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
            status === 429
              ? 'rate_limit_error'
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

      if (
        !res.writableEnded
      ) {

        res.end();
      }

      return;
    }

    res.status(
      404
    ).json({

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
      `429 retries: ${
        MAX_ATTEMPTS - 1
      }`
    );

    console.log(
      '429 backoff: 10s -> 30s -> 60s -> 120s -> 240s'
    );

    console.log(
      'Streaming only: ENABLED'
    );

    console.log(
      '============================================'
    );
  }
);
