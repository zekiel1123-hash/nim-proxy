// server.js
// OpenAI-compatible NVIDIA NIM proxy
// STREAMING ONLY
//
// Client-facing models:
// - glm-5.2         -> z-ai/glm-5.2
// - kimi-k2.6       -> moonshotai/kimi-k2.6
// - deepseek-v4     -> deepseek-ai/deepseek-v4-pro
// - step-3.7-flash  -> stepfun-ai/step-3.7-flash
//
// IMPORTANT:
// - HTTP 429 responses are retried internally.
// - Successful HTTP 200 responses are streamed normally.
// - Retry delays:
//     10 seconds
//     30 seconds
//     60 seconds
//     120 seconds
//     300 seconds
//
// Step 3.7 Flash:
// - Thinking/reasoning intentionally disabled.
// - No chat_template_kwargs.
// - No reasoning_effort.
// - No generated <think> tags.
// - Stray <think> / </think> tags are removed.
//
// Other models:
// - Reasoning output remains enabled when SHOW_REASONING = true.

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

const SHOW_REASONING = true;

const ENABLE_THINKING_MODE = true;

const REASONING_EFFORT = 'low';

// ============================================
// FALLBACK
// ============================================

const FALLBACK_MODEL =
  'z-ai/glm-5.2';

// ============================================
// RETRY CONFIG
// ============================================
//
// 429 retry delays.
//
// Attempt 1:
//   initial request
//
// If 429:
//   wait 10 seconds
//
// If 429 again:
//   wait 30 seconds
//
// If 429 again:
//   wait 60 seconds
//
// If 429 again:
//   wait 120 seconds
//
// If 429 again:
//   wait 300 seconds
//
// Then give up.
//
// Total possible waits:
// 10 + 30 + 60 + 120 + 300 = 520 seconds
// ============================================

const RETRY_DELAYS = [
  10 * 1000,
  30 * 1000,
  60 * 1000,
  120 * 1000,
  300 * 1000
];

const MAX_RETRIES =
  RETRY_DELAYS.length;

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
// MODEL HELPERS
// ============================================

function isStep37Flash(model) {
  return (
    typeof model === 'string' &&
    (
      model === 'step-3.7-flash' ||
      model ===
        'stepfun-ai/step-3.7-flash' ||
      model.includes(
        'step-3.7-flash'
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
  // ============================================
  //
  // Thinking is intentionally disabled.
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
// CLEAN STEP 3.7 CONTENT
// ============================================

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
  if (
    error?.response
  ) {
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
    (resolve) =>
      setTimeout(
        resolve,
        ms
      )
  );
}

// ============================================
// READ ERROR STREAM
// ============================================

async function readErrorStream(
  stream
) {
  let body = '';

  if (
    !stream
  ) {
    return '';
  }

  try {
    for await (
      const chunk of stream
    ) {
      body +=
        chunk.toString(
          'utf8'
        );

      if (
        body.length >=
        100000
      ) {
        body =
          body.slice(
            0,
            100000
          );

        break;
      }
    }
  } catch {
    // Ignore error-stream read failures.
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
    return null;
  }

  try {
    return JSON.parse(
      body
    );
  } catch {
    return body;
  }
}

// ============================================
// GET ERROR MESSAGE FROM BODY
// ============================================

function getErrorBodyMessage(
  parsed,
  status
) {
  if (
    typeof parsed === 'string' &&
    parsed
  ) {
    return parsed;
  }

  if (
    parsed?.error?.message
  ) {
    return parsed.error.message;
  }

  if (
    typeof parsed?.error ===
    'string'
  ) {
    return parsed.error;
  }

  if (
    parsed?.message
  ) {
    return parsed.message;
  }

  return (
    `NVIDIA API returned HTTP ${status}`
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

      retry_on_429: true,

      retry_delays_seconds:
        RETRY_DELAYS.map(
          (ms) =>
            ms / 1000
        ),

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
    let upstreamStream =
      null;

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

      const clientModel =
        model || FALLBACK_MODEL;

      const nimModel =
        MODEL_MAPPING[
          clientModel
        ] ||
        FALLBACK_MODEL;

      const step37 =
        isStep37Flash(
          nimModel
        );

      // ============================================
      // BUILD NVIDIA REQUEST
      // ============================================

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
      // THINKING
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

      console.log(
        `[Proxy] ${clientModel} -> ${nimModel}` +
        (
          step37
            ? ' [REASONING DISABLED]'
            : ''
        )
      );

      // ============================================
      // CLIENT ABORT HANDLING
      // ============================================
      //
      // IMPORTANT:
      //
      // DO NOT use req.on('close') here.
      //
      // A normal HTTP request can emit "close"
      // after the request body has finished.
      //
      // That was causing the retry cycle / stream
      // to incorrectly report:
      //
      // "Client disconnected"
      //
      // Instead use "aborted", which indicates
      // that the incoming request was actually
      // aborted.
      // ============================================

      req.on(
        'aborted',
        () => {
          clientAborted =
            true;

          console.log(
            '[Proxy] Client request aborted'
          );

          if (
            upstreamStream &&
            typeof upstreamStream.destroy ===
              'function'
          ) {
            upstreamStream.destroy();
          }
        }
      );

      // ============================================
      // NVIDIA REQUEST / RETRY LOOP
      // ============================================

      let response =
        null;

      let successfulResponse =
        false;

      for (
        let attempt = 1;
        attempt <=
          MAX_RETRIES + 1;
        attempt++
      ) {
        if (
          clientAborted ||
          req.aborted
        ) {
          console.log(
            '[Proxy] Request was actually aborted; stopping retry cycle'
          );

          return;
        }

        console.log(
          `[NVIDIA Request] Attempt ${attempt}/${MAX_RETRIES + 1}`
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

                // IMPORTANT:
                //
                // We inspect HTTP status ourselves
                // so 429 can be handled internally.
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
            successfulResponse =
              true;

            console.log(
              `[NVIDIA Request] HTTP ${response.status} - success`
            );

            break;
          }

          // ==========================================
          // 429
          // ==========================================

          if (
            response.status === 429
          ) {
            const errorBody =
              await readErrorStream(
                response.data
              );

            const parsedError =
              parseErrorBody(
                errorBody
              );

            const errorMessage =
              getErrorBodyMessage(
                parsedError,
                429
              );

            const retryIndex =
              attempt - 1;

            // No retries remaining.
            if (
              retryIndex >=
              RETRY_DELAYS.length
            ) {
              console.error(
                '[NVIDIA Retry] Maximum 429 retries reached'
              );

              console.error(
                '[NVIDIA Retry]',
                errorMessage
              );

              return res.status(
                429
              ).json({
                error: {
                  message:
                    errorMessage,

                  type:
                    'rate_limit_error',

                  code: 429
                }
              });
            }

            const delay =
              RETRY_DELAYS[
                retryIndex
              ];

            console.warn(
              `[NVIDIA Retry] HTTP 429. ` +
              `Retry ${retryIndex + 1}/${MAX_RETRIES} ` +
              `in ${delay / 1000}s`
            );

            console.warn(
              `[NVIDIA Retry] ${errorMessage}`
            );

            // ========================================
            // WAIT
            // ========================================

            await sleep(
              delay
            );

            // Go back around and issue a
            // completely new NVIDIA request.
            continue;
          }

          // ==========================================
          // NON-429 HTTP ERROR
          // ==========================================

          const errorBody =
            await readErrorStream(
              response.data
            );

          const parsedError =
            parseErrorBody(
              errorBody
            );

          const errorMessage =
            getErrorBodyMessage(
              parsedError,
              response.status
            );

          console.error(
            `[NVIDIA Error] HTTP ${response.status}: ${errorMessage}`
          );

          return res.status(
            response.status
          ).json({
            error: {
              message:
                errorMessage,

              type:
                'nvidia_api_error',

              code:
                response.status
            }
          });
        } catch (error) {
          // ==========================================
          // NETWORK / AXIOS ERROR
          // ==========================================

          logProxyError(
            error
          );

          // Do NOT retry arbitrary network
          // errors here.
          //
          // The requested retry behavior is
          // specifically for HTTP 429.
          //

          if (
            !res.headersSent
          ) {
            return res.status(
              error?.response?.status ||
                500
            ).json({
              error: {
                message:
                  getSafeErrorMessage(
                    error
                  ),

                type:
                  'invalid_request_error',

                code:
                  error?.response
                    ?.status ||
                  500
              }
            });
          }

          return;
        }
      }

      // ============================================
      // SAFETY CHECK
      // ============================================

      if (
        !successfulResponse ||
        !response
      ) {
        if (
          !res.headersSent &&
          !res.writableEnded
        ) {
          return res.status(500).json({
            error: {
              message:
                'Unable to obtain a successful NVIDIA response',

              type:
                'nvidia_api_error',

              code: 500
            }
          });
        }

        return;
      }

      // ============================================
      // HTTP 200 STREAM
      // ============================================
      //
      // From this point forward, the behavior is
      // the normal streaming path.
      //
      // No retry happens after streaming begins.
      // ============================================

      upstreamStream =
        response.data;

      console.log(
        '[NVIDIA Stream] HTTP 200 - starting stream'
      );

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
      // FLUSH HEADERS IMMEDIATELY
      // ============================================

      if (
        typeof res.flushHeaders ===
        'function'
      ) {
        res.flushHeaders();
      }

      // ============================================
      // SSE STREAM STATE
      // ============================================

      let buffer = '';

      let reasoningOpen =
        false;

      let finished =
        false;

      let receivedData =
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
        // CLOSE REASONING
        // ==========================================

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
            // Client may have disconnected.
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
          // Ignore write failure.
        }

        if (
          !res.writableEnded
        ) {
          res.end();
        }

        console.log(
          '[NVIDIA Stream] Complete'
        );

        if (
          !receivedData
        ) {
          console.warn(
            '[NVIDIA Stream] WARNING: HTTP 200 but no SSE data was received'
          );
        }
      }

      // ============================================
      // WRITE SSE
      // ============================================

      function writeSSE(data) {
        if (
          finished ||
          res.writableEnded
        ) {
          return;
        }

        try {
          const output =
            `data: ${JSON.stringify(
              data
            )}\n\n`;

          res.write(
            output
          );

          receivedData =
            true;
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
        if (
          finished
        ) {
          return;
        }

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

        // SSE comments
        if (
          line.startsWith(':')
        ) {
          return;
        }

        // NVIDIA SSE
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
        // JSON
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
            error.message,
            raw.slice(
              0,
              500
            )
          );

          return;
        }

        // ==========================================
        // DELTA
        // ==========================================

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
        // REPLACE CONTENT
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
        // SEND
        // ==========================================

        writeSSE(
          data
        );
      }

      // ============================================
      // UPSTREAM DATA
      // ============================================

      upstreamStream.on(
        'data',
        (chunk) => {
          if (
            finished ||
            res.writableEnded
          ) {
            return;
          }

          receivedData =
            true;

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
      // UPSTREAM END
      // ============================================

      upstreamStream.on(
        'end',
        () => {
          // Process remaining complete
          // buffered data.
          if (
            buffer.trim()
          ) {
            processLine(
              buffer
            );
          }

          // If NVIDIA closed the HTTP stream
          // without sending [DONE], we still
          // send [DONE] to the client.
          sendDone();
        }
      );

      // ============================================
      // UPSTREAM ERROR
      // ============================================

      upstreamStream.on(
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

          // Once a 200 streaming response has
          // started, send an SSE error rather
          // than attempting res.json().
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

      // ============================================
      // CLIENT ABORT AFTER STREAM START
      // ============================================
      //
      // Again:
      //
      // We intentionally do NOT listen to
      // req.close.
      //
      // "aborted" is the actual abort signal.
      // ============================================

      req.on(
        'aborted',
        () => {
          if (
            finished
          ) {
            return;
          }

          finished =
            true;

          console.log(
            '[Proxy] Client aborted active stream'
          );

          if (
            upstreamStream &&
            typeof upstreamStream.destroy ===
              'function'
          ) {
            upstreamStream.destroy();
          }
        }
      );
    } catch (error) {
      // ============================================
      // SAFE TOP-LEVEL ERROR
      // ============================================

      logProxyError(
        error
      );

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
      '429 retry delays: 10s, 30s, 60s, 120s, 300s'
    );

    console.log(
      'Streaming only: ENABLED'
    );

    console.log(
      '============================================'
    );
  }
);
