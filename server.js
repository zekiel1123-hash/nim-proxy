// server.js
// OpenAI-compatible NVIDIA NIM proxy
// STREAMING ONLY
//
// Client models:
// - glm-5.2          -> z-ai/glm-5.2
// - kimi-k2.6        -> moonshotai/kimi-k2.6
// - deepseek-v4      -> deepseek-ai/deepseek-v4-pro
// - step-3.7-flash   -> stepfun-ai/step-3.7-flash
//
// Features:
// - Streaming only
// - Exponential/slower retry backoff
// - Retry delays: 10s -> 30s -> 60s -> 60s -> 60s
// - HTTP 429 retry
// - HTTP 5xx retry
// - Safe Axios error handling
// - No API key logging
// - Step 3.7 reasoning disabled
// - Other model reasoning supported
// - Robust NVIDIA SSE parsing
// - OpenAI-compatible SSE output
// - Guarantees final [DONE]
// - Flushes SSE headers immediately
//
// IMPORTANT:
// Step 3.7 Flash intentionally receives NO thinking parameters.
// We also remove stray </think> tags from its output.

// ============================================
// IMPORTS
// ============================================

const express = require('express');
const cors = require('cors');
const axios = require('axios');

// ============================================
// APP
// ============================================

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
// DISPLAY / THINKING CONFIG
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
//
// Deliberately slow:
//
// Attempt 1
//   immediate
//
// Attempt 2
//   wait 10 seconds
//
// Attempt 3
//   wait 30 seconds
//
// Attempt 4
//   wait 60 seconds
//
// Attempt 5
//   wait 60 seconds
//
// Attempt 6
//   wait 60 seconds
//
// Total possible retry waiting:
// 10 + 30 + 60 + 60 + 60 = 220 seconds
//
// This is especially useful for NVIDIA 429
// rate-limit responses.
//

const MAX_ATTEMPTS = 6;

const RETRY_DELAYS_MS = [
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
// Client-visible model names are on the left.
// NVIDIA NIM model names are on the right.
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
// FALLBACK
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
    model.includes(
      'step-3.7-flash'
    ) ||
    model.includes(
      'stepfun-ai/step-3.7-flash'
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
  // INTENTIONALLY NOTHING.
  //
  // No:
  //   chat_template_kwargs
  //
  // No:
  //   reasoning_effort
  //
  // No thinking flag.
  //

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
// CLEAN STEP CONTENT
// ============================================
//
// Step 3.7 can occasionally emit:
//
// </think>
//
// even though reasoning is disabled.
//
// Remove both opening and closing tags.
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
// RETRYABLE STATUS
// ============================================

function isRetryableStatus(
  status
) {
  return (
    status === 408 ||
    status === 409 ||
    status === 425 ||
    status === 429 ||
    status >= 500
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
    return body;
  }

  try {
    for await (
      const chunk of stream
    ) {
      body += chunk.toString(
        'utf8'
      );

      if (
        body.length >= 100000
      ) {
        break;
      }
    }
  } catch {
    // Ignore read failure.
  }

  return body;
}

// ============================================
// NVIDIA REQUEST WITH RETRIES
// ============================================
//
// IMPORTANT:
//
// We only retry before a successful
// streaming response has begun.
//
// Once NVIDIA returns HTTP 200,
// the stream is handed to the client.
//
// We NEVER retry halfway through a
// successful stream.
//

async function requestNvidia(
  nimRequest,
  req
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

            timeout: 0,

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
          `[NVIDIA Request] HTTP ${response.status}`
        );

        return response;
      }

      // ============================================
      // HTTP ERROR
      // ============================================

      const errorBody =
        await readErrorStream(
          response.data
        );

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
              parsed?.error?.message ||
              parsed?.message ||
              `NVIDIA API returned HTTP ${response.status}`
            );

      console.error(
        `[NVIDIA Error] HTTP ${response.status}: ${message}`
      );

      // ============================================
      // NOT RETRYABLE
      // ============================================

      if (
        !isRetryableStatus(
          response.status
        )
      ) {
        const error =
          new Error(
            message
          );

        error.response = {
          status:
            response.status,

          data:
            parsed
        };

        throw error;
      }

      // ============================================
      // LAST ATTEMPT
      // ============================================

      if (
        attempt >= MAX_ATTEMPTS
      ) {
        const error =
          new Error(
            message
          );

        error.response = {
          status:
            response.status,

          data:
            parsed
        };

        throw error;
      }

      // ============================================
      // RETRY DELAY
      // ============================================

      const delay =
        RETRY_DELAYS_MS[
          attempt - 1
        ] ??
        60_000;

      console.log(
        `[NVIDIA Retry] HTTP ${response.status}. ` +
        `Retry ${attempt}/${MAX_ATTEMPTS - 1} ` +
        `in ${Math.round(delay / 1000)}s`
      );

      await sleep(
        delay
      );

      // If the HTTP request is still
      // associated with an active request,
      // continue.
      //
      // We intentionally do NOT abort simply
      // because req.close fired here. Some
      // clients/proxies can emit close while
      // maintaining the logical request.
      //

      lastError =
        new Error(
          message
        );

      lastError.response = {
        status:
          response.status,

        data:
          parsed
      };
    } catch (error) {
      // ============================================
      // AXIOS / NETWORK ERROR
      // ============================================

      lastError =
        error;

      const status =
        error?.response?.status;

      // ============================================
      // NON-RETRYABLE
      // ============================================

      if (
        !status ||
        !isRetryableStatus(
          status
        )
      ) {
        throw error;
      }

      // ============================================
      // LAST ATTEMPT
      // ============================================

      if (
        attempt >= MAX_ATTEMPTS
      ) {
        throw error;
      }

      const delay =
        RETRY_DELAYS_MS[
          attempt - 1
        ] ??
        60_000;

      console.log(
        `[NVIDIA Retry] ${status} ` +
        `Retry ${attempt}/${MAX_ATTEMPTS - 1} ` +
        `in ${Math.round(delay / 1000)}s`
      );

      await sleep(
        delay
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

      streaming_only:
        true,

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
              ms / 1000
          )
      },

      models: {
        'glm-5.2': {
          reasoning:
            true
        },

        'kimi-k2.6': {
          reasoning:
            true
        },

        'deepseek-v4': {
          reasoning:
            true
        },

        'step-3.7-flash': {
          reasoning:
            false
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
            id:
              model,

            object:
              'model',

            created:
              Math.floor(
                Date.now() /
                  1000
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

    let streamFinished =
      false;

    let clientClosed =
      false;

    try {
      const {
        model,
        messages,
        temperature,
        max_tokens,
        top_p,
        seed
      } =
        req.body || {};

      // ============================================
      // VALIDATE
      // ============================================

      if (
        !Array.isArray(
          messages
        )
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

      // ============================================
      // RESOLVE MODEL
      // ============================================

      const nimModel =
        MODEL_MAPPING[
          model
        ] ||
        FALLBACK_MODEL;

      const step37 =
        isStep37Flash(
          nimModel
        );

      console.log(
        `[Proxy] ${model || 'unknown'} -> ${nimModel}` +
        (
          step37
            ? ' [REASONING DISABLED]'
            : ''
        )
      );

      // ============================================
      // BUILD NVIDIA REQUEST
      // ============================================

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

      // ============================================
      // TOP P
      // ============================================

      if (
        top_p !==
          undefined &&
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
        seed !==
          undefined &&
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

      // ============================================
      // LOG REQUEST
      // ============================================

      console.log(
        '[NVIDIA Request] Model:',
        nimModel
      );

      console.log(
        '[NVIDIA Request] Streaming: true'
      );

      if (
        step37
      ) {
        console.log(
          '[NVIDIA Request] Step 3.7 reasoning: OFF'
        );
      }

      // ============================================
      // REQUEST NVIDIA
      // ============================================

      response =
        await requestNvidia(
          nimRequest,
          req
        );

      // ============================================
      // NVIDIA SUCCESS
      // ============================================

      console.log(
        `[NVIDIA Stream] HTTP ${response.status} - connected`
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

      // Send an initial harmless
      // SSE comment to force proxies
      // and clients to recognize the stream.
      //
      // This is NOT an OpenAI data chunk.
      //

      try {
        res.write(
          ': connected\n\n'
        );
      } catch {
        // Ignore.
      }

      // ============================================
      // STREAM STATE
      // ============================================

      let buffer = '';

      let reasoningOpen =
        false;

      let contentChunks =
        0;

      let reasoningChunks =
        0;

      let totalBytes =
        0;

      // ============================================
      // SEND DONE
      // ============================================

      function sendDone() {
        if (
          streamFinished
        ) {
          return;
        }

        streamFinished =
          true;

        // ============================================
        // CLOSE REASONING
        // ============================================

        if (
          SHOW_REASONING &&
          reasoningOpen
        ) {
          const closeChunk =
            {
              id:
                `chatcmpl-proxy-${Date.now()}`,

              object:
                'chat.completion.chunk',

              created:
                Math.floor(
                  Date.now() /
                    1000
                ),

              model:
                model ||
                nimModel,

              choices: [
                {
                  index:
                    0,

                  delta: {
                    content:
                      '\n</think>\n'
                  },

                  finish_reason:
                    null
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
            // Ignore.
          }

          reasoningOpen =
            false;
        }

        // ============================================
        // FINAL FINISH CHUNK
        // ============================================
        //
        // Some OpenAI-compatible clients rely
        // on an explicit finish_reason chunk.
        //

        const finishChunk =
          {
            id:
              `chatcmpl-proxy-${Date.now()}`,

            object:
              'chat.completion.chunk',

            created:
              Math.floor(
                Date.now() /
                  1000
              ),

            model:
              model ||
              nimModel,

            choices: [
              {
                index:
                  0,

                delta: {},

                finish_reason:
                  'stop'
              }
            ]
          };

        try {
          if (
            !res.writableEnded
          ) {
            res.write(
              `data: ${JSON.stringify(
                finishChunk
              )}\n\n`
            );

            res.write(
              'data: [DONE]\n\n'
            );
          }
        } catch {
          // Client disconnected.
        }

        if (
          !res.writableEnded
        ) {
          res.end();
        }

        console.log(
          `[NVIDIA Stream] Complete ` +
          `content_chunks=${contentChunks} ` +
          `reasoning_chunks=${reasoningChunks} ` +
          `bytes=${totalBytes}`
        );
      }

      // ============================================
      // WRITE SSE
      // ============================================

      function writeSSE(data) {
        if (
          streamFinished ||
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
        } catch (
          error
        ) {
          console.error(
            '[SSE Write Error]',
            error.message
          );
        }
      }

      // ============================================
      // PROCESS SSE DATA
      // ============================================

      function processLine(line) {
        // Normalize CRLF.
        line =
          line.replace(
            /\r$/,
            ''
          );

        // Ignore blank lines.
        if (
          !line.trim()
        ) {
          return;
        }

        // Ignore SSE comments.
        if (
          line.startsWith(':')
        ) {
          return;
        }

        // We only care about data lines.
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
          sendDone();
          return;
        }

        if (
          !raw
        ) {
          return;
        }

        // ============================================
        // PARSE JSON
        // ============================================

        let data;

        try {
          data =
            JSON.parse(
              raw
            );
        } catch (
          error
        ) {
          console.error(
            '[SSE Parse Error]',
            error.message,
            'Raw:',
            raw.slice(
              0,
              500
            )
          );

          return;
        }

        // ============================================
        // GET CHOICE
        // ============================================

        const choice =
          data?.choices?.[0];

        const delta =
          choice?.delta;

        // ============================================
        // NON-DELTA CHUNK
        // ============================================

        if (
          !delta
        ) {
          writeSSE(
            data
          );

          return;
        }

        // ============================================
        // STEP 3.7
        // ============================================

        if (
          step37
        ) {
          // Remove reasoning fields.
          delete delta.reasoning;

          delete delta.reasoning_content;

          // Clean any stray think tags.
          if (
            typeof delta.content ===
            'string'
          ) {
            const before =
              delta.content;

            const after =
              cleanStepContent(
                before
              );

            if (
              before !==
              after
            ) {
              console.log(
                '[Step 3.7] Removed stray <think> tag'
              );
            }

            delta.content =
              after;
          }

          // Log actual client-visible content.
          if (
            delta.content
          ) {
            contentChunks++;

            console.log(
              `[NVIDIA -> Client] content: ${JSON.stringify(
                delta.content
              )}`
            );
          }

          writeSSE(
            data
          );

          return;
        }

        // ============================================
        // OTHER MODELS
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

        let output =
          '';

        // ============================================
        // REASONING
        // ============================================

        if (
          SHOW_REASONING &&
          reasoning
        ) {
          reasoningChunks++;

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
        // CONTENT
        // ============================================

        if (
          content
        ) {
          contentChunks++;

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
        // IMPORTANT
        // ============================================
        //
        // If NVIDIA sent reasoning/content,
        // make absolutely sure the client gets
        // it through delta.content.
        //

        if (
          output
        ) {
          delta.content =
            output;

          console.log(
            `[NVIDIA -> Client] content: ${JSON.stringify(
              output
            )}`
          );
        }

        // Remove raw reasoning fields.
        delete delta.reasoning;

        delete delta.reasoning_content;

        // ============================================
        // FORWARD CHUNK
        // ============================================

        writeSSE(
          data
        );
      }

      // ============================================
      // NVIDIA DATA
      // ============================================

      response.data.on(
        'data',
        (chunk) => {
          if (
            streamFinished
          ) {
            return;
          }

          const text =
            chunk.toString(
              'utf8'
            );

          totalBytes +=
            Buffer.byteLength(
              text,
              'utf8'
            );

          buffer +=
            text;

          // Handle both:
          //
          // \n
          // \r\n
          //
          const lines =
            buffer.split(
              '\n'
            );

          buffer =
            lines.pop() ||
            '';

          for (
            const line
            of lines
          ) {
            if (
              streamFinished
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
      // NVIDIA STREAM END
      // ============================================

      response.data.on(
        'end',
        () => {
          if (
            streamFinished
          ) {
            return;
          }

          // ============================================
          // PROCESS REMAINING BUFFER
          // ============================================

          if (
            buffer.trim()
          ) {
            processLine(
              buffer
            );

            buffer =
              '';
          }

          // ============================================
          // ALWAYS FINISH CLIENT STREAM
          // ============================================

          sendDone();
        }
      );

      // ============================================
      // NVIDIA STREAM ERROR
      // ============================================

      response.data.on(
        'error',
        (error) => {
          console.error(
            '[NVIDIA Stream Error]',
            error.message
          );

          if (
            streamFinished
          ) {
            return;
          }

          streamFinished =
            true;

          // If headers have already gone out,
          // send an SSE error rather than JSON.
          //

          try {
            if (
              !res.writableEnded
            ) {
              res.write(
                `data: ${JSON.stringify({
                  error: {
                    message:
                      error.message ||
                      'NVIDIA stream error',

                    type:
                      'stream_error'
                  }
                })}\n\n`
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
      );

      // ============================================
      // REQUEST CLOSE
      // ============================================
      //
      // DO NOT immediately destroy the NVIDIA
      // stream here.
      //
      // Some reverse proxies / clients can emit
      // close around the lifecycle of a request.
      //
      // Only destroy once the response is actually
      // finished or the request is truly gone.
      //

      req.on(
        'aborted',
        () => {
          clientClosed =
            true;

          console.log(
            '[Proxy] Client request aborted'
          );

          if (
            !streamFinished &&
            response?.data &&
            typeof response.data.destroy ===
              'function'
          ) {
            response.data.destroy();
          }
        }
      );

      res.on(
        'close',
        () => {
          if (
            res.writableEnded
          ) {
            return;
          }

          if (
            !streamFinished
          ) {
            clientClosed =
              true;

            console.log(
              '[Proxy] Client response closed before stream completion'
            );

            if (
              response?.data &&
              typeof response.data.destroy ===
                'function'
            ) {
              response.data.destroy();
            }
          }
        }
      );
    } catch (
      error
    ) {
      // ============================================
      // SAFE ERROR HANDLING
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
      return res.end();
    }

    res.status(
      404
    ).json({
      error: {
        message:
          `Endpoint ${req.path} not found`,

        type:
          'invalid_request_error',

        code:
          404
      }
    });
  }
);

// ============================================
// START
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
      'Retry schedule: 10s -> 30s -> 60s -> 60s -> 60s'
    );

    console.log(
      'Streaming only: ENABLED'
    );

    console.log(
      '============================================'
    );
  }
);
