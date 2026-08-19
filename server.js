// server.js
// OpenAI-compatible NVIDIA NIM proxy
// STREAMING ONLY
//
// Client models:
// - glm-5.2       -> z-ai/glm-5.2
// - kimi-k2.6     -> moonshotai/kimi-k2.6
// - deepseek-v4   -> deepseek-ai/deepseek-v4-pro
// - step-3.7-flash -> stepfun-ai/step-3.7-flash
//
// IMPORTANT:
// - 429 responses are retried INTERNALLY.
// - The client does NOT receive anything until NVIDIA
//   successfully returns HTTP 200.
// - Retry delays: 10s, 30s, 60s, 60s, 60s.
// - A successful 200 response is streamed normally.
// - Step 3.7 Flash reasoning is disabled.
// - No reasoning parameters are sent to Step 3.7 Flash.
// - Stray </think> tags from Step 3.7 Flash are removed.
//
// STREAMING ONLY
// ============================================

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
// 429 retry delays:
//
// Attempt 1 -> 429 -> wait 10 seconds
// Attempt 2 -> 429 -> wait 30 seconds
// Attempt 3 -> 429 -> wait 60 seconds
// Attempt 4 -> 429 -> wait 60 seconds
// Attempt 5 -> 429 -> wait 60 seconds
//
// Total NVIDIA attempts = 6
//
// These retries happen BEFORE the client receives
// an SSE response.
//
// ============================================

const RETRY_DELAYS_MS = [
  10_000,
  30_000,
  60_000,
  60_000,
  60_000
];

const MAX_NVIDIA_ATTEMPTS =
  RETRY_DELAYS_MS.length + 1;

// ============================================
// MODEL MAPPING
// ============================================

const MODEL_MAPPING = {

  // Client-visible name
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
  // ============================================
  //
  // Intentionally disabled.
  //
  // No:
  // chat_template_kwargs
  // reasoning_effort
  // thinking=true
  //
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
// STEP 3.7 CLEANUP
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
//
// Used only when NVIDIA returns an HTTP error.
// Never tries to JSON.stringify a Node stream.
//
// ============================================

async function readErrorStream(stream) {

  let body = '';

  if (!stream) {
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

  } catch (error) {

    console.error(
      '[NVIDIA Error Stream]',
      error.message
    );
  }

  return body;
}

// ============================================
// PARSE NVIDIA ERROR
// ============================================

function parseNvidiaError(body) {

  if (
    !body
  ) {
    return null;
  }

  try {
    return JSON.parse(body);
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
// The client response object is NOT touched here.
//
// We make the NVIDIA request.
// If NVIDIA returns 429:
//   - consume the error stream
//   - wait
//   - make another NVIDIA request
//
// Only after HTTP 200 do we return the stream.
//
// ============================================

async function requestNvidiaWithRetry(
  nimRequest,
  clientRequest
) {

  for (
    let attempt = 1;
    attempt <= MAX_NVIDIA_ATTEMPTS;
    attempt++
  ) {

    // ============================================
    // CLIENT DISCONNECTED BEFORE NVIDIA SUCCESS
    // ============================================

    if (
      clientRequest.destroyed ||
      clientRequest.aborted
    ) {

      const error =
        new Error(
          'Client disconnected before NVIDIA request completed'
        );

      error.code =
        'CLIENT_DISCONNECTED';

      throw error;
    }

    console.log(
      `[NVIDIA Request] Attempt ${attempt}/${MAX_NVIDIA_ATTEMPTS}`
    );

    let response;

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
            // Let us inspect 429 ourselves.
            validateStatus:
              () => true,

            // No Axios request timeout.
            //
            // NVIDIA streaming requests can remain
            // open for a long time.
            timeout: 0
          }
        );

    } catch (error) {

      // Network-level Axios failure.
      //
      // This is different from HTTP 429.
      //
      console.error(
        '[NVIDIA Request Error]',
        error.message
      );

      throw error;
    }

    // ============================================
    // SUCCESS
    // ============================================

    if (
      response.status >= 200 &&
      response.status < 300
    ) {

      console.log(
        `[NVIDIA Request] HTTP ${response.status} - SUCCESS`
      );

      return response;
    }

    // ============================================
    // HTTP 429
    // ============================================

    if (
      response.status === 429
    ) {

      const errorBody =
        await readErrorStream(
          response.data
        );

      const parsedError =
        parseNvidiaError(
          errorBody
        );

      const message =
        typeof parsedError ===
        'string'
          ? parsedError
          : (
              parsedError?.error?.message ||
              parsedError?.message ||
              'NVIDIA API returned HTTP 429'
            );

      // ============================================
      // LAST ATTEMPT
      // ============================================

      if (
        attempt >=
        MAX_NVIDIA_ATTEMPTS
      ) {

        console.error(
          `[NVIDIA Retry] Exhausted all ${MAX_NVIDIA_ATTEMPTS} attempts`
        );

        const error =
          new Error(
            message
          );

        error.response = {
          status: 429,
          data: parsedError
        };

        throw error;
      }

      const delay =
        RETRY_DELAYS_MS[
          attempt - 1
        ];

      console.warn(
        `[NVIDIA Retry] HTTP 429. ` +
        `Retry ${attempt}/${RETRY_DELAYS_MS.length} ` +
        `in ${delay / 1000}s`
      );

      console.warn(
        `[NVIDIA Retry] ${message}`
      );

      // ============================================
      // WAIT
      // ============================================

      await sleep(
        delay
      );

      // ============================================
      // RETRY
      // ============================================

      continue;
    }

    // ============================================
    // OTHER NVIDIA HTTP ERROR
    // ============================================

    const errorBody =
      await readErrorStream(
        response.data
      );

    const parsedError =
      parseNvidiaError(
        errorBody
      );

    const message =
      typeof parsedError ===
      'string'
        ? parsedError
        : (
            parsedError?.error?.message ||
            parsedError?.message ||
            `NVIDIA API returned HTTP ${response.status}`
          );

    console.error(
      `[NVIDIA Error] HTTP ${response.status}: ${message}`
    );

    const error =
      new Error(
        message
      );

    error.response = {
      status:
        response.status,

      data:
        parsedError
    };

    throw error;
  }

  throw new Error(
    'NVIDIA retry loop unexpectedly exited'
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

      retry_delays_seconds:
        RETRY_DELAYS_MS.map(
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
  async (
    req,
    res
  ) => {

    let response = null;

    let streamingStarted =
      false;

    let clientDisconnected =
      false;

    // ============================================
    // CLIENT DISCONNECT
    // ============================================
    //
    // IMPORTANT:
    //
    // This only records the disconnect.
    // It does NOT immediately interfere with the
    // retry function.
    //
    // The retry function checks the request state
    // between attempts.
    //
    // ============================================

    req.on(
      'close',
      () => {

        clientDisconnected =
          true;

        console.log(
          '[Proxy] Client connection closed'
        );

        // Once NVIDIA streaming has started,
        // destroy the upstream stream.
        //
        // During retry/waiting, there may not
        // be an active NVIDIA stream yet.
        //
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

              code:
                400
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
        isStep37Flash(
          nimModel
        );

      console.log(
        `[Proxy] ${model || 'unknown'} -> ${nimModel}`
      );

      if (step37) {

        console.log(
          '[Proxy] Step 3.7 Flash reasoning disabled'
        );
      }

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

      // ============================================
      // NVIDIA REQUEST + RETRY
      // ============================================
      //
      // NOTHING HAS BEEN SENT TO THE CLIENT YET.
      //
      // This is intentional.
      //
      // If NVIDIA returns:
      //
      // 429 -> wait -> retry
      //
      // The client remains waiting.
      //
      // Once NVIDIA returns 200:
      //
      // start SSE response.
      //
      // ============================================

      response =
        await requestNvidiaWithRetry(
          nimRequest,
          req
        );

      // ============================================
      // SUCCESSFUL NVIDIA RESPONSE
      // ============================================

      console.log(
        '[NVIDIA Stream] HTTP 200 received'
      );

      console.log(
        '[NVIDIA Stream] Starting client stream'
      );

      // ============================================
      // CHECK CLIENT
      // ============================================

      if (
        clientDisconnected ||
        req.destroyed
      ) {

        console.warn(
          '[Proxy] Client disconnected before stream started'
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
      //
      // This tells the client that the stream
      // has actually started.
      //
      if (
        typeof res.flushHeaders ===
        'function'
      ) {

        res.flushHeaders();
      }

      streamingStarted =
        true;

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

        if (
          finished
        ) {
          return;
        }

        finished =
          true;

        // ============================================
        // CLOSE THINK
        // ============================================

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
            // Ignore client disconnect.
          }

          reasoningOpen =
            false;
        }

        // ============================================
        // DONE
        // ============================================

        try {

          if (
            !res.writableEnded
          ) {

            res.write(
              'data: [DONE]\n\n'
            );
          }

        } catch {
          // Ignore client disconnect.
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
      // PROCESS SSE LINE
      // ============================================

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

        // ============================================
        // DONE
        // ============================================

        if (
          raw === '[DONE]'
        ) {

          sendDone();

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

        // Some SSE messages contain metadata
        // without a delta.
        //
        // Pass those through unchanged.
        //

        if (
          !delta
        ) {

          writeSSE(
            data
          );

          return;
        }

        // ============================================
        // STEP 3.7 FLASH
        // NO REASONING
        // ============================================

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

        // ============================================
        // OTHER MODELS
        // REASONING
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

        // ============================================
        // CONTENT REPLACEMENT
        // ============================================

        if (
          output
        ) {

          delta.content =
            output;
        }

        // ============================================
        // REMOVE RAW REASONING
        // ============================================

        delete delta.reasoning;

        delete delta.reasoning_content;

        // ============================================
        // SEND
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
            lines.pop() ||
            '';

          for (
            const line
            of lines
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
      // NVIDIA STREAM END
      // ============================================

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
            finished ||
            res.writableEnded
          ) {

            return;
          }

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

    } catch (
      error
    ) {

      // ============================================
      // SAFE ERROR
      // ============================================

      logProxyError(
        error
      );

      // ============================================
      // CLIENT ALREADY DISCONNECTED
      // ============================================

      if (
        error?.code ===
        'CLIENT_DISCONNECTED'
      ) {

        return;
      }

      // ============================================
      // STREAM ALREADY STARTED
      // ============================================

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

      // ============================================
      // NORMAL HTTP ERROR
      // ============================================

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

    res
      .status(404)
      .json({

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
      `Maximum NVIDIA attempts: ${MAX_NVIDIA_ATTEMPTS}`
    );

    console.log(
      'Streaming only: ENABLED'
    );

    console.log(
      '============================================'
    );
  }
);
