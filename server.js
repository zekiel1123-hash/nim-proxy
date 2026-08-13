// server.js
// OpenAI-compatible NVIDIA NIM proxy
// STREAMING ONLY
//
// Models:
// - glm-5.1       -> z-ai/glm-5.2
// - kimi-k2.6     -> moonshotai/kimi-k2.6
// - deepseek-v4   -> deepseek-ai/deepseek-v4-pro
// - stepfun-3.7   -> stepfun-ai/step-3.7-flash
//
// StepFun special handling:
// Step-3.7-Flash may stream reasoning as normal content and
// may emit </think> without first emitting <think>.
// This proxy normalizes that stream so clients receive:
//
//   <think>
//   reasoning...
//   </think>
//
// followed by the actual answer.
//
// STREAMING ONLY

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

// Enable model-specific thinking configuration
const ENABLE_THINKING_MODE = true;

// DeepSeek reasoning effort
const REASONING_EFFORT = 'low';

// Default fallback
const FALLBACK_MODEL = 'z-ai/glm-5.2';

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

  'stepfun-3.7':
    'stepfun-ai/step-3.7-flash'
};

// ============================================
// MODEL HELPERS
// ============================================

function isStepFunModel(model) {
  return (
    typeof model === 'string' &&
    model.toLowerCase().includes('step-3.7-flash')
  );
}

function isGLMModel(model) {
  return (
    typeof model === 'string' &&
    model.toLowerCase().includes('glm')
  );
}

function isKimiModel(model) {
  return (
    typeof model === 'string' &&
    model.toLowerCase().includes('kimi')
  );
}

function isDeepSeekModel(model) {
  return (
    typeof model === 'string' &&
    model.toLowerCase().includes('deepseek')
  );
}

// ============================================
// THINKING CONFIG
// ============================================

function buildThinkingConfig(model) {
  // ============================================
  // STEP-3.7-FLASH
  // ============================================
  //
  // NVIDIA's current Step-3.7-Flash API example
  // does not specify a chat_template_kwargs
  // thinking parameter.
  //
  // Do NOT send:
  //
  //   thinking: true
  //   enable_thinking: true
  //
  // to StepFun unless NVIDIA documents support
  // for those parameters.
  //
  // The proxy instead normalizes StepFun's
  // streamed reasoning markers below.
  //
  if (isStepFunModel(model)) {
    return {};
  }

  // ============================================
  // GLM
  // ============================================

  if (isGLMModel(model)) {
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

  if (isKimiModel(model)) {
    return {
      chat_template_kwargs: {
        thinking: true
      }
    };
  }

  // ============================================
  // DEEPSEEK
  // ============================================

  if (isDeepSeekModel(model)) {
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
  return (
    delta?.reasoning ||
    delta?.reasoning_content ||
    ''
  );
}

// ============================================
// SAFE ERROR SERIALIZATION
// ============================================
//
// IMPORTANT:
//
// Axios response.data can be a Node.js stream.
// Trying to put that object directly into JSON:
//
//   res.json({
//     error: error.response.data
//   })
//
// can produce:
//
//   TypeError: Converting circular structure to JSON
//
// Therefore we always convert errors into plain strings/objects.
//

async function getSafeErrorPayload(error) {
  const status =
    error?.response?.status || 500;

  const responseData =
    error?.response?.data;

  let message =
    error?.message ||
    'Unknown proxy error';

  // Axios response data may be a stream
  if (
    responseData &&
    typeof responseData.on === 'function'
  ) {
    try {
      let body = '';

      await new Promise((resolve) => {
        let settled = false;

        const finish = () => {
          if (!settled) {
            settled = true;
            resolve();
          }
        };

        responseData.setEncoding?.('utf8');

        responseData.on(
          'data',
          (chunk) => {
            body += chunk.toString();
          }
        );

        responseData.on(
          'end',
          finish
        );

        responseData.on(
          'error',
          finish
        );
      });

      if (body) {
        try {
          message = JSON.parse(body);
        } catch {
          message = body;
        }
      }
    } catch {
      message =
        error?.message ||
        'NVIDIA request failed';
    }
  }

  // Normal Axios JSON response
  else if (
    responseData !== undefined &&
    responseData !== null
  ) {
    if (
      typeof responseData === 'string'
    ) {
      message = responseData;
    } else {
      try {
        message =
          JSON.stringify(
            responseData
          );
      } catch {
        message =
          error?.message ||
          'NVIDIA request failed';
      }
    }
  }

  return {
    status,
    body: {
      error: {
        message,
        type:
          'invalid_request_error',
        code: status
      }
    }
  };
}

// ============================================
// WRITE SSE
// ============================================

function writeSSE(res, data) {
  if (res.writableEnded) {
    return;
  }

  res.write(
    `data: ${JSON.stringify(data)}\n\n`
  );
}

function writeDone(res) {
  if (res.writableEnded) {
    return;
  }

  res.write('data: [DONE]\n\n');
}

// ============================================
// CREATE CONTENT CHUNK
// ============================================

function createContentChunk(content) {
  return {
    choices: [
      {
        delta: {
          content
        }
      }
    ]
  };
}

// ============================================
// NORMALIZE REASONING MARKERS
// ============================================
//
// This function handles:
//
// 1. Separate NVIDIA reasoning fields:
//
//    delta.reasoning
//    delta.reasoning_content
//
// 2. Models that put:
//
//    <think>...</think>
//
//    directly inside delta.content.
//
// 3. StepFun behavior where the stream can contain:
//
//    reasoning...
//    </think>
//
//    without:
//
//    <think>
//
// For StepFun, content before </think> is buffered.
// Once </think> arrives, we know that buffered content
// was reasoning and can safely emit:
//
//    <think>
//    [buffered reasoning]
//    </think>
//
// This avoids accidentally displaying reasoning as
// normal answer text.
//

class ReasoningNormalizer {
  constructor(model) {
    this.model = model;

    this.stepFun =
      isStepFunModel(model);

    this.reasoningOpen = false;

    this.reasoningSeen = false;

    this.stepFunBuffer = '';

    this.stepFunFinishedThinking =
      false;

    this.stepFunExplicitThink =
      false;
  }

  // ------------------------------------------
  // STEP FUNCTION
  // ------------------------------------------

  processStepFunContent(content) {
    if (!content) {
      return '';
    }

    // Once thinking has already finished,
    // everything is ordinary answer content.
    if (
      this.stepFunFinishedThinking
    ) {
      return content;
    }

    // ------------------------------------------
    // If an explicit <think> appears
    // ------------------------------------------

    if (
      content.includes('<think>')
    ) {
      this.stepFunExplicitThink =
        true;

      this.reasoningOpen = true;

      this.reasoningSeen = true;

      // If we already buffered something before
      // seeing <think>, prepend it after the tag.
      const parts =
        content.split('<think>');

      let output = '';

      if (this.stepFunBuffer) {
        output +=
          '<think>\n' +
          this.stepFunBuffer;

        this.stepFunBuffer = '';
      } else {
        output +=
          '<think>\n';
      }

      output += parts[1] || '';

      // Handle closing tag in the same chunk.
      if (
        output.includes('</think>')
      ) {
        const closeIndex =
          output.indexOf(
            '</think>'
          );

        const beforeClose =
          output.slice(
            0,
            closeIndex
          );

        const afterClose =
          output.slice(
            closeIndex +
              '</think>'.length
          );

        this.reasoningOpen = false;

        this.stepFunFinishedThinking =
          true;

        return (
          beforeClose +
          '</think>\n\n' +
          afterClose
        );
      }

      return output;
    }

    // ------------------------------------------
    // Explicit thinking already started
    // ------------------------------------------

    if (
      this.stepFunExplicitThink ||
      this.reasoningOpen
    ) {
      this.reasoningSeen = true;

      if (
        content.includes('</think>')
      ) {
        const closeIndex =
          content.indexOf(
            '</think>'
          );

        const beforeClose =
          content.slice(
            0,
            closeIndex
          );

        const afterClose =
          content.slice(
            closeIndex +
              '</think>'.length
          );

        this.reasoningOpen = false;

        this.stepFunFinishedThinking =
          true;

        return (
          beforeClose +
          '</think>\n\n' +
          afterClose
        );
      }

      return content;
    }

    // ------------------------------------------
    // No opening tag yet
    //
    // Buffer the content because it may be
    // reasoning. We only know for certain when
    // </think> arrives.
    // ------------------------------------------

    const combined =
      this.stepFunBuffer +
      content;

    const closeIndex =
      combined.indexOf(
        '</think>'
      );

    // No closing tag yet.
    // Keep buffering.
    if (closeIndex === -1) {
      this.stepFunBuffer =
        combined;

      return '';
    }

    // ------------------------------------------
    // We found </think> without <think>
    // ------------------------------------------

    const reasoningText =
      combined.slice(
        0,
        closeIndex
      );

    const answerText =
      combined.slice(
        closeIndex +
          '</think>'.length
      );

    this.stepFunBuffer = '';

    this.reasoningSeen = true;

    this.reasoningOpen = false;

    this.stepFunFinishedThinking =
      true;

    return (
      '<think>\n' +
      reasoningText +
      '</think>\n\n' +
      answerText
    );
  }

  // ------------------------------------------
  // GENERAL NORMALIZATION
  // ------------------------------------------

  processContent(content) {
    if (!content) {
      return '';
    }

    // StepFun needs special handling.
    if (this.stepFun) {
      return this.processStepFunContent(
        content
      );
    }

    // Other models:
    // preserve explicit tags if supplied.
    return content;
  }

  // ------------------------------------------
  // SEPARATE REASONING FIELD
  // ------------------------------------------

  processSeparateReasoning(
    reasoning
  ) {
    if (
      !reasoning ||
      !SHOW_REASONING
    ) {
      return '';
    }

    if (
      !this.reasoningOpen
    ) {
      this.reasoningOpen = true;

      this.reasoningSeen = true;

      return (
        '<think>\n' +
        reasoning
      );
    }

    return reasoning;
  }

  // ------------------------------------------
  // CLOSE AT END
  // ------------------------------------------

  finish() {
    let output = '';

    // ------------------------------------------
    // StepFun:
    //
    // If no </think> ever arrived, do NOT blindly
    // label the buffered text as reasoning.
    //
    // We instead return it as normal content.
    // ------------------------------------------

    if (
      this.stepFun &&
      this.stepFunBuffer
    ) {
      output +=
        this.stepFunBuffer;

      this.stepFunBuffer = '';
    }

    // ------------------------------------------
    // Separate reasoning streams
    // ------------------------------------------

    if (
      SHOW_REASONING &&
      this.reasoningOpen
    ) {
      output +=
        '\n</think>\n';

      this.reasoningOpen = false;
    }

    return output;
  }
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

      models:
        Object.keys(
          MODEL_MAPPING
        )
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
    let upstreamResponse = null;

    try {
      const {
        model,
        messages,
        temperature,
        max_tokens,
        top_p,
        seed
      } = req.body;

      // ==========================================
      // VALIDATE MESSAGES
      // ==========================================

      if (
        !Array.isArray(messages)
      ) {
        return res.status(400).json({
          error: {
            message:
              '`messages` must be an array',

            type:
              'invalid_request_error',

            code: 400
          }
        });
      }

      // ==========================================
      // FALLBACK
      // ==========================================

      const nimModel =
        MODEL_MAPPING[model] ||
        FALLBACK_MODEL;

      // ==========================================
      // BUILD REQUEST
      // ==========================================

      const nimRequest = {
        model: nimModel,

        messages,

        temperature:
          temperature ?? 1.0,

        top_p:
          top_p ?? 0.95,

        max_tokens:
          max_tokens ?? 16384,

        stream: true,

        ...(seed !== undefined
          ? { seed }
          : {}),

        ...(ENABLE_THINKING_MODE
          ? buildThinkingConfig(
              nimModel
            )
          : {})
      };

      console.log(
        `[NIM] ${model || 'unknown'} -> ${nimModel}`
      );

      // ==========================================
      // NVIDIA REQUEST
      // ==========================================

      upstreamResponse =
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

            // Don't let Axios transform
            // the streaming response.
            transformResponse: [
              (data) => data
            ],

            // Allow us to handle NVIDIA
            // HTTP errors ourselves.
            validateStatus:
              (status) =>
                status >= 200 &&
                status < 300,

            timeout: 0,

            maxContentLength:
              Infinity,

            maxBodyLength:
              Infinity
          }
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

      // Flush headers immediately.
      res.flushHeaders?.();

      // ==========================================
      // REASONING NORMALIZER
      // ==========================================

      const normalizer =
        new ReasoningNormalizer(
          nimModel
        );

      // ==========================================
      // STREAM STATE
      // ==========================================

      let buffer = '';

      let finished = false;

      // ==========================================
      // CLEANUP
      // ==========================================

      const cleanup =
        () => {
          if (
            upstreamResponse?.data &&
            typeof upstreamResponse
              .data.destroy ===
              'function'
          ) {
            upstreamResponse.data.destroy();
          }
        };

      req.on(
        'close',
        () => {
          if (!res.writableEnded) {
            cleanup();
          }
        }
      );

      // ==========================================
      // HANDLE ONE SSE DATA LINE
      // ==========================================

      const processLine =
        (line) => {
          const trimmed =
            line.trim();

          // Ignore blank lines.
          if (!trimmed) {
            return;
          }

          // NVIDIA uses:
          //
          // data: {...}
          //
          if (
            !trimmed.startsWith(
              'data:'
            )
          ) {
            return;
          }

          const payload =
            trimmed
              .slice(5)
              .trim();

          // ========================================
          // DONE
          // ========================================

          if (
            payload === '[DONE]'
          ) {
            // Flush any remaining StepFun
            // buffered content.
            const finalContent =
              normalizer.finish();

            if (finalContent) {
              writeSSE(
                res,
                createContentChunk(
                  finalContent
                )
              );
            }

            writeDone(res);

            finished = true;

            if (
              !res.writableEnded
            ) {
              res.end();
            }

            return;
          }

          // ========================================
          // PARSE JSON
          // ========================================

          let data;

          try {
            data =
              JSON.parse(payload);
          } catch (err) {
            console.error(
              '[SSE] JSON parse error:',
              err.message
            );

            return;
          }

          // ========================================
          // CHOICES / DELTA
          // ========================================

          const choice =
            data?.choices?.[0];

          const delta =
            choice?.delta;

          if (!delta) {
            // Some providers may send metadata
            // chunks without delta content.
            writeSSE(res, data);

            return;
          }

          // ========================================
          // RAW REASONING
          // ========================================

          const reasoning =
            extractReasoning(
              delta
            );

          // ========================================
          // RAW CONTENT
          // ========================================

          const content =
            typeof delta.content ===
            'string'
              ? delta.content
              : '';

          let output = '';

          // ========================================
          // REASONING FIELD
          // ========================================

          if (
            SHOW_REASONING &&
            reasoning
          ) {
            output +=
              normalizer.processSeparateReasoning(
                reasoning
              );
          }

          // ========================================
          // CONTENT
          // ========================================

          if (content) {
            output +=
              normalizer.processContent(
                content
              );
          }

          // ========================================
          // COPY DATA
          // ========================================
          //
          // Do not mutate the provider object
          // unnecessarily.
          //

          if (
            output
          ) {
            data.choices[0].delta =
              {
                ...delta,

                content:
                  output
              };
          } else {
            // Keep the original delta except
            // remove raw reasoning fields.
            data.choices[0].delta =
              {
                ...delta
              };
          }

          // ========================================
          // REMOVE RAW REASONING
          // ========================================

          delete data
            .choices[0]
            .delta
            .reasoning;

          delete data
            .choices[0]
            .delta
            .reasoning_content;

          // ========================================
          // DON'T SEND EMPTY DELTAS
          // ========================================

          const finalDelta =
            data
              ?.choices?.[0]
              ?.delta;

          const hasContent =
            finalDelta &&
            Object.keys(
              finalDelta
            ).length > 0;

          if (
            !hasContent &&
            !output
          ) {
            // Preserve metadata chunks that contain
            // other useful fields.
            const choiceCopy =
              {
                ...data.choices[0]
              };

            delete choiceCopy.delta;

            if (
              Object.keys(
                choiceCopy
              ).length === 0
            ) {
              return;
            }
          }

          // ========================================
          // SEND
          // ========================================

          writeSSE(
            res,
            data
          );
        };

      // ==========================================
      // STREAM DATA
      // ==========================================

      upstreamResponse.data.on(
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
              /\r?\n/
            );

          // Keep incomplete line.
          buffer =
            lines.pop() || '';

          for (
            const line of lines
          ) {
            if (
              finished ||
              res.writableEnded
            ) {
              break;
            }

            processLine(line);
          }
        }
      );

      // ==========================================
      // STREAM END
      // ==========================================

      upstreamResponse.data.on(
        'end',
        () => {
          if (
            finished ||
            res.writableEnded
          ) {
            return;
          }

          // Process any final buffered SSE line.
          if (buffer.trim()) {
            processLine(buffer);
            buffer = '';
          }

          if (
            finished ||
            res.writableEnded
          ) {
            return;
          }

          // Flush remaining normalized content.
          const finalContent =
            normalizer.finish();

          if (finalContent) {
            writeSSE(
              res,
              createContentChunk(
                finalContent
              )
            );
          }

          writeDone(res);

          finished = true;

          res.end();
        }
      );

      // ==========================================
      // STREAM ERROR
      // ==========================================

      upstreamResponse.data.on(
        'error',
        (err) => {
          console.error(
            '[NIM STREAM ERROR]',
            err.message
          );

          if (
            !res.writableEnded
          ) {
            // If the SSE response has already
            // started, we cannot safely replace it
            // with an HTTP JSON error.
            //
            // Just close the stream.
            res.end();
          }
        }
      );
    } catch (error) {
      console.error(
        '[PROXY ERROR]',
        error?.message ||
          'Unknown error'
      );

      // ==========================================
      // IMPORTANT:
      //
      // Never JSON.stringify the entire Axios
      // error or response stream.
      //
      // That is what causes:
      //
      // TypeError:
      // Converting circular structure to JSON
      // ==========================================

      const safe =
        await getSafeErrorPayload(
          error
        );

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

      res
        .status(safe.status)
        .json(safe.body);
    }
  }
);

// ============================================
// 404
// ============================================

app.use(
  (req, res) => {
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
// GLOBAL ERROR HANDLER
// ============================================

app.use(
  async (
    error,
    req,
    res,
    next
  ) => {
    console.error(
      '[GLOBAL ERROR]',
      error?.message ||
        'Unknown error'
    );

    if (
      res.headersSent
    ) {
      return next(error);
    }

    const safe =
      await getSafeErrorPayload(
        error
      );

    res
      .status(safe.status)
      .json(safe.body);
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
      'NVIDIA NIM OpenAI-compatible proxy'
    );

    console.log(
      '============================================'
    );

    console.log(
      `Port: ${PORT}`
    );

    console.log(
      `NIM API: ${NIM_API_BASE}`
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
      'Models:'
    );

    for (
      const [
        alias,
        nimModel
      ] of Object.entries(
        MODEL_MAPPING
      )
    ) {
      console.log(
        `  ${alias} -> ${nimModel}`
      );
    }

    console.log(
      '============================================'
    );
  }
);
