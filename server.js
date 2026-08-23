// server.js
// OpenAI-compatible NVIDIA NIM proxy
//
// NON-STREAMING
//
// Models:
// - glm-5.2        -> z-ai/glm-5.2
// - kimi-k2.6      -> moonshotai/kimi-k2.6
// - deepseek-v4    -> deepseek-ai/deepseek-v4-pro
// - step-3.7-flash  -> stepfun-ai/step-3.7-flash
//
// IMPORTANT:
// - NVIDIA endpoint is hard-coded.
// - No retry logic.
// - No exponential backoff.
// - No streaming.
// - Step 3.7 Flash reasoning is disabled.
// - Other models can expose reasoning when enabled.
//

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

// HARD-CODED NVIDIA NIM CHAT COMPLETIONS ENDPOINT
const NIM_CHAT_ENDPOINT =
  'https://integrate.api.nvidia.com/v1/chat/completions';

const NIM_API_KEY =
  process.env.NIM_API_KEY;

// Display reasoning in <think> blocks
const SHOW_REASONING = true;

// Enable model-specific thinking parameters
const ENABLE_THINKING_MODE = true;

// DeepSeek reasoning effort
const REASONING_EFFORT = 'low';

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
    .replace(/<think>\s*/gi, '')
    .replace(/\s*<\/think>/gi, '');
}

// ============================================
// SAFE ERROR SERIALIZATION
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

    // String response
    if (
      typeof data === 'string'
    ) {
      return data;
    }

    // JSON/object response
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

  // ============================================
  // NETWORK / AXIOS ERROR
  // ============================================

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
// HEALTH
// ============================================

app.get('/health', (req, res) => {

  res.json({
    status: 'ok',

    streaming_only: false,

    streaming: false,

    reasoning_display:
      SHOW_REASONING,

    thinking_mode:
      ENABLE_THINKING_MODE,

    fallback_model:
      FALLBACK_MODEL,

    endpoint:
      NIM_CHAT_ENDPOINT,

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

      created:
        Math.floor(
          Date.now() / 1000
        ),

      owned_by:
        'nvidia-nim-proxy'
    }))
  });
});

// ============================================
// CHAT COMPLETIONS
// NON-STREAMING
// ============================================

app.post(
  '/v1/chat/completions',
  async (req, res) => {

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

      const nimModel =
        MODEL_MAPPING[model] ||
        FALLBACK_MODEL;

      const step37 =
        isStep37Flash(nimModel);

      // ============================================
      // BUILD REQUEST
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

        // ==========================================
        // NON-STREAMING
        // ==========================================

        stream: false
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

      } else if (step37) {

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
        `[Proxy] ${model || 'unknown'} -> ${nimModel}` +
        `${step37
          ? ' [REASONING DISABLED]'
          : ' [REASONING ENABLED]'}`
      );

      console.log(
        '[NVIDIA Request] POST',
        NIM_CHAT_ENDPOINT
      );

      // ============================================
      // NVIDIA REQUEST
      // ============================================
      //
      // IMPORTANT:
      // There is intentionally NO retry logic here.
      //
      // If NVIDIA returns 429, 4xx, 5xx, etc.,
      // that response is passed back immediately.
      //
      // stream: false means Axios receives the
      // complete JSON response.
      //

      const response =
        await axios.post(
          NIM_CHAT_ENDPOINT,
          nimRequest,
          {
            headers: {
              Authorization:
                `Bearer ${NIM_API_KEY}`,

              'Content-Type':
                'application/json',

              Accept:
                'application/json'
            },

            responseType:
              'json',

            timeout:
              300000,

            validateStatus:
              () => true
          }
        );

      // ============================================
      // LOG STATUS
      // ============================================

      console.log(
        `[NVIDIA Response] HTTP ${response.status}`
      );

      // ============================================
      // NVIDIA ERROR
      // ============================================

      if (
        response.status < 200 ||
        response.status >= 300
      ) {

        const data =
          response.data;

        let message;

        if (
          typeof data === 'string'
        ) {

          message = data;

        } else if (
          data?.error?.message
        ) {

          message =
            data.error.message;

        } else if (
          data?.message
        ) {

          message =
            data.message;

        } else {

          try {

            message =
              JSON.stringify(
                data
              );

          } catch {

            message =
              `NVIDIA API returned HTTP ${response.status}`;
          }
        }

        console.error(
          `[NVIDIA Error] HTTP ${response.status}: ${message}`
        );

        return res
          .status(response.status)
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

      // ============================================
      // SUCCESS
      // ============================================

      const data =
        response.data;

      // ============================================
      // SAFETY CHECK
      // ============================================

      if (
        !data ||
        typeof data !== 'object'
      ) {

        console.error(
          '[Proxy] NVIDIA returned an invalid JSON response'
        );

        return res.status(502).json({
          error: {
            message:
              'NVIDIA API returned an invalid response',

            type:
              'nvidia_api_error',

            code: 502
          }
        });
      }

      // ============================================
      // STEP 3.7 FLASH
      // REASONING DISABLED
      // ============================================

      if (step37) {

        const choice =
          data?.choices?.[0];

        const message =
          choice?.message;

        if (
          message &&
          typeof message.content ===
            'string'
        ) {

          message.content =
            cleanStepContent(
              message.content
            );
        }

        if (message) {

          delete message.reasoning;

          delete message.reasoning_content;
        }

        console.log(
          '[NVIDIA Response] Step 3.7 Flash response received'
        );

        return res
          .status(200)
          .json(data);
      }

      // ============================================
      // OTHER MODELS
      // REASONING PROCESSING
      // ============================================

      const choice =
        data?.choices?.[0];

      const message =
        choice?.message;

      if (
        message &&
        typeof message === 'object'
      ) {

        const reasoning =
          extractReasoning(
            message
          );

        const content =
          typeof message.content ===
          'string'
            ? message.content
            : '';

        let finalContent =
          '';

        // ==========================================
        // REASONING
        // ==========================================

        if (
          SHOW_REASONING &&
          reasoning
        ) {

          finalContent +=
            '<think>\n';

          finalContent +=
            reasoning;

          finalContent +=
            '\n</think>\n\n';
        }

        // ==========================================
        // CONTENT
        // ==========================================

        if (content) {

          finalContent +=
            content;
        }

        // ==========================================
        // REPLACE CONTENT
        // ==========================================

        if (finalContent) {

          message.content =
            finalContent;
        }

        // ==========================================
        // REMOVE RAW REASONING
        // ==========================================

        delete message.reasoning;

        delete message.reasoning_content;
      }

      // ============================================
      // SEND FINAL RESPONSE
      // ============================================

      console.log(
        '[NVIDIA Response] Complete'
      );

      return res
        .status(200)
        .json(data);

    } catch (error) {

      // ============================================
      // SAFE ERROR HANDLING
      // ============================================

      logProxyError(error);

      if (
        res.headersSent ||
        res.writableEnded
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
      `NVIDIA endpoint: ${NIM_CHAT_ENDPOINT}`
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
      'Streaming: DISABLED'
    );

    console.log(
      'Retries: DISABLED'
    );

    console.log(
      'Step 3.7 Flash reasoning: DISABLED'
    );

    console.log(
      '============================================'
    );
  }
);
