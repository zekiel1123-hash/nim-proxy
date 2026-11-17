// ============================================================
// NVIDIA NIM -> OpenAI-Compatible Streaming Proxy
// ============================================================
//
// CURRENT MODELS
//
// muse-glimmer-30b       -> meta/muse-glimmer-30b
// kimi-k3.1-pro          -> moonshotai/kimi-k3.1-pro
// step-5-preview        -> stepfun-ai/step-5-preview
// glm-5-3-flash         -> z-ai/glm-5.3-flash
// gemma-4-31b-it        -> google/gemma-4-31b-it
//
// FALLBACK
//
// muse-glimmer-30b
//
// Note: NVIDIA currently marks the Kimi K3.1 Pro hosted endpoint as
// preview/playground-only and not available for programmatic API access.
// The model is registered here, but NVIDIA may reject live API requests.

// ============================================================

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT =
  process.env.PORT || 3000;

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY =
  process.env.NIM_API_KEY;

// ============================================================
// CONFIGURATION
// ============================================================

const SHOW_REASONING =
  String(
    process.env.SHOW_REASONING || "true"
  ).toLowerCase() === "true";

const DEFAULT_KIMI_REASONING =
  process.env.KIMI_REASONING_EFFORT ||
  "high";

const DEFAULT_MUSE_REASONING =
  process.env.MUSE_REASONING_EFFORT ||
  "high";

const DEFAULT_GLM_REASONING =
  process.env.GLM_REASONING_EFFORT ||
  "max";

const DEFAULT_GEMMA_THINKING =
  String(
    process.env.GEMMA_ENABLE_THINKING || "true"
  ).toLowerCase() === "true";

// ============================================================
// FALLBACK
// ============================================================

const FALLBACK_MODEL =
  "muse-glimmer-30b";

// ============================================================
// MODEL DEFINITIONS
// ============================================================

const MODELS = {
  // ==========================================================
  // META - MUSE GLIMMER 30B
  // ==========================================================

  "muse-glimmer-30b": {
    id: "muse-glimmer-30b",
    upstream: "meta/muse-glimmer-30b",
    owner: "meta",

    modalities: {
      image: true,
      video: false
    },

    contextWindow: 131072,
    maxTokens: 131072,
    defaultMaxTokens: 2048,

    temperature: 0.95,
    top_p: 1.0,

    reasoning: {
      type: "reasoning_effort",
      allowed: [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "max"
      ],
      default: DEFAULT_MUSE_REASONING
    },

    supports: {
      temperature: true,
      top_p: true,
      presence_penalty: true,
      frequency_penalty: true,
      seed: true,
      tools: true,
      stream_options: false,
      stop: true,
      chat_template_kwargs: true
    }
  },

  // ==========================================================
  // MOONSHOT AI - KIMI K3.1 PRO
  // ==========================================================

  "kimi-k3.1-pro": {
    id: "kimi-k3.1-pro",
    upstream: "moonshotai/kimi-k3.1-pro",
    owner: "moonshotai",

    // The published NVIDIA chat endpoint documents image parts but
    // does not document a video_url message part for this route.
    modalities: {
      image: true,
      video: false
    },

    contextWindow: 1048576,
    maxTokens: 65536,
    defaultMaxTokens: 16384,

    temperature: 1.0,

    reasoning: {
      type: "reasoning_effort",
      allowed: [
        "low",
        "medium",
        "high",
        "max"
      ],
      default: DEFAULT_KIMI_REASONING
    },

    // The NVIDIA model page currently says this hosted endpoint is
    // preview/playground-only, not open for programmatic API calls.
    // Do not silently substitute Muse if NVIDIA rejects a Kimi request.
    fallbackOnError: false,

    supports: {
      temperature: true,
      top_p: false,
      presence_penalty: false,
      frequency_penalty: false,
      seed: true,
      tools: true,
      stream_options: true,
      stop: false,
      chat_template_kwargs: false
    }
  },

  // ==========================================================
  // STEPFUN AI - STEP 5 PREVIEW
  // ==========================================================

  "step-5-preview": {
    id: "step-5-preview",
    upstream: "stepfun-ai/step-5-preview",
    owner: "stepfun-ai",

    modalities: {
      image: true,
      video: false
    },

    contextWindow: 1048576,
    maxTokens: 1048576,
    defaultMaxTokens: 4096,

    temperature: 0.5,
    top_p: 1.0,

    // Step 5 Preview reasoning is enabled by the checkpoint.
    // NVIDIA specifically says not to set enable_thinking=false.
    reasoning: {
      type: "always_on"
    },

    supports: {
      temperature: true,
      top_p: true,
      presence_penalty: true,
      frequency_penalty: true,
      seed: true,
      tools: true,
      stream_options: false,
      stop: true,
      chat_template_kwargs: true
    }
  },

  // ==========================================================
  // Z.AI - GLM 5.3 FLASH
  // ==========================================================

  "glm-5-3-flash": {
    id: "glm-5-3-flash",
    upstream: "z-ai/glm-5.3-flash",
    owner: "z-ai",

    modalities: {
      image: true,
      video: false
    },

    contextWindow: 1048576,
    defaultMaxTokens: 1024,

    temperature: 0.5,
    top_p: 1.0,

    reasoning: {
      type: "reasoning_effort",
      allowed: [
        "low",
        "high",
        "max"
      ],
      default: DEFAULT_GLM_REASONING
    },

    supports: {
      temperature: true,
      top_p: true,
      presence_penalty: false,
      frequency_penalty: false,
      seed: false,
      tools: true,
      stream_options: false,
      stop: false,
      chat_template_kwargs: true
    }
  },

  // ==========================================================
  // GOOGLE - GEMMA 4 31B IT
  // ==========================================================

  "gemma-4-31b-it": {
    id: "gemma-4-31b-it",
    upstream: "google/gemma-4-31b-it",
    owner: "google",

    modalities: {
      image: true,
      video: false
    },

    contextWindow: 262144,
    maxTokens: 32768,
    defaultMaxTokens: 4096,

    temperature: 1.0,
    top_p: 0.95,

    reasoning: {
      type: "chat_template_thinking",
      default: DEFAULT_GEMMA_THINKING
    },

    supports: {
      temperature: true,
      top_p: true,
      presence_penalty: false,
      frequency_penalty: false,
      seed: false,
      tools: true,
      stream_options: false,
      stop: false,
      chat_template_kwargs: true
    }
  }
};

// ============================================================
// EXPRESS MIDDLEWARE
// ============================================================

app.use(
  cors()
);

app.use(
  express.json({
    limit: "100mb"
  })
);

app.use(
  express.urlencoded({
    limit: "100mb",
    extended: true
  })
);

// ============================================================
// MODEL ALIASES
// ============================================================

const MODEL_ALIASES = {
  "gemma-4-31b": "gemma-4-31b-it",
  "gemma-4-31b-it": "gemma-4-31b-it",

  "z-ai-glm-5.3-flash": "glm-5-3-flash",
  "z-ai/glm-5.3-flash": "glm-5-3-flash",

  "stepfun-ai/step-5-preview": "step-5-preview",

  "moonshotai/kimi-k3.1-pro": "kimi-k3.1-pro",

  "meta/muse-glimmer-30b": "muse-glimmer-30b"
};

// ============================================================
// HELPERS
// ============================================================

function getModel(modelName) {
  if (!modelName) {
    return null;
  }

  if (MODELS[modelName]) {
    return MODELS[modelName];
  }

  const alias =
    MODEL_ALIASES[modelName];

  if (alias && MODELS[alias]) {
    return MODELS[alias];
  }

  return null;
}

// ============================================================
// ERROR MESSAGE EXTRACTION
// ============================================================

function extractErrorMessage(errorData) {
  if (!errorData) {
    return "Unknown NVIDIA API error";
  }

  if (typeof errorData === "string") {
    return errorData ||
      "Unknown NVIDIA API error";
  }

  if (errorData.error) {
    if (typeof errorData.error === "string") {
      return errorData.error;
    }

    if (errorData.error.message) {
      return String(
        errorData.error.message
      );
    }
  }

  if (errorData.message) {
    return String(
      errorData.message
    );
  }

  try {
    return JSON.stringify(errorData);
  } catch {
    return "Unknown NVIDIA API error";
  }
}

// ============================================================
// OPENAI ERROR RESPONSE
// ============================================================

function sendOpenAIError(
  res,
  status,
  message,
  type = "proxy_error"
) {
  if (res.headersSent) {
    try {
      if (!res.writableEnded) {
        res.end();
      }
    } catch {
      // Ignore.
    }

    return;
  }

  return res
    .status(status)
    .json({
      error: {
        message:
          message ||
          "Proxy error",

        type,

        param: null,

        code: null
      }
    });
}

// ============================================================
// THINK TAG CLEANUP
// ============================================================

function stripThinkTags(text) {
  if (typeof text !== "string") {
    return text;
  }

  return text
    .replace(
      /<think>[\s\S]*?<\/think>/gi,
      ""
    )
    .replace(
      /<\/?think>/gi,
      ""
    );
}

// ============================================================
// MEDIA DETECTION
// ============================================================

function containsImage(messages) {
  if (!Array.isArray(messages)) {
    return false;
  }

  for (const message of messages) {
    if (!Array.isArray(message?.content)) {
      continue;
    }

    for (const part of message.content) {
      if (part?.type === "image_url") {
        return true;
      }
    }
  }

  return false;
}

function containsVideo(messages) {
  if (!Array.isArray(messages)) {
    return false;
  }

  for (const message of messages) {
    if (!Array.isArray(message?.content)) {
      continue;
    }

    for (const part of message.content) {
      if (part?.type === "video_url") {
        return true;
      }
    }
  }

  return false;
}

// ============================================================
// MESSAGE VALIDATION
// ============================================================

function validateMessages(messages) {
  if (!Array.isArray(messages)) {
    return "messages must be an array";
  }

  if (messages.length === 0) {
    return "messages cannot be empty";
  }

  for (
    let i = 0;
    i < messages.length;
    i++
  ) {
    const message = messages[i];

    if (
      !message ||
      typeof message !== "object"
    ) {
      return (
        `messages[${i}] must be an object`
      );
    }

    if (typeof message.role !== "string") {
      return (
        `messages[${i}].role must be a string`
      );
    }

    if (
      message.content === undefined &&
      message.tool_calls === undefined
    ) {
      return (
        `messages[${i}] must contain content or tool_calls`
      );
    }
  }

  return null;
}

// ============================================================
// REASONING VALIDATION
// ============================================================

function getReasoningEffort(
  requestBody,
  model
) {
  if (
    model.reasoning.type !== "reasoning_effort"
  ) {
    return null;
  }

  const requested =
    requestBody.reasoning_effort ??
    model.reasoning.default;

  if (
    model.reasoning.allowed.includes(
      requested
    )
  ) {
    return requested;
  }

  console.warn(
    `[Reasoning] Invalid reasoning_effort "${requested}" ` +
    `for ${model.id}; using ${model.reasoning.default}`
  );

  return model.reasoning.default;
}

// ============================================================
// BUILD NVIDIA REQUEST
// ============================================================

function buildNvidiaRequest(
  body,
  model
) {
  const request = {
    model: model.upstream,
    messages: body.messages,
    stream: true
  };

  // ==========================================================
  // TEMPERATURE
  // ==========================================================

  if (model.supports.temperature) {
    if (body.temperature !== undefined) {
      request.temperature =
        body.temperature;
    } else if (model.temperature !== undefined) {
      request.temperature =
        model.temperature;
    }
  }

  // ==========================================================
  // TOP P
  // ==========================================================

  if (model.supports.top_p) {
    if (body.top_p !== undefined) {
      request.top_p =
        body.top_p;
    } else if (model.top_p !== undefined) {
      request.top_p =
        model.top_p;
    }
  }

  // ==========================================================
  // MAX TOKENS
  // ==========================================================

  let requestedMaxTokens =
    body.max_tokens;

  if (requestedMaxTokens === undefined) {
    requestedMaxTokens =
      body.max_completion_tokens;
  }

  if (requestedMaxTokens === undefined) {
    requestedMaxTokens =
      model.defaultMaxTokens ||
      model.maxTokens;
  }

  if (
    model.maxTokens &&
    requestedMaxTokens > model.maxTokens
  ) {
    requestedMaxTokens =
      model.maxTokens;
  }

  if (
    requestedMaxTokens !== undefined &&
    requestedMaxTokens !== null
  ) {
    request.max_tokens =
      requestedMaxTokens;
  }

  // ==========================================================
  // REASONING EFFORT
  // ==========================================================

  if (
    model.reasoning.type === "reasoning_effort"
  ) {
    request.reasoning_effort =
      getReasoningEffort(body, model);
  }

  // ==========================================================
  // CHAT TEMPLATE THINKING
  // ==========================================================

  if (
    model.reasoning.type === "chat_template_thinking"
  ) {
    request.chat_template_kwargs = {
      enable_thinking:
        model.reasoning.default
    };
  }

  // ==========================================================
  // STEP 5 REASONING
  // ==========================================================

  // Step 5 Preview has always-on reasoning.
  // Do not send enable_thinking=false.

  if (
    model.reasoning.type === "always_on" &&
    model.supports.chat_template_kwargs
  ) {
    if (body.chat_template_kwargs !== undefined) {
      request.chat_template_kwargs = {
        ...body.chat_template_kwargs,
        enable_thinking: true
      };
    }
  }

  // ==========================================================
  // CUSTOM CHAT TEMPLATE PARAMETERS
  // ==========================================================

  if (
    model.supports.chat_template_kwargs &&
    body.chat_template_kwargs &&
    model.reasoning.type !== "always_on" &&
    model.reasoning.type !== "chat_template_thinking"
  ) {
    request.chat_template_kwargs = {
      ...body.chat_template_kwargs
    };
  }

  // ==========================================================
  // SEED
  // ==========================================================

  if (
    model.supports.seed &&
    body.seed !== undefined
  ) {
    request.seed =
      body.seed;
  }

  // ==========================================================
  // TOOLS
  // ==========================================================

  if (
    model.supports.tools &&
    Array.isArray(body.tools) &&
    body.tools.length > 0
  ) {
    request.tools =
      body.tools;
  }

  // ==========================================================
  // TOOL CHOICE
  // ==========================================================

  if (
    model.supports.tools &&
    body.tool_choice !== undefined
  ) {
    request.tool_choice =
      body.tool_choice;
  }

  // ==========================================================
  // STOP
  // ==========================================================

  if (
    model.supports.stop &&
    body.stop !== undefined
  ) {
    request.stop =
      body.stop;
  }

  // ==========================================================
  // PRESENCE PENALTY
  // ==========================================================

  if (
    model.supports.presence_penalty &&
    body.presence_penalty !== undefined
  ) {
    request.presence_penalty =
      body.presence_penalty;
  }

  // ==========================================================
  // FREQUENCY PENALTY
  // ==========================================================

  if (
    model.supports.frequency_penalty &&
    body.frequency_penalty !== undefined
  ) {
    request.frequency_penalty =
      body.frequency_penalty;
  }

  // ==========================================================
  // STREAM OPTIONS
  // ==========================================================

  if (
    model.supports.stream_options &&
    body.stream_options !== undefined
  ) {
    request.stream_options =
      body.stream_options;
  }

  return request;
}

// ============================================================
// GET MODELS
// ============================================================

app.get(
  "/v1/models",
  (req, res) => {
    const models =
      Object.values(MODELS).map(
        model => ({
          id: model.id,
          object: "model",
          created: 0,
          owned_by: model.owner
        })
      );

    res.json({
      object: "list",
      data: models
    });
  }
);

// ============================================================
// ROOT
// ============================================================

app.get(
  "/",
  (req, res) => {
    res.json({
      object: "proxy",
      status: "ok",
      streaming: true,
      fallback_model: FALLBACK_MODEL,
      models: Object.keys(MODELS)
    });
  }
);

// ============================================================
// STREAMING CHAT COMPLETIONS
// ============================================================

app.post(
  "/v1/chat/completions",
  async (
    req,
    res
  ) => {
    const body =
      req.body || {};

    // ========================================================
    // STREAMING ONLY
    // ========================================================

    if (body.stream !== true) {
      return sendOpenAIError(
        res,
        400,
        "This proxy supports streaming chat completions only. Set stream=true.",
        "stream_required"
      );
    }

    // ========================================================
    // VALIDATE MESSAGES
    // ========================================================

    const validationError =
      validateMessages(body.messages);

    if (validationError) {
      return sendOpenAIError(
        res,
        400,
        validationError,
        "invalid_messages"
      );
    }

    // ========================================================
    // SELECT MODEL
    // ========================================================

    const requestedModel =
      body.model;

    let selectedModel =
      getModel(requestedModel);

    // ========================================================
    // UNKNOWN MODEL -> FALLBACK
    // ========================================================

    if (!selectedModel) {
      console.warn(
        `[Fallback] Unknown model "${requestedModel}". ` +
        `Using "${FALLBACK_MODEL}".`
      );

      selectedModel =
        MODELS[FALLBACK_MODEL];
    }

    // ========================================================
    // MULTIMODAL VALIDATION
    // ========================================================

    if (
      containsImage(body.messages) &&
      !selectedModel.modalities.image
    ) {
      return sendOpenAIError(
        res,
        400,
        `"${selectedModel.id}" does not support image input.`,
        "multimodal_not_supported"
      );
    }

    if (
      containsVideo(body.messages) &&
      !selectedModel.modalities.video
    ) {
      return sendOpenAIError(
        res,
        400,
        `"${selectedModel.id}" does not support video input.`,
        "multimodal_not_supported"
      );
    }

    // ========================================================
    // BUILD NVIDIA REQUEST
    // ========================================================

    const primaryRequest =
      buildNvidiaRequest(
        body,
        selectedModel
      );

    console.log(
      `[Request] ${selectedModel.id} -> ` +
      `${selectedModel.upstream} ` +
      `[STREAMING]`
    );

    console.log(
      `[Endpoint] ${NIM_API_BASE}`
    );

    console.log(
      `[Request Config] ` +
      `temperature=${primaryRequest.temperature ?? "default"} ` +
      `top_p=${primaryRequest.top_p ?? "not_sent"} ` +
      `max_tokens=${primaryRequest.max_tokens ?? "default"} ` +
      `reasoning_effort=${primaryRequest.reasoning_effort ?? "not_sent"}`
    );

    // ========================================================
    // NVIDIA REQUEST
    // ========================================================

    let upstreamResponse;

    try {
      upstreamResponse =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          primaryRequest,
          {
            headers: {
              Authorization:
                `Bearer ${NIM_API_KEY}`,

              "Content-Type":
                "application/json",

              Accept:
                "text/event-stream"
            },

            responseType: "stream",

            timeout: 0,

            validateStatus: () => true
          }
        );

    } catch (error) {
      console.error(
        "[NVIDIA Connection Error]",
        error.message
      );

      // ======================================================
      // CONNECTION FALLBACK
      // ======================================================

      if (
        selectedModel.id !== FALLBACK_MODEL &&
        selectedModel.fallbackOnError !== false
      ) {
        console.warn(
          `[Fallback] ${selectedModel.id} connection failed. ` +
          `Retrying with ${FALLBACK_MODEL}.`
        );

        try {
          const fallbackModel =
            MODELS[FALLBACK_MODEL];

          const fallbackRequest =
            buildNvidiaRequest(
              {
                ...body,
                model: FALLBACK_MODEL
              },
              fallbackModel
            );

          upstreamResponse =
            await axios.post(
              `${NIM_API_BASE}/chat/completions`,
              fallbackRequest,
              {
                headers: {
                  Authorization:
                    `Bearer ${NIM_API_KEY}`,

                  "Content-Type":
                    "application/json",

                  Accept:
                    "text/event-stream"
                },

                responseType: "stream",

                timeout: 0,

                validateStatus: () => true
              }
            );

          selectedModel =
            fallbackModel;

        } catch (fallbackError) {
          console.error(
            "[Fallback Connection Error]",
            fallbackError.message
          );

          return sendOpenAIError(
            res,
            502,
            fallbackError.message,
            "nvidia_fallback_error"
          );
        }

      } else {
        return sendOpenAIError(
          res,
          502,
          error.message,
          "nvidia_connection_error"
        );
      }
    }

    // ========================================================
    // LOG UPSTREAM RESPONSE
    // ========================================================

    console.log(
      `[NVIDIA Response] status=${upstreamResponse.status}`
    );

    console.log(
      `[NVIDIA Response] content-type=` +
      `${upstreamResponse.headers?.["content-type"] || "unknown"}`
    );

    // ========================================================
    // UPSTREAM HTTP ERROR BEFORE STREAM
    // ========================================================

    if (
      upstreamResponse.status < 200 ||
      upstreamResponse.status >= 300
    ) {
      let errorText = "";

      try {
        for await (
          const chunk of upstreamResponse.data
        ) {
          errorText +=
            chunk.toString("utf8");

          if (errorText.length > 100000) {
            break;
          }
        }
      } catch {
        // Ignore stream-read failure.
      }

      let parsedError =
        errorText;

      try {
        parsedError =
          JSON.parse(errorText);
      } catch {
        // Keep string.
      }

      const upstreamMessage =
        extractErrorMessage(parsedError);

      console.error(
        `[NVIDIA HTTP ${upstreamResponse.status}] ` +
        `${selectedModel.upstream}: ` +
        upstreamMessage
      );

      // ======================================================
      // HTTP FALLBACK
      // ======================================================

      if (
        selectedModel.id !== FALLBACK_MODEL &&
        selectedModel.fallbackOnError !== false
      ) {
        console.warn(
          `[Fallback] ${selectedModel.id} returned HTTP ` +
          `${upstreamResponse.status}. ` +
          `Retrying with ${FALLBACK_MODEL}.`
        );

        try {
          const fallbackModel =
            MODELS[FALLBACK_MODEL];

          const fallbackRequest =
            buildNvidiaRequest(
              {
                ...body,
                model: FALLBACK_MODEL
              },
              fallbackModel
            );

          const fallbackResponse =
            await axios.post(
              `${NIM_API_BASE}/chat/completions`,
              fallbackRequest,
              {
                headers: {
                  Authorization:
                    `Bearer ${NIM_API_KEY}`,

                  "Content-Type":
                    "application/json",

                  Accept:
                    "text/event-stream"
                },

                responseType: "stream",

                timeout: 0,

                validateStatus: () => true
              }
            );

          if (
            fallbackResponse.status < 200 ||
            fallbackResponse.status >= 300
          ) {
            let fallbackText = "";

            try {
              for await (
                const chunk of fallbackResponse.data
              ) {
                fallbackText +=
                  chunk.toString("utf8");

                if (fallbackText.length > 100000) {
                  break;
                }
              }
            } catch {
              // Ignore.
            }

            let fallbackParsed =
              fallbackText;

            try {
              fallbackParsed =
                JSON.parse(fallbackText);
            } catch {
              // Keep string.
            }

            const fallbackMessage =
              extractErrorMessage(fallbackParsed);

            console.error(
              `[Fallback HTTP ${fallbackResponse.status}] ` +
              fallbackMessage
            );

            return sendOpenAIError(
              res,
              fallbackResponse.status,
              fallbackMessage,
              "nvidia_fallback_error"
            );
          }

          upstreamResponse =
            fallbackResponse;

          selectedModel =
            fallbackModel;

          console.log(
            `[Fallback] Now streaming from ` +
            `${selectedModel.upstream}`
          );

        } catch (fallbackError) {
          console.error(
            "[Fallback Error]",
            fallbackError.message
          );

          return sendOpenAIError(
            res,
            502,
            fallbackError.message,
            "nvidia_fallback_error"
          );
        }

      } else {
        return sendOpenAIError(
          res,
          upstreamResponse.status,
          upstreamMessage,
          "nvidia_api_error"
        );
      }
    }

    // ========================================================
    // SSE RESPONSE HEADERS
    // ========================================================

    res.status(200);

    res.setHeader(
      "Content-Type",
      "text/event-stream; charset=utf-8"
    );

    res.setHeader(
      "Cache-Control",
      "no-cache, no-transform"
    );

    res.setHeader(
      "Connection",
      "keep-alive"
    );

    res.setHeader(
      "X-Accel-Buffering",
      "no"
    );

    if (typeof res.flushHeaders === "function") {
      res.flushHeaders();
    }

    // ========================================================
    // STREAM STATE
    // ========================================================

    let buffer = "";

    let finished = false;

    // ========================================================
    // WRITE SSE
    // ========================================================

    function writeSSE(data) {
      if (
        finished ||
        res.writableEnded
      ) {
        return;
      }

      try {
        res.write(
          `data: ${JSON.stringify(data)}\n\n`
        );
      } catch (error) {
        console.error(
          "[SSE Write Error]",
          error.message
        );
      }
    }

    // ========================================================
    // FINISH STREAM
    // ========================================================

    function finishStream() {
      if (finished) {
        return;
      }

      finished = true;

      try {
        if (!res.writableEnded) {
          res.write(
            "data: [DONE]\n\n"
          );
        }
      } catch {
        // Client may already be gone.
      }

      if (!res.writableEnded) {
        res.end();
      }
    }

    // ========================================================
    // PROCESS SSE LINE
    // ========================================================

    function processSSELine(line) {
      line =
        line.replace(/\r$/, "");

      if (!line.trim()) {
        return;
      }

      if (line.startsWith(":")) {
        return;
      }

      if (!line.startsWith("data:")) {
        return;
      }

      const raw =
        line
          .slice(5)
          .trim();

      if (raw === "[DONE]") {
        finishStream();
        return;
      }

      let parsed;

      try {
        parsed =
          JSON.parse(raw);

      } catch (error) {
        console.error(
          "[SSE JSON Parse Error]",
          error.message,
          raw.substring(0, 500)
        );

        return;
      }

      // ======================================================
      // NORMALIZE RESPONSE
      // ======================================================

      if (Array.isArray(parsed.choices)) {
        for (const choice of parsed.choices) {
          if (choice?.delta) {

            if (
              typeof choice.delta.content === "string"
            ) {
              choice.delta.content =
                stripThinkTags(
                  choice.delta.content
                );
            }

            if (!SHOW_REASONING) {
              delete choice.delta.reasoning;
              delete choice.delta.reasoning_content;
            }
          }
        }
      }

      // ======================================================
      // SEND TO CLIENT
      // ======================================================

      writeSSE(parsed);
    }

    // ========================================================
    // UPSTREAM DATA
    // ========================================================

    upstreamResponse.data.on(
      "data",
      chunk => {
        if (
          finished ||
          res.writableEnded
        ) {
          return;
        }

        buffer +=
          chunk.toString("utf8");

        const lines =
          buffer.split("\n");

        buffer =
          lines.pop() || "";

        for (const line of lines) {
          if (finished) {
            break;
          }

          processSSELine(line);
        }
      }
    );

    // ========================================================
    // UPSTREAM END
    // ========================================================

    upstreamResponse.data.on(
      "end",
      () => {
        if (buffer.trim()) {
          processSSELine(buffer);
        }

        finishStream();
      }
    );

    // ========================================================
    // UPSTREAM ERROR
    // ========================================================

    upstreamResponse.data.on(
      "error",
      error => {
        console.error(
          "[NVIDIA Stream Error]",
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
              "NVIDIA streaming error",

            type: "stream_error"
          }
        });

        finished = true;

        if (!res.writableEnded) {
          res.end();
        }
      }
    );

    // ========================================================
    // CLIENT DISCONNECT
    // ========================================================

    req.on(
      "close",
      () => {
        if (finished) {
          return;
        }

        finished = true;

        if (upstreamResponse?.data?.destroy) {
          upstreamResponse.data.destroy();
        }
      }
    );
  }
);

// ============================================================
// 404
// ============================================================

app.use(
  (
    req,
    res
  ) => {
    if (res.headersSent) {
      return;
    }

    sendOpenAIError(
      res,
      404,
      `Endpoint ${req.path} not found`,
      "not_found"
    );
  }
);

// ============================================================
// START SERVER
// ============================================================

app.listen(
  PORT,
  () => {
    console.log(
      "=================================================="
    );

    console.log(
      " NVIDIA NIM OpenAI-Compatible Proxy"
    );

    console.log(
      "=================================================="
    );

    console.log(
      `Port: ${PORT}`
    );

    console.log(
      `NVIDIA API: ${NIM_API_BASE}`
    );

    console.log(
      `API key configured: ${
        NIM_API_KEY
          ? "YES"
          : "NO"
      }`
    );

    console.log(
      "Streaming: REQUIRED"
    );

    console.log(
      `Reasoning display: ${
        SHOW_REASONING
          ? "ON"
          : "OFF"
      }`
    );

    console.log(
      `Fallback model: ${FALLBACK_MODEL}`
    );

    console.log(
      "--------------------------------------------------"
    );

    console.log(
      "Configured models:"
    );

    for (const model of Object.values(MODELS)) {
      console.log(
        `  ${model.id} -> ${model.upstream}`
      );
    }

    console.log(
      "=================================================="
    );
  }
);
