// ============================================================
// NVIDIA NIM -> OpenAI-Compatible Streaming Proxy
// ============================================================
//
// CURRENT MODELS
//
// kimi-k3
//   -> moonshotai/kimi-k3
//
// deepseek-v4-pro
//   -> deepseek-ai/deepseek-v4-pro-0813
//
// deepseek-v4-flash
//   -> deepseek-ai/deepseek-v4-flash-0731
//
// muse-glimmer-30b
//   -> meta/muse-glimmer-30b
//
// nemotron-3-ultra
//   -> nvidia/nemotron-3-ultra-550b-a55b
//
// gemma-4-31b
//   -> google/gemma-4-31b-it
//
// COMMUNITY / DEPLOYMENT
//
// deepseek-r1-32b-uncensored
//   -> nicoboss/DeepSeek-R1-Distill-Qwen-32B-Uncensored
//   -> https://nim.api.nvidia.com/v1
//
// llama3.3-8b-heretic
//   -> DavidAU/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning
//   -> https://nim.api.nvidia.com/v1
//
// FALLBACK
//
// gemma-4-31b
//
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

// Community / Deployment NIM endpoint.
const COMMUNITY_NIM_API_BASE =
  process.env.COMMUNITY_NIM_API_BASE ||
  "https://nim.api.nvidia.com/v1";

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

const DEFAULT_DEEPSEEK_REASONING =
  process.env.DEEPSEEK_REASONING_EFFORT ||
  "high";

const DEFAULT_MUSE_REASONING =
  process.env.MUSE_REASONING_EFFORT ||
  "high";

const DEFAULT_NEMOTRON_THINKING =
  String(
    process.env.NEMOTRON_ENABLE_THINKING ||
      "true"
  ).toLowerCase() === "true";

const DEFAULT_GEMMA_THINKING =
  String(
    process.env.GEMMA_ENABLE_THINKING ||
      "true"
  ).toLowerCase() === "true";

// ============================================================
// FALLBACK
// ============================================================

const FALLBACK_MODEL =
  "kimi-k3";

// ============================================================
// MODEL DEFINITIONS
// ============================================================

const MODELS = {

  // ==========================================================
  // KIMI K3
  // ==========================================================

  "kimi-k3": {

    id:
      "kimi-k3",

    upstream:
      "moonshotai/kimi-k3",

    owner:
      "moonshotai",

    multimodal:
      true,

    contextWindow:
      1048576,

    maxTokens:
      65536,

    temperature:
      1.0,

    reasoning: {

      type:
        "reasoning_effort",

      allowed: [
        "low",
        "high",
        "max"
      ],

      default:
        DEFAULT_KIMI_REASONING
    },

    supports: {

      top_p:
        false,

      presence_penalty:
        false,

      frequency_penalty:
        false,

      seed:
        true,

      tools:
        true,

      stream_options:
        true,

      stop:
        false
    }
  },

  // ==========================================================
  // DEEPSEEK V4 PRO 0813
  // ==========================================================

  "deepseek-v4-pro": {

    id:
      "deepseek-v4-pro",

    upstream:
      "deepseek-ai/deepseek-v4-pro-0813",

    owner:
      "deepseek-ai",

    multimodal:
      false,

    contextWindow:
      1000000,

    maxTokens:
      16384,

    temperature:
      1.0,

    top_p:
      0.95,

    reasoning: {

      type:
        "reasoning_effort",

      allowed: [
        "low",
        "high",
        "max"
      ],

      default:
        DEFAULT_DEEPSEEK_REASONING
    },

    supports: {

      top_p:
        true,

      presence_penalty:
        false,

      frequency_penalty:
        false,

      seed:
        true,

      tools:
        true,

      stream_options:
        false,

      stop:
        false
    }
  },

  // ==========================================================
  // DEEPSEEK V4 FLASH 0731
  // ==========================================================

  "deepseek-v4-flash": {

    id:
      "deepseek-v4-flash",

    upstream:
      "deepseek-ai/deepseek-v4-flash-0731",

    owner:
      "deepseek-ai",

    multimodal:
      false,

    contextWindow:
      1000000,

    maxTokens:
      16384,

    temperature:
      1.0,

    top_p:
      0.95,

    reasoning: {

      type:
        "reasoning_effort",

      allowed: [
        "low",
        "high",
        "max"
      ],

      default:
        DEFAULT_DEEPSEEK_REASONING
    },

    supports: {

      top_p:
        true,

      presence_penalty:
        false,

      frequency_penalty:
        false,

      seed:
        true,

      tools:
        true,

      stream_options:
        false,

      stop:
        false
    }
  },

  // ==========================================================
  // MUSE GLIMMER 30B
  // ==========================================================

  "muse-glimmer-30b": {

    id:
      "muse-glimmer-30b",

    upstream:
      "meta/muse-glimmer-30b",

    owner:
      "meta",

    multimodal:
      true,

    contextWindow:
      131072,

    maxTokens:
      131072,

    temperature:
      0.95,

    top_p:
      1.0,

    reasoning: {

      type:
        "reasoning_effort",

      allowed: [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "max"
      ],

      default:
        DEFAULT_MUSE_REASONING
    },

    supports: {

      top_p:
        true,

      presence_penalty:
        true,

      frequency_penalty:
        true,

      seed:
        false,

      tools:
        true,

      stream_options:
        false,

      stop:
        true
    }
  },

  // ==========================================================
  // NEMOTRON 3 ULTRA
  // ==========================================================

  "nemotron-3-ultra": {

    id:
      "nemotron-3-ultra",

    upstream:
      "nvidia/nemotron-3-ultra-550b-a55b",

    owner:
      "nvidia",

    multimodal:
      false,

    contextWindow:
      1000000,

    maxTokens:
      32768,

    temperature:
      1.0,

    top_p:
      0.95,

    reasoning: {

      type:
        "chat_template_thinking",

      default:
        DEFAULT_NEMOTRON_THINKING
    },

    supports: {

      top_p:
        true,

      presence_penalty:
        false,

      frequency_penalty:
        false,

      seed:
        false,

      tools:
        true,

      stream_options:
        false,

      stop:
        true
    }
  },

  // ==========================================================
  // GEMMA 4 31B IT
  // ==========================================================

  "gemma-4-31b": {

    id:
      "gemma-4-31b",

    upstream:
      "google/gemma-4-31b-it",

    owner:
      "google",

    multimodal:
      true,

    contextWindow:
      262144,

    maxTokens:
      16384,

    temperature:
      1.0,

    top_p:
      0.95,

    reasoning: {

      type:
        "chat_template_thinking",

      default:
        DEFAULT_GEMMA_THINKING
    },

    supports: {

      top_p:
        true,

      presence_penalty:
        false,

      frequency_penalty:
        false,

      seed:
        false,

      tools:
        true,

      stream_options:
        false,

      stop:
        false
    }
  },

  // ==========================================================
  // COMMUNITY / DEPLOYMENT
  // DEEPSEEK-R1-DISTILL-QWEN-32B-UNCENSORED
  // ==========================================================

  "deepseek-r1-32b-uncensored": {

    id:
      "deepseek-r1-32b-uncensored",

    upstream:
      "nicoboss/DeepSeek-R1-Distill-Qwen-32B-Uncensored",

    owner:
      "nicoboss",

    apiBase:
      COMMUNITY_NIM_API_BASE,

    multimodal:
      false,

    contextWindow:
      131072,

    maxTokens:
      32768,

    defaultMaxTokens:
      1024,

    temperature:
      0.5,

    top_p:
      1.0,

    reasoning: {

      type:
        "none"
    },

    supports: {

      top_p:
        true,

      presence_penalty:
        false,

      frequency_penalty:
        false,

      seed:
        false,

      tools:
        false,

      stream_options:
        false,

      stop:
        false
    }
  },

  // ==========================================================
  // COMMUNITY / DEPLOYMENT
  // DAVIDAU LLAMA 3.3 8B
  // THINKING / HERETIC / UNCENSORED
  // ==========================================================

  "llama3.3-8b-heretic": {

    id:
      "llama3.3-8b-heretic",

    upstream:
      "DavidAU/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning",

    owner:
      "DavidAU",

    apiBase:
      COMMUNITY_NIM_API_BASE,

    multimodal:
      false,

    contextWindow:
      131072,

    maxTokens:
      32768,

    defaultMaxTokens:
      1024,

    temperature:
      0.5,

    top_p:
      1.0,

    reasoning: {

      type:
        "none"
    },

    supports: {

      top_p:
        true,

      presence_penalty:
        false,

      frequency_penalty:
        false,

      seed:
        false,

      tools:
        false,

      stream_options:
        false,

      stop:
        false
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
    limit:
      "100mb"
  })
);

app.use(
  express.urlencoded({
    limit:
      "100mb",

    extended:
      true
  })
);

// ============================================================
// HELPERS
// ============================================================

function getModel(
  modelName
) {

  if (
    modelName &&
    MODELS[modelName]
  ) {

    return MODELS[
      modelName
    ];
  }

  return null;
}

// ============================================================
// ERROR MESSAGE EXTRACTION
// ============================================================

function extractErrorMessage(
  errorData
) {

  if (
    !errorData
  ) {

    return "Unknown NVIDIA API error";
  }

  if (
    typeof errorData ===
    "string"
  ) {

    return errorData ||
      "Unknown NVIDIA API error";
  }

  if (
    errorData.error
  ) {

    if (
      typeof errorData.error ===
      "string"
    ) {

      return errorData.error;
    }

    if (
      errorData.error.message
    ) {

      return String(
        errorData.error.message
      );
    }
  }

  if (
    errorData.message
  ) {

    return String(
      errorData.message
    );
  }

  try {

    return JSON.stringify(
      errorData
    );

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

  if (
    res.headersSent
  ) {

    try {

      if (
        !res.writableEnded
      ) {

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

        param:
          null,

        code:
          null
      }
    });
}

// ============================================================
// THINK TAG CLEANUP
// ============================================================

function stripThinkTags(
  text
) {

  if (
    typeof text !==
    "string"
  ) {

    return text;
  }

  return text

    .replace(
      /<think>[\s\S]*?<\/think>/gi,
      ""
    )

    .replace(
      /<\/?think>/gi,
      "");
}

// ============================================================
// MEDIA DETECTION
// ============================================================

function containsMedia(
  messages
) {

  if (
    !Array.isArray(
      messages
    )
  ) {

    return false;
  }

  for (
    const message of
    messages
  ) {

    if (
      !Array.isArray(
        message?.content
      )
    ) {

      continue;
    }

    for (
      const part of
      message.content
    ) {

      if (
        part?.type ===
          "image_url" ||

        part?.type ===
          "video_url"
      ) {

        return true;
      }
    }
  }

  return false;
}

// ============================================================
// MESSAGE VALIDATION
// ============================================================

function validateMessages(
  messages
) {

  if (
    !Array.isArray(
      messages
    )
  ) {

    return (
      "messages must be an array"
    );
  }

  if (
    messages.length ===
    0
  ) {

    return (
      "messages cannot be empty"
    );
  }

  for (
    let i = 0;
    i < messages.length;
    i++
  ) {

    const message =
      messages[i];

    if (
      !message ||
      typeof message !==
        "object"
    ) {

      return (
        `messages[${i}] must be an object`
      );
    }

    if (
      typeof message.role !==
      "string"
    ) {

      return (
        `messages[${i}].role must be a string`
      );
    }

    if (
      message.content ===
        undefined &&

      message.tool_calls ===
        undefined
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
    model.reasoning.type !==
    "reasoning_effort"
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

  // ==========================================================
  // DAVIDAU COMMUNITY MODEL
  // ==========================================================
  //
  // EXACTLY MATCHES THE USER'S WORKING OPENAI SDK REQUEST:
  //
  // model
  // messages
  // temperature: 0.5
  // top_p: 1
  // max_tokens: 1024
  // stream: true
  //
  // No reasoning_effort.
  // No chat_template_kwargs.
  // No tools.
  // No stop.
  // No stream_options.
  //
  // ==========================================================

  if (
    model.id ===
    "llama3.3-8b-heretic"
  ) {

    return {

      model:
        model.upstream,

      messages:
        body.messages,

      temperature:
        0.5,

      top_p:
        1,

      max_tokens:
        1024,

      stream:
        true
    };
  }

  // ==========================================================
  // DEEPSEEK R1 COMMUNITY MODEL
  // ==========================================================

  if (
    model.id ===
    "deepseek-r1-32b-uncensored"
  ) {

    return {

      model:
        model.upstream,

      messages:
        body.messages,

      temperature:
        body.temperature !==
        undefined
          ? body.temperature
          : 0.5,

      top_p:
        body.top_p !==
        undefined
          ? body.top_p
          : 1,

      max_tokens:
        body.max_tokens !==
        undefined
          ? body.max_tokens
          : 1024,

      stream:
        true
    };
  }

  // ==========================================================
  // STANDARD NVIDIA MODELS
  // ==========================================================

  const request = {

    model:
      model.upstream,

    messages:
      body.messages,

    stream:
      true
  };

  // ==========================================================
  // TEMPERATURE
  // ==========================================================

  if (
    body.temperature !==
    undefined
  ) {

    request.temperature =
      body.temperature;

  } else {

    request.temperature =
      model.temperature;
  }

  // ==========================================================
  // TOP P
  // ==========================================================

  if (
    model.supports.top_p
  ) {

    if (
      body.top_p !==
      undefined
    ) {

      request.top_p =
        body.top_p;

    } else if (
      model.top_p !==
      undefined
    ) {

      request.top_p =
        model.top_p;
    }
  }

  // ==========================================================
  // MAX TOKENS
  // ==========================================================

  let requestedMaxTokens =
    body.max_tokens;

  if (
    requestedMaxTokens ===
    undefined
  ) {

    requestedMaxTokens =
      body.max_completion_tokens;
  }

  if (
    requestedMaxTokens ===
    undefined
  ) {

    requestedMaxTokens =
      model.maxTokens;
  }

  if (
    model.maxTokens &&
    requestedMaxTokens >
      model.maxTokens
  ) {

    requestedMaxTokens =
      model.maxTokens;
  }

  request.max_tokens =
    requestedMaxTokens;

  // ==========================================================
  // REASONING EFFORT
  // ==========================================================

  if (
    model.reasoning.type ===
    "reasoning_effort"
  ) {

    request.reasoning_effort =
      getReasoningEffort(
        body,
        model
      );
  }

  // ==========================================================
  // CHAT TEMPLATE THINKING
  // ==========================================================

  if (
    model.reasoning.type ===
    "chat_template_thinking"
  ) {

    request.chat_template_kwargs = {

      enable_thinking:
        model.id ===
          "nemotron-3-ultra"
          ? DEFAULT_NEMOTRON_THINKING
          : DEFAULT_GEMMA_THINKING
    };
  }

  // ==========================================================
  // SEED
  // ==========================================================

  if (
    model.supports.seed &&
    body.seed !==
      undefined
  ) {

    request.seed =
      body.seed;
  }

  // ==========================================================
  // TOOLS
  // ==========================================================

  if (
    model.supports.tools &&
    Array.isArray(
      body.tools
    ) &&
    body.tools.length >
      0
  ) {

    request.tools =
      body.tools;
  }

  // ==========================================================
  // TOOL CHOICE
  // ==========================================================

  if (
    model.supports.tools &&
    body.tool_choice !==
      undefined
  ) {

    request.tool_choice =
      body.tool_choice;
  }

  // ==========================================================
  // STOP
  // ==========================================================

  if (
    model.supports.stop &&
    body.stop !==
      undefined
  ) {

    request.stop =
      body.stop;
  }

  // ==========================================================
  // PRESENCE PENALTY
  // ==========================================================

  if (
    model.supports.presence_penalty &&
    body.presence_penalty !==
      undefined
  ) {

    request.presence_penalty =
      body.presence_penalty;
  }

  // ==========================================================
  // FREQUENCY PENALTY
  // ==========================================================

  if (
    model.supports.frequency_penalty &&
    body.frequency_penalty !==
      undefined
  ) {

    request.frequency_penalty =
      body.frequency_penalty;
  }

  // ==========================================================
  // STREAM OPTIONS
  // ==========================================================

  if (
    model.supports.stream_options &&
    body.stream_options !==
      undefined
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
      Object.values(
        MODELS
      ).map(
        model => ({

          id:
            model.id,

          object:
            "model",

          created:
            0,

          owned_by:
            model.owner
        })
      );

    res.json({

      object:
        "list",

      data:
        models
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

      object:
        "proxy",

      status:
        "ok",

      streaming:
        true,

      fallback_model:
        FALLBACK_MODEL,

      models:
        Object.keys(
          MODELS
        )
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

    if (
      body.stream !==
      true
    ) {

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
      validateMessages(
        body.messages
      );

    if (
      validationError
    ) {

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
      getModel(
        requestedModel
      );

    // ========================================================
    // UNKNOWN MODEL -> FALLBACK
    // ========================================================

    if (
      !selectedModel
    ) {

      console.warn(
        `[Fallback] Unknown model "${requestedModel}". ` +
        `Using "${FALLBACK_MODEL}".`
      );

      selectedModel =
        MODELS[
          FALLBACK_MODEL
        ];
    }

    // ========================================================
    // MULTIMODAL VALIDATION
    // ========================================================

    if (
      containsMedia(
        body.messages
      ) &&
      !selectedModel.multimodal
    ) {

      return sendOpenAIError(
        res,
        400,

        `"${selectedModel.id}" does not support image/video input.`,

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
      `[Endpoint] ` +
      `${selectedModel.apiBase || NIM_API_BASE}`
    );

    console.log(
      `[Request Config] ` +
      `temperature=${primaryRequest.temperature} ` +
      `top_p=${primaryRequest.top_p ?? "default"} ` +
      `max_tokens=${primaryRequest.max_tokens} ` +
      `reasoning_effort=${primaryRequest.reasoning_effort ?? "not_sent"}`
    );

    // ========================================================
    // NVIDIA REQUEST
    // ========================================================

    let upstreamResponse;

    try {

      upstreamResponse =
        await axios.post(

          `${selectedModel.apiBase || NIM_API_BASE}/chat/completions`,

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

            responseType:
              "stream",

            timeout:
              0,

            validateStatus:
              () => true
          }
        );

    } catch (
      error
    ) {

      console.error(
        "[NVIDIA Connection Error]",
        error.message
      );

      // ======================================================
      // CONNECTION FALLBACK
      // ======================================================

      if (
        selectedModel.id !==
        FALLBACK_MODEL
      ) {

        console.warn(
          `[Fallback] ${selectedModel.id} connection failed. ` +
          `Retrying with ${FALLBACK_MODEL}.`
        );

        try {

          const fallbackModel =
            MODELS[
              FALLBACK_MODEL
            ];

          const fallbackRequest =
            buildNvidiaRequest(

              {
                ...body,

                model:
                  FALLBACK_MODEL
              },

              fallbackModel
            );

          upstreamResponse =
            await axios.post(

              `${fallbackModel.apiBase || NIM_API_BASE}/chat/completions`,

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

                responseType:
                  "stream",

                timeout:
                  0,

                validateStatus:
                  () => true
              }
            );

          selectedModel =
            fallbackModel;

        } catch (
          fallbackError
        ) {

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
      upstreamResponse.status <
        200 ||
      upstreamResponse.status >=
        300
    ) {

      let errorText =
        "";

      try {

        for await (
          const chunk of
          upstreamResponse.data
        ) {

          errorText +=
            chunk.toString(
              "utf8"
            );

          if (
            errorText.length >
            100000
          ) {

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
          JSON.parse(
            errorText
          );

      } catch {
        // Keep string.
      }

      const upstreamMessage =
        extractErrorMessage(
          parsedError
        );

      console.error(
        `[NVIDIA HTTP ${upstreamResponse.status}] ` +
        `${selectedModel.upstream}: ` +
        upstreamMessage
      );

      // ======================================================
      // HTTP FALLBACK
      // ======================================================

      if (
        selectedModel.id !==
        FALLBACK_MODEL
      ) {

        console.warn(
          `[Fallback] ${selectedModel.id} returned HTTP ` +
          `${upstreamResponse.status}. ` +
          `Retrying with ${FALLBACK_MODEL}.`
        );

        try {

          const fallbackModel =
            MODELS[
              FALLBACK_MODEL
            ];

          const fallbackRequest =
            buildNvidiaRequest(

              {
                ...body,

                model:
                  FALLBACK_MODEL
              },

              fallbackModel
            );

          const fallbackResponse =
            await axios.post(

              `${fallbackModel.apiBase || NIM_API_BASE}/chat/completions`,

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

                responseType:
                  "stream",

                timeout:
                  0,

                validateStatus:
                  () => true
              }
            );

          if (
            fallbackResponse.status <
              200 ||
            fallbackResponse.status >=
              300
          ) {

            let fallbackText =
              "";

            try {

              for await (
                const chunk of
                fallbackResponse.data
              ) {

                fallbackText +=
                  chunk.toString(
                    "utf8"
                  );

                if (
                  fallbackText.length >
                  100000
                ) {

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
                JSON.parse(
                  fallbackText
                );

            } catch {
              // Keep string.
            }

            const fallbackMessage =
              extractErrorMessage(
                fallbackParsed
              );

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
            MODELS[
              FALLBACK_MODEL
            ];

          console.log(
            `[Fallback] Now streaming from ` +
            `${selectedModel.upstream}`
          );

        } catch (
          fallbackError
        ) {

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
    // DAVIDAU COMMUNITY MODEL - RAW SSE PASSTHROUGH
    // ========================================================
    //
    // IMPORTANT:
    //
    // The supplied working OpenAI SDK example for the DavidAU
    // model receives the NVIDIA Community stream directly.
    //
    // Therefore this model bypasses the generic SSE parser.
    //
    // We do NOT:
    //
    // - parse JSON chunks
    // - reconstruct SSE
    // - strip <think> tags
    // - alter delta.content
    // - alter reasoning_content
    // - add fields
    // - remove fields
    //
    // The exact bytes received from NVIDIA are forwarded to
    // the downstream client.
    //
    // ========================================================

    if (
      selectedModel.id ===
      "llama3.3-8b-heretic"
    ) {

      console.log(
        "[DavidAU] Using RAW SSE passthrough"
      );

      console.log(
        "[DavidAU] Upstream request:",
        JSON.stringify(
          primaryRequest,
          null,
          2
        )
      );

      console.log(
        "[DavidAU] Upstream HTTP status:",
        upstreamResponse.status
      );

      console.log(
        "[DavidAU] Upstream content-type:",
        upstreamResponse.headers?.[
          "content-type"
        ]
      );

      // ======================================================
      // CLIENT SSE HEADERS
      // ======================================================

      res.status(
        200
      );

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

      if (
        typeof res.flushHeaders ===
        "function"
      ) {

        res.flushHeaders();
      }

      // ======================================================
      // RAW STREAM STATE
      // ======================================================

      let rawBytes =
        0;

      let rawChunks =
        0;

      let rawEnded =
        false;

      // ======================================================
      // RAW UPSTREAM DATA
      // ======================================================

      upstreamResponse.data.on(
        "data",
        chunk => {

          if (
            rawEnded ||
            res.writableEnded
          ) {

            return;
          }

          rawChunks++;

          rawBytes +=
            chunk.length;

          console.log(
            `[DavidAU RAW CHUNK] ${chunk.length} bytes`
          );

          console.log(
            "[DavidAU RAW DATA]",
            chunk.toString(
              "utf8"
            )
          );

          try {

            res.write(
              chunk
            );

          } catch (
            error
          ) {

            console.error(
              "[DavidAU Response Write Error]",
              error.message
            );
          }
        }
      );

      // ======================================================
      // RAW STREAM END
      // ======================================================

      upstreamResponse.data.on(
        "end",
        () => {

          if (
            rawEnded
          ) {

            return;
          }

          rawEnded =
            true;

          console.log(
            `[DavidAU STREAM END] chunks=${rawChunks} bytes=${rawBytes}`
          );

          if (
            !res.writableEnded
          ) {

            res.end();
          }
        }
      );

      // ======================================================
      // RAW STREAM ERROR
      // ======================================================

      upstreamResponse.data.on(
        "error",
        error => {

          console.error(
            "[DavidAU RAW STREAM ERROR]",
            error.message
          );

          rawEnded =
            true;

          if (
            !res.writableEnded
          ) {

            res.end();
          }
        }
      );

      // ======================================================
      // CLIENT DISCONNECT
      // ======================================================

      req.on(
        "close",
        () => {

          if (
            rawEnded
          ) {

            return;
          }

          console.log(
            "[DavidAU] Client connection closed"
          );

          rawEnded =
            true;

          if (
            upstreamResponse?.data?.destroy
          ) {

            upstreamResponse.data.destroy();
          }
        }
      );

      return;
    }

    // ========================================================
    // SSE RESPONSE HEADERS
    // ========================================================

    res.status(
      200
    );

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

    if (
      typeof res.flushHeaders ===
      "function"
    ) {

      res.flushHeaders();
    }

    // ========================================================
    // STREAM STATE
    // ========================================================

    let buffer =
      "";

    let finished =
      false;

    // ========================================================
    // WRITE SSE
    // ========================================================

    function writeSSE(
      data
    ) {

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

      } catch (
        error
      ) {

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

      if (
        finished
      ) {

        return;
      }

      finished =
        true;

      try {

        if (
          !res.writableEnded
        ) {

          res.write(
            "data: [DONE]\n\n"
          );
        }

      } catch {
        // Client may already be gone.
      }

      if (
        !res.writableEnded
      ) {

        res.end();
      }
    }

    // ========================================================
    // PROCESS SSE LINE
    // ========================================================

    function processSSELine(
      line
    ) {

      line =
        line.replace(
          /\r$/,
          ""
        );

      if (
        !line.trim()
      ) {

        return;
      }

      if (
        line.startsWith(":")
      ) {

        return;
      }

      if (
        !line.startsWith(
          "data:"
        )
      ) {

        return;
      }

      const raw =
        line
          .slice(5)
          .trim();

      if (
        raw ===
        "[DONE]"
      ) {

        finishStream();

        return;
      }

      let parsed;

      try {

        parsed =
          JSON.parse(
            raw
          );

      } catch (
        error
      ) {

        console.error(
          "[SSE JSON Parse Error]",
          error.message,

          raw.substring(
            0,
            500
          )
        );

        return;
      }

      // ======================================================
      // NORMALIZE RESPONSE
      // ======================================================

      if (
        Array.isArray(
          parsed.choices
        )
      ) {

        for (
          const choice of
          parsed.choices
        ) {

          if (
            choice?.delta
          ) {

            // Do not modify DavidAU here because that model
            // already uses the raw passthrough above.

            if (
              typeof choice.delta.content ===
                "string" &&

              selectedModel.id !==
                "llama3.3-8b-heretic"
            ) {

              choice.delta.content =
                stripThinkTags(
                  choice.delta.content
                );
            }

            if (
              !SHOW_REASONING
            ) {

              delete choice.delta.reasoning;

              delete choice.delta.reasoning_content;
            }
          }
        }
      }

      // ======================================================
      // SEND TO CLIENT
      // ======================================================

      writeSSE(
        parsed
      );
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
          chunk.toString(
            "utf8"
          );

        const lines =
          buffer.split(
            "\n"
          );

        buffer =
          lines.pop() ||
          "";

        for (
          const line of
          lines
        ) {

          if (
            finished
          ) {

            break;
          }

          processSSELine(
            line
          );
        }
      }
    );

    // ========================================================
    // UPSTREAM END
    // ========================================================

    upstreamResponse.data.on(
      "end",
      () => {

        if (
          buffer.trim()
        ) {

          processSSELine(
            buffer
          );
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

            type:
              "stream_error"
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

    // ========================================================
    // CLIENT DISCONNECT
    // ========================================================

    req.on(
      "close",
      () => {

        if (
          finished
        ) {

          return;
        }

        finished =
          true;

        if (
          upstreamResponse?.data?.destroy
        ) {

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

    if (
      res.headersSent
    ) {

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
      `Community NVIDIA API: ${COMMUNITY_NIM_API_BASE}`
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

    for (
      const model of
      Object.values(
        MODELS
      )
    ) {

      console.log(
        `  ${model.id} -> ${model.upstream}`
      );
    }

    console.log(
      "=================================================="
    );
  }
);
