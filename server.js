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
// FALLBACK
//
// deepseek-v4-flash
//
// ============================================================

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = process.env.PORT || 3000;

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY =
  process.env.NIM_API_KEY;

// ============================================================
// CONFIGURATION
// ============================================================

// Set to true to expose reasoning_content to the client.
//
// IMPORTANT:
// Kimi K3 requires reasoning_content to be preserved and sent
// back on later turns. If your client strips it before sending
// the next request, Kimi K3 multi-turn reasoning can be affected.
//
// If your frontend already handles reasoning_content correctly,
// leave this true.
//
// If you only want final visible answers, set false.
const SHOW_REASONING =
  String(
    process.env.SHOW_REASONING || "true"
  ).toLowerCase() === "true";

// ------------------------------------------------------------
// DEFAULT REASONING
// ------------------------------------------------------------

const DEFAULT_KIMI_REASONING =
  process.env.KIMI_REASONING_EFFORT || "high";

const DEFAULT_DEEPSEEK_REASONING =
  process.env.DEEPSEEK_REASONING_EFFORT || "high";

const DEFAULT_MUSE_REASONING =
  process.env.MUSE_REASONING_EFFORT || "high";

const DEFAULT_NEMOTRON_THINKING =
  String(
    process.env.NEMOTRON_ENABLE_THINKING || "true"
  ).toLowerCase() === "true";

// ============================================================
// FALLBACK
// ============================================================
//
// This is the model used when:
//
// 1. The client requests an unknown public model name.
//
// 2. NVIDIA rejects the selected upstream model before the
//    streaming response begins.
//
// 3. The upstream connection fails before any response has
//    been sent to the client.
//
// We do NOT attempt to change models after a stream has begun.
//

const FALLBACK_MODEL =
  "deepseek-v4-flash";

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

function getModel(modelName) {

  if (
    modelName &&
    MODELS[modelName]
  ) {
    return MODELS[modelName];
  }

  return null;
}

// ------------------------------------------------------------
// THINK TAG CLEANUP
// ------------------------------------------------------------

function stripThinkTags(text) {

  if (
    typeof text !== "string"
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

// ------------------------------------------------------------
// MEDIA DETECTION
// ------------------------------------------------------------

function containsMedia(messages) {

  if (
    !Array.isArray(messages)
  ) {
    return false;
  }

  for (
    const message of messages
  ) {

    if (
      !Array.isArray(
        message?.content
      )
    ) {
      continue;
    }

    for (
      const part of message.content
    ) {

      if (
        part?.type === "image_url" ||
        part?.type === "video_url"
      ) {
        return true;
      }
    }
  }

  return false;
}

// ------------------------------------------------------------
// MESSAGE VALIDATION
// ------------------------------------------------------------

function validateMessages(messages) {

  if (
    !Array.isArray(messages)
  ) {
    return "messages must be an array";
  }

  if (
    messages.length === 0
  ) {
    return "messages cannot be empty";
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
      typeof message !== "object"
    ) {
      return (
        `messages[${i}] must be an object`
      );
    }

    if (
      typeof message.role !== "string"
    ) {
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

// ------------------------------------------------------------
// REASONING VALIDATION
// ------------------------------------------------------------

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

  const request = {

    model:
      model.upstream,

    messages:
      body.messages,

    stream:
      true
  };

  // ----------------------------------------------------------
  // TEMPERATURE
  // ----------------------------------------------------------

  if (
    body.temperature !== undefined
  ) {

    request.temperature =
      body.temperature;

  } else {

    request.temperature =
      model.temperature;
  }

  // ----------------------------------------------------------
  // TOP P
  // ----------------------------------------------------------

  if (
    model.supports.top_p
  ) {

    if (
      body.top_p !== undefined
    ) {

      request.top_p =
        body.top_p;

    } else if (
      model.top_p !== undefined
    ) {

      request.top_p =
        model.top_p;
    }
  }

  // ----------------------------------------------------------
  // MAX TOKENS
  // ----------------------------------------------------------

  let maxTokens =
    body.max_tokens;

  if (
    maxTokens === undefined ||
    maxTokens === null
  ) {

    maxTokens =
      16384;
  }

  maxTokens =
    Number(maxTokens);

  if (
    !Number.isFinite(
      maxTokens
    )
  ) {

    maxTokens =
      16384;
  }

  maxTokens =
    Math.floor(
      maxTokens
    );

  if (
    maxTokens < 1
  ) {
    maxTokens = 1;
  }

  if (
    maxTokens >
    model.maxTokens
  ) {

    maxTokens =
      model.maxTokens;
  }

  request.max_tokens =
    maxTokens;

  // ----------------------------------------------------------
  // SEED
  // ----------------------------------------------------------

  if (
    model.supports.seed &&
    body.seed !== undefined
  ) {

    request.seed =
      body.seed;
  }

  // ----------------------------------------------------------
  // STOP
  // ----------------------------------------------------------

  if (
    model.supports.stop &&
    body.stop !== undefined
  ) {

    request.stop =
      body.stop;
  }

  // ----------------------------------------------------------
  // TOOLS
  // ----------------------------------------------------------

  if (
    model.supports.tools &&
    body.tools !== undefined
  ) {

    request.tools =
      body.tools;
  }

  if (
    model.supports.tools &&
    body.tool_choice !== undefined
  ) {

    request.tool_choice =
      body.tool_choice;
  }

  // ----------------------------------------------------------
  // STREAM OPTIONS
  // ----------------------------------------------------------

  if (
    model.supports.stream_options &&
    body.stream_options !== undefined
  ) {

    request.stream_options =
      body.stream_options;
  }

  // ----------------------------------------------------------
  // PRESENCE PENALTY
  // ----------------------------------------------------------

  if (
    model.supports.presence_penalty &&
    body.presence_penalty !== undefined
  ) {

    request.presence_penalty =
      body.presence_penalty;
  }

  // ----------------------------------------------------------
  // FREQUENCY PENALTY
  // ----------------------------------------------------------

  if (
    model.supports.frequency_penalty &&
    body.frequency_penalty !== undefined
  ) {

    request.frequency_penalty =
      body.frequency_penalty;
  }

  // ==========================================================
  // MODEL-SPECIFIC REASONING
  // ==========================================================

  // ----------------------------------------------------------
  // KIMI K3
  // ----------------------------------------------------------

  if (
    model.id === "kimi-k3"
  ) {

    request.reasoning_effort =
      getReasoningEffort(
        body,
        model
      );
  }

  // ----------------------------------------------------------
  // DEEPSEEK V4 PRO / FLASH
  // ----------------------------------------------------------

  else if (
    model.id === "deepseek-v4-pro" ||
    model.id === "deepseek-v4-flash"
  ) {

    request.reasoning_effort =
      getReasoningEffort(
        body,
        model
      );

    //
    // NVIDIA's current DeepSeek Build example also exposes
    // thinking through chat_template_kwargs.
    //
    // If the client explicitly supplies chat_template_kwargs,
    // preserve them.
    //
    if (
      body.chat_template_kwargs &&
      typeof body.chat_template_kwargs ===
        "object"
    ) {

      request.chat_template_kwargs =
        body.chat_template_kwargs;
    }
  }

  // ----------------------------------------------------------
  // MUSE GLIMMER
  // ----------------------------------------------------------

  else if (
    model.id === "muse-glimmer-30b"
  ) {

    request.reasoning_effort =
      getReasoningEffort(
        body,
        model
      );

    //
    // Preserve explicit client template settings.
    //
    if (
      body.chat_template_kwargs &&
      typeof body.chat_template_kwargs ===
        "object"
    ) {

      request.chat_template_kwargs =
        body.chat_template_kwargs;
    }
  }

  // ----------------------------------------------------------
  // NEMOTRON 3 ULTRA
  // ----------------------------------------------------------
  //
  // NVIDIA documents thinking through:
  //
  // chat_template_kwargs:
  // {
  //   enable_thinking: true
  // }
  //
  // Do NOT send reasoning_effort to Nemotron.
  //

  else if (
    model.id === "nemotron-3-ultra"
  ) {

    request.chat_template_kwargs = {

      ...(body.chat_template_kwargs || {}),

      enable_thinking:
        body.chat_template_kwargs
          ?.enable_thinking ??
        DEFAULT_NEMOTRON_THINKING
    };
  }

  return request;
}

// ============================================================
// REASONING NORMALIZATION
// ============================================================

function normalizeChunk(
  data
) {

  if (
    !data ||
    typeof data !== "object"
  ) {
    return data;
  }

  if (
    Array.isArray(
      data.choices
    )
  ) {

    for (
      const choice of data.choices
    ) {

      const delta =
        choice?.delta;

      if (
        !delta
      ) {
        continue;
      }

      // ------------------------------------------------------
      // CLEAN VISIBLE CONTENT
      // ------------------------------------------------------

      if (
        typeof delta.content ===
        "string"
      ) {

        delta.content =
          stripThinkTags(
            delta.content
          );
      }

      // ------------------------------------------------------
      // HIDE REASONING IF REQUESTED
      // ------------------------------------------------------

      if (
        !SHOW_REASONING
      ) {

        delete delta.reasoning;

        delete delta.reasoning_content;
      }
    }
  }

  return data;
}

// ============================================================
// SAFE ERROR MESSAGE
// ============================================================

function extractErrorMessage(
  responseData
) {

  if (
    typeof responseData ===
    "string"
  ) {
    return responseData;
  }

  if (
    responseData?.error?.message
  ) {

    return responseData.error.message;
  }

  if (
    typeof responseData?.error ===
    "string"
  ) {

    return responseData.error;
  }

  if (
    responseData?.message
  ) {

    return responseData.message;
  }

  try {

    return JSON.stringify(
      responseData
    );

  } catch {

    return "NVIDIA API request failed";
  }
}

// ============================================================
// WRITE ERROR
// ============================================================

function sendOpenAIError(
  res,
  status,
  message,
  code = "nvidia_api_error"
) {

  if (
    res.headersSent
  ) {
    return;
  }

  res
    .status(status)
    .json({
      error: {
        message,
        type:
          "invalid_request_error",
        code
      }
    });
}

// ============================================================
// HEALTH
// ============================================================

app.get(
  "/health",
  (req, res) => {

    res.json({

      status:
        "ok",

      streaming:
        true,

      api_base:
        NIM_API_BASE,

      fallback_model:
        FALLBACK_MODEL,

      reasoning_display:
        SHOW_REASONING,

      models:
        Object.values(
          MODELS
        ).map(
          model => ({
            id:
              model.id,

            upstream:
              model.upstream,

            multimodal:
              model.multimodal,

            context_window:
              model.contextWindow,

            max_tokens:
              model.maxTokens
          })
        )
    });
  }
);

// ============================================================
// OPENAI /v1/models
// ============================================================

app.get(
  "/v1/models",
  (req, res) => {

    const created =
      Math.floor(
        Date.now() / 1000
      );

    res.json({

      object:
        "list",

      data:
        Object.values(
          MODELS
        ).map(
          model => ({

            id:
              model.id,

            object:
              "model",

            created,

            owned_by:
              model.owner
          })
        )
    });
  }
);

// ============================================================
// STREAMING CHAT COMPLETIONS
// ============================================================

app.post(
  "/v1/chat/completions",
  async (req, res) => {

    const body =
      req.body || {};

    // --------------------------------------------------------
    // STREAMING ONLY
    // --------------------------------------------------------

    if (
      body.stream !== true
    ) {

      return sendOpenAIError(
        res,
        400,
        "This proxy supports streaming chat completions only. Set stream=true.",
        "stream_required"
      );
    }

    // --------------------------------------------------------
    // VALIDATE MESSAGES
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // REQUESTED MODEL
    // --------------------------------------------------------

    const requestedModel =
      body.model;

    let selectedModel =
      getModel(
        requestedModel
      );

    // --------------------------------------------------------
    // UNKNOWN MODEL -> FALLBACK
    // --------------------------------------------------------

    if (
      !selectedModel
    ) {

      console.warn(
        `[Fallback] Unknown model "${requestedModel}". ` +
        `Using "${FALLBACK_MODEL}".`
      );

      selectedModel =
        MODELS[FALLBACK_MODEL];
    }

    // --------------------------------------------------------
    // MEDIA VALIDATION
    // --------------------------------------------------------

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

    // --------------------------------------------------------
    // NVIDIA REQUEST
    // --------------------------------------------------------

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
      `[Request Config] ` +
      `temperature=${primaryRequest.temperature} ` +
      `top_p=${primaryRequest.top_p ?? "default"} ` +
      `max_tokens=${primaryRequest.max_tokens} ` +
      `reasoning_effort=${primaryRequest.reasoning_effort ?? "template"}`
    );

    // --------------------------------------------------------
    // SEND UPSTREAM REQUEST
    // --------------------------------------------------------

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

            responseType:
              "stream",

            timeout:
              0,

            validateStatus:
              () => true
          }
        );

    } catch (error) {

      console.error(
        "[NVIDIA Connection Error]",
        error.message
      );

      // ------------------------------------------------------
      // FALLBACK BEFORE STREAMING
      // ------------------------------------------------------

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
            MODELS[FALLBACK_MODEL];

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
    // UPSTREAM HTTP ERROR BEFORE STREAM
    // ========================================================

    if (
      upstreamResponse.status < 200 ||
      upstreamResponse.status >= 300
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

      // ------------------------------------------------------
      // FALLBACK
      // ------------------------------------------------------

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
            MODELS[FALLBACK_MODEL];

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

                responseType:
                  "stream",

                timeout:
                  0,

                validateStatus:
                  () => true
              }
            );

          if (
            fallbackResponse.status < 200 ||
            fallbackResponse.status >= 300
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

          // Use fallback stream.
          upstreamResponse =
            fallbackResponse;

          selectedModel =
            MODELS[FALLBACK_MODEL];

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

    // ========================================================
    // STREAM STATE
    // ========================================================

    let buffer =
      "";

    let finished =
      false;

    // ========================================================
    // SSE WRITE
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

      } catch (error) {

        console.error(
          "[SSE Write Error]",
          error.message
        );
      }
    }

    // ========================================================
    // DONE
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

      // SSE heartbeat/comment.
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
        raw === "[DONE]"
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

      } catch (error) {

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

      // ------------------------------------------------------
      // NORMALIZE CHOICES
      // ------------------------------------------------------

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

            if (
              typeof choice.delta.content ===
              "string"
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

      // ------------------------------------------------------
      // SEND TO CLIENT
      // ------------------------------------------------------

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
          lines.pop() || "";

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
  (req, res) => {

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
      `API key configured: ${
        NIM_API_KEY
          ? "YES"
          : "NO"
      }`
    );

    console.log(
      `Streaming: REQUIRED`
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
