// ============================================================
// NVIDIA NIM -> OpenAI-Compatible Streaming Proxy
// ============================================================

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

// ============================================================
// SERVER
// ============================================================

const PORT =
  process.env.PORT || 3000;

// ============================================================
// NVIDIA ENDPOINTS
// ============================================================

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1";

const COMMUNITY_NIM_API_BASE =
  process.env.COMMUNITY_NIM_API_BASE ||
  "https://nim.api.nvidia.com/v1";

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
  "gemma-4-31b";

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
  // DEEPSEEK V4 PRO
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
  // DEEPSEEK V4 FLASH
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
  // COMMUNITY DEPLOYMENT
  //
  // nicoboss/DeepSeek-R1-Distill-Qwen-32B-Uncensored
  //
  // IMPORTANT:
  //
  // This model has a deliberately isolated request path.
  //
  // Verified working request:
  //
  // baseURL:
  //   https://nim.api.nvidia.com/v1
  //
  // model:
  //   nicoboss/DeepSeek-R1-Distill-Qwen-32B-Uncensored
  //
  // temperature:
  //   0.5
  //
  // top_p:
  //   1
  //
  // max_tokens:
  //   1024
  //
  // stream:
  //   true
  //
  // No reasoning_effort.
  // No chat_template_kwargs.
  // No tools.
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

    temperature:
      0.5,

    top_p:
      1.0,

    community:
      true,

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
// MIDDLEWARE
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
    return MODELS[modelName];
  }

  return null;
}

// ============================================================
// MEDIA DETECTION
// ============================================================

function containsMedia(
  messages
) {

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
// REASONING EFFORT
// ============================================================

function getReasoningEffort(
  body,
  model
) {

  if (
    model.reasoning?.type !==
      "reasoning_effort"
  ) {
    return null;
  }

  const requested =
    body.reasoning_effort ??
    model.reasoning.default;

  if (
    model.reasoning.allowed.includes(
      requested
    )
  ) {
    return requested;
  }

  return model.reasoning.default;
}

// ============================================================
// BUILD STANDARD NVIDIA REQUEST
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
    model.supports?.top_p
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

  let maxTokens =
    body.max_tokens;

  if (
    maxTokens ===
      undefined ||
    maxTokens ===
      null
  ) {

    maxTokens =
      16384;
  }

  maxTokens =
    Number(
      maxTokens
    );

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

    maxTokens =
      1;
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

  // ==========================================================
  // SEED
  // ==========================================================

  if (
    model.supports?.seed &&
    body.seed !==
      undefined
  ) {

    request.seed =
      body.seed;
  }

  // ==========================================================
  // STOP
  // ==========================================================

  if (
    model.supports?.stop &&
    body.stop !==
      undefined
  ) {

    request.stop =
      body.stop;
  }

  // ==========================================================
  // TOOLS
  // ==========================================================

  if (
    model.supports?.tools &&
    body.tools !==
      undefined
  ) {

    request.tools =
      body.tools;
  }

  if (
    model.supports?.tools &&
    body.tool_choice !==
      undefined
  ) {

    request.tool_choice =
      body.tool_choice;
  }

  // ==========================================================
  // STREAM OPTIONS
  // ==========================================================

  if (
    model.supports?.stream_options &&
    body.stream_options !==
      undefined
  ) {

    request.stream_options =
      body.stream_options;
  }

  // ==========================================================
  // PRESENCE PENALTY
  // ==========================================================

  if (
    model.supports?.presence_penalty &&
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
    model.supports?.frequency_penalty &&
    body.frequency_penalty !==
      undefined
  ) {

    request.frequency_penalty =
      body.frequency_penalty;
  }

  // ==========================================================
  // KIMI
  // ==========================================================

  if (
    model.id ===
      "kimi-k3"
  ) {

    request.reasoning_effort =
      getReasoningEffort(
        body,
        model
      );
  }

  // ==========================================================
  // DEEPSEEK V4
  // ==========================================================

  else if (
    model.id ===
      "deepseek-v4-pro" ||
    model.id ===
      "deepseek-v4-flash"
  ) {

    request.reasoning_effort =
      getReasoningEffort(
        body,
        model
      );

    if (
      body.chat_template_kwargs &&
      typeof body.chat_template_kwargs ===
        "object"
    ) {

      request.chat_template_kwargs =
        body.chat_template_kwargs;
    }
  }

  // ==========================================================
  // MUSE
  // ==========================================================

  else if (
    model.id ===
      "muse-glimmer-30b"
  ) {

    request.reasoning_effort =
      getReasoningEffort(
        body,
        model
      );

    if (
      body.chat_template_kwargs &&
      typeof body.chat_template_kwargs ===
        "object"
    ) {

      request.chat_template_kwargs =
        body.chat_template_kwargs;
    }
  }

  // ==========================================================
  // NEMOTRON
  // ==========================================================

  else if (
    model.id ===
      "nemotron-3-ultra"
  ) {

    request.chat_template_kwargs = {

      ...(body.chat_template_kwargs ||
        {}),

      enable_thinking:
        body
          .chat_template_kwargs
          ?.enable_thinking ??
        DEFAULT_NEMOTRON_THINKING
    };
  }

  // ==========================================================
  // GEMMA
  // ==========================================================

  else if (
    model.id ===
      "gemma-4-31b"
  ) {

    request.chat_template_kwargs = {

      ...(body.chat_template_kwargs ||
        {}),

      enable_thinking:
        body
          .chat_template_kwargs
          ?.enable_thinking ??
        DEFAULT_GEMMA_THINKING
    };
  }

  return request;
}

// ============================================================
// BUILD COMMUNITY REQUEST
//
// THIS IS INTENTIONALLY SEPARATE.
//
// It reproduces the user's verified working example.
//
// The client's temperature/max_tokens can override the
// defaults, because the proxy should preserve client settings.
// ============================================================

function buildCommunityRequest(
  body
) {

  return {

    model:
      "nicoboss/DeepSeek-R1-Distill-Qwen-32B-Uncensored",

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

// ============================================================
// ERROR EXTRACTION
// ============================================================

function extractErrorMessage(
  data
) {

  if (
    typeof data ===
      "string"
  ) {

    return data;
  }

  if (
    data?.error?.message
  ) {

    return data.error.message;
  }

  if (
    typeof data?.error ===
      "string"
  ) {

    return data.error;
  }

  if (
    data?.message
  ) {

    return data.message;
  }

  try {

    return JSON.stringify(
      data
    );

  } catch {

    return "NVIDIA API request failed";
  }
}

// ============================================================
// OPENAI ERROR
// ============================================================

function sendOpenAIError(
  res,
  status,
  message,
  code
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

        code:
          code ||
          "nvidia_api_error"
      }
    });
}

// ============================================================
// COMMON NVIDIA AXIOS CONFIG
// ============================================================

function getAxiosConfig() {

  return {

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
  };
}

// ============================================================
// READ UPSTREAM ERROR STREAM
// ============================================================

async function readErrorStream(
  stream
) {

  let text =
    "";

  try {

    for await (
      const chunk of stream
    ) {

      text +=
        chunk.toString(
          "utf8"
        );

      if (
        text.length >=
        100000
      ) {

        break;
      }
    }

  } catch {
    // Ignore secondary stream errors.
  }

  return text;
}

// ============================================================
// WRITE STREAM HEADERS
// ============================================================

function prepareSSE(
  res
) {

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

  // Critical for streaming through Node/proxies.
  if (
    typeof res.flushHeaders ===
      "function"
  ) {

    res.flushHeaders();
  }
}

// ============================================================
// COMMUNITY STREAM
//
// IMPORTANT:
// Do NOT parse/reconstruct this stream.
//
// The supplied working OpenAI SDK example proves that NVIDIA
// returns a valid OpenAI-compatible stream. Passing the SSE
// bytes through directly eliminates another possible source
// of "No response from bot" errors.
// ============================================================

async function handleCommunityStream(
  req,
  res,
  body
) {

  const request =
    buildCommunityRequest(
      body
    );

  const endpoint =
    `${COMMUNITY_NIM_API_BASE}/chat/completions`;

  console.log(
    `[Community Request] ${request.model} [STREAMING]`
  );

  console.log(
    `[Community Endpoint] ${endpoint}`
  );

  console.log(
    `[Community Config] ` +
    `temperature=${request.temperature} ` +
    `top_p=${request.top_p} ` +
    `max_tokens=${request.max_tokens} ` +
    `stream=true`
  );

  console.log(
    "[Community Request Body]",
    JSON.stringify(
      request
    )
  );

  let upstream;

  try {

    upstream =
      await axios.post(
        endpoint,
        request,
        getAxiosConfig()
      );

  } catch (
    error
  ) {

    console.error(
      "[Community Connection Error]",
      error.code ||
      error.message
    );

    return sendOpenAIError(

      res,

      502,

      error.message ||
        "Community NVIDIA endpoint connection failed",

      "community_connection_error"
    );
  }

  // ==========================================================
  // UPSTREAM HTTP ERROR
  // ==========================================================

  if (
    upstream.status <
      200 ||
    upstream.status >=
      300
  ) {

    const errorText =
      await readErrorStream(
        upstream.data
      );

    let parsed =
      errorText;

    try {

      parsed =
        JSON.parse(
          errorText
        );

    } catch {
      // Raw text.
    }

    const message =
      extractErrorMessage(
        parsed
      );

    console.error(
      `[Community HTTP ${upstream.status}] ${message}`
    );

    return sendOpenAIError(

      res,

      upstream.status,

      message,

      "community_api_error"
    );
  }

  // ==========================================================
  // START CLIENT STREAM IMMEDIATELY
  // ==========================================================

  prepareSSE(
    res
  );

  let clientClosed =
    false;

  const onClientClose =
    () => {

      clientClosed =
        true;

      if (
        upstream?.data &&
        !upstream.data.destroyed
      ) {

        upstream.data.destroy();
      }
    };

  req.once(
    "close",
    onClientClose
  );

  // ==========================================================
  // DIRECT STREAM
  // ==========================================================

  upstream.data.on(
    "error",
    error => {

      if (
        clientClosed ||
        res.writableEnded
      ) {

        return;
      }

      console.error(
        "[Community Stream Error]",
        error.code ||
        error.message
      );

      try {

        res.write(
          `data: ${JSON.stringify({

            error: {

              message:
                error.message ||
                "Community NVIDIA streaming error",

              type:
                "stream_error"
            }

          })}\n\n`
        );

      } catch {
        // Client disconnected.
      }

      if (
        !res.writableEnded
      ) {

        res.end();
      }
    }
  );

  upstream.data.on(
    "end",
    () => {

      if (
        clientClosed
      ) {

        return;
      }

      if (
        !res.writableEnded
      ) {

        try {

          res.write(
            "data: [DONE]\n\n"
          );

        } catch {
          // Client disconnected.
        }

        if (
          !res.writableEnded
        ) {

          res.end();
        }
      }
    }
  );

  upstream.data.on(
    "data",
    chunk => {

      if (
        clientClosed ||
        res.writableEnded
      ) {

        return;
      }

      try {

        res.write(
          chunk
        );

      } catch (
        error
      ) {

        console.error(
          "[Community Client Stream Error]",
          error.message
        );

        clientClosed =
          true;

        if (
          !upstream.data.destroyed
        ) {

          upstream.data.destroy();
        }
      }
    }
  );
}

// ============================================================
// STANDARD NIM STREAM
// ============================================================

async function handleStandardStream(
  req,
  res,
  body,
  model
) {

  const request =
    buildNvidiaRequest(
      body,
      model
    );

  const endpoint =
    `${NIM_API_BASE}/chat/completions`;

  console.log(
    `[Request] ${model.id} -> ${model.upstream} [STREAMING]`
  );

  console.log(
    `[Endpoint] ${endpoint}`
  );

  console.log(
    `[Request Config] ` +
    `temperature=${request.temperature} ` +
    `top_p=${request.top_p ?? "default"} ` +
    `max_tokens=${request.max_tokens} ` +
    `reasoning_effort=${request.reasoning_effort ?? "not-sent"}`
  );

  let upstream;

  try {

    upstream =
      await axios.post(
        endpoint,
        request,
        getAxiosConfig()
      );

  } catch (
    error
  ) {

    console.error(
      "[NVIDIA Connection Error]",
      error.code ||
      error.message
    );

    throw error;
  }

  // ==========================================================
  // HTTP ERROR
  // ==========================================================

  if (
    upstream.status <
      200 ||
    upstream.status >=
      300
  ) {

    const errorText =
      await readErrorStream(
        upstream.data
      );

    let parsed =
      errorText;

    try {

      parsed =
        JSON.parse(
          errorText
        );

    } catch {
      // Keep text.
    }

    const message =
      extractErrorMessage(
        parsed
      );

    const error =
      new Error(
        message
      );

    error.status =
      upstream.status;

    throw error;
  }

  prepareSSE(
    res
  );

  let buffer =
    "";

  let finished =
    false;

  let clientClosed =
    false;

  function finish() {

    if (
      finished
    ) {

      return;
    }

    finished =
      true;

    if (
      !res.writableEnded
    ) {

      try {

        res.write(
          "data: [DONE]\n\n"
        );

      } catch {
        // Client disconnected.
      }

      if (
        !res.writableEnded
      ) {

        res.end();
      }
    }
  }

  function sendChunk(
    data
  ) {

    if (
      finished ||
      res.writableEnded ||
      clientClosed
    ) {

      return;
    }

    try {

      if (
        Array.isArray(
          data?.choices
        )
      ) {

        for (
          const choice of
          data.choices
        ) {

          if (
            !choice?.delta
          ) {

            continue;
          }

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

      clientClosed =
        true;

      if (
        !upstream.data.destroyed
      ) {

        upstream.data.destroy();
      }
    }
  }

  function processLine(
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

      finish();

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
        "[SSE Parse Error]",
        error.message
      );

      return;
    }

    sendChunk(
      parsed
    );
  }

  req.once(
    "close",
    () => {

      if (
        finished
      ) {

        return;
      }

      clientClosed =
        true;

      if (
        upstream?.data &&
        !upstream.data.destroyed
      ) {

        upstream.data.destroy();
      }
    }
  );

  upstream.data.on(
    "data",
    chunk => {

      if (
        finished ||
        clientClosed
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
          finished ||
          clientClosed
        ) {

          break;
        }

        processLine(
          line
        );
      }
    }
  );

  upstream.data.on(
    "end",
    () => {

      if (
        finished ||
        clientClosed
      ) {

        return;
      }

      if (
        buffer.trim()
      ) {

        processLine(
          buffer
        );
      }

      finish();
    }
  );

  upstream.data.on(
    "error",
    error => {

      if (
        finished ||
        clientClosed ||
        res.writableEnded
      ) {

        return;
      }

      console.error(
        "[NVIDIA Stream Error]",
        error.code ||
        error.message
      );

      try {

        res.write(
          `data: ${JSON.stringify({

            error: {

              message:
                error.message ||
                "NVIDIA streaming error",

              type:
                "stream_error"
            }

          })}\n\n`
        );

      } catch {
        // Client disconnected.
      }

      if (
        !res.writableEnded
      ) {

        res.end();
      }
    }
  );
}

// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post(
  "/v1/chat/completions",
  async (
    req,
    res
  ) => {

    const body =
      req.body ||
      {};

    // ========================================================
    // STREAMING ONLY
    // ========================================================

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
    // MODEL
    // ========================================================

    const requestedModel =
      body.model;

    let model =
      getModel(
        requestedModel
      );

    // ========================================================
    // UNKNOWN MODEL
    // ========================================================

    if (
      !model
    ) {

      console.warn(
        `[Fallback] Unknown model "${requestedModel}". ` +
        `Using "${FALLBACK_MODEL}".`
      );

      model =
        MODELS[
          FALLBACK_MODEL
        ];
    }

    // ========================================================
    // MEDIA VALIDATION
    // ========================================================

    if (
      containsMedia(
        body.messages
      ) &&
      !model.multimodal
    ) {

      return sendOpenAIError(

        res,

        400,

        `"${model.id}" does not support image/video input.`,

        "multimodal_not_supported"
      );
    }

    // ========================================================
    // COMMUNITY MODEL
    //
    // This path is intentionally isolated from the normal
    // NIM request builder.
    // ========================================================

    if (
      model.community
    ) {

      try {

        await handleCommunityStream(
          req,
          res,
          body
        );

        return;

      } catch (
        error
      ) {

        console.error(
          "[Community Model Error]",
          error.code ||
          error.message
        );

        // ----------------------------------------------------
        // FALLBACK
        // ----------------------------------------------------

        if (
          FALLBACK_MODEL &&
          FALLBACK_MODEL !==
            model.id
        ) {

          console.warn(
            `[Fallback] Community model failed. ` +
            `Falling back to ${FALLBACK_MODEL}.`
          );

          try {

            const fallbackModel =
              MODELS[
                FALLBACK_MODEL
              ];

            await handleStandardStream(
              req,
              res,
              {
                ...body,

                model:
                  FALLBACK_MODEL
              },
              fallbackModel
            );

            return;

          } catch (
            fallbackError
          ) {

            console.error(
              "[Fallback Error]",
              fallbackError.code ||
              fallbackError.message
            );

            if (
              !res.headersSent
            ) {

              return sendOpenAIError(

                res,

                502,

                fallbackError.message ||
                  "Fallback model failed.",

                "fallback_error"
              );
            }

            return;
          }
        }

        if (
          !res.headersSent
        ) {

          return sendOpenAIError(

            res,

            502,

            error.message ||
              "Community model request failed.",

            "community_model_error"
          );
        }

        return;
      }
    }

    // ========================================================
    // STANDARD MODEL
    // ========================================================

    try {

      await handleStandardStream(
        req,
        res,
        body,
        model
      );

    } catch (
      error
    ) {

      console.error(
        "[Standard NVIDIA Error]",
        error.code ||
        error.message
      );

      // ======================================================
      // FALLBACK
      // ======================================================

      if (
        model.id !==
        FALLBACK_MODEL
      ) {

        console.warn(
          `[Fallback] ${model.id} failed. ` +
          `Falling back to ${FALLBACK_MODEL}.`
        );

        try {

          const fallbackModel =
            MODELS[
              FALLBACK_MODEL
            ];

          await handleStandardStream(

            req,

            res,

            {
              ...body,

              model:
                FALLBACK_MODEL
            },

            fallbackModel
          );

          return;

        } catch (
          fallbackError
        ) {

          console.error(
            "[Fallback Error]",
            fallbackError.code ||
            fallbackError.message
          );

          if (
            !res.headersSent
          ) {

            return sendOpenAIError(

              res,

              502,

              fallbackError.message ||
                "Fallback model failed.",

              "fallback_error"
            );
          }

          return;
        }
      }

      if (
        !res.headersSent
      ) {

        return sendOpenAIError(

          res,

          error.status ||
            502,

          error.message ||
            "NVIDIA API request failed.",

          "nvidia_api_error"
        );
      }
    }
  }
);

// ============================================================
// HEALTH
// ============================================================

app.get(
  "/health",
  (
    req,
    res
  ) => {

    res.json({

      status:
        "ok",

      streaming:
        true,

      api_base:
        NIM_API_BASE,

      community_api_base:
        COMMUNITY_NIM_API_BASE,

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

            community:
              Boolean(
                model.community
              ),

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
// /v1/models
// ============================================================

app.get(
  "/v1/models",
  (
    req,
    res
  ) => {

    const created =
      Math.floor(
        Date.now() /
          1000
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
      " NVIDIA NIM OpenAI-Compatible Streaming Proxy"
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
      `Community API: ${COMMUNITY_NIM_API_BASE}`
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
        `  ${model.id} -> ${model.upstream}` +
        `${
          model.community
            ? " [COMMUNITY]"
            : ""
        }`
      );
    }

    console.log(
      "=================================================="
    );
  }
);
