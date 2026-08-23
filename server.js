// server.js
// OpenAI-compatible proxy for a single NVIDIA NIM model:
//   stepfun-ai/step-3.7-flash  (exposed to clients as "step-3.7-flash")
//
// Source of truth for request shape:
//   https://build.nvidia.com/stepfun-ai/step-3.7-flash
//   https://docs.api.nvidia.com/nim/reference/stepfun-ai-step-3-7-flash
//
// Notes baked in from that page:
// - Step-3.7-Flash is a vision-language model (text + image input, text-only
//   output). Messages can use OpenAI-style content arrays with
//   {type: "image_url", image_url: {url: ...}} parts — those are passed
//   through untouched.
// - NVIDIA's published example sends NO chat_template_kwargs and NO
//   reasoning_effort for this model. This proxy never sends either.
// - The model can occasionally leak a stray "</think>" even though no
//   reasoning mode is enabled. We strip that defensively from all output.
// - Supports both streaming and non-streaming, since real OpenAI clients
//   expect both (unlike the multi-model reference this was adapted from,
//   which was streaming-only).

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));

// ============================================
// CONFIG
// ============================================

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const PUBLIC_MODEL_NAME = 'step-3.7-flash';
const NIM_MODEL_NAME = 'stepfun-ai/step-3.7-flash';

// Defaults straight from NVIDIA's published example for this model.
const DEFAULTS = {
  temperature: 1,
  top_p: 0.95,
  max_tokens: 16384
};

if (!NIM_API_KEY) {
  console.warn('[WARN] NIM_API_KEY is not set. Requests to NVIDIA will fail with 401.');
}

// ============================================
// HELPERS
// ============================================

// Remove any stray <think>/</think> tags. Step-3.7-Flash has no reasoning
// mode, so nothing should ever legitimately be inside them.
function stripThinkTags(text) {
  if (typeof text !== 'string' || !text) return text;
  return text.replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/<\/?think>/gi, '');
}

function getSafeErrorMessage(error) {
  if (error?.response) {
    const data = error.response.data;
    if (typeof data === 'string') return data;
    if (data && typeof data === 'object') {
      if (typeof data.error === 'string') return data.error;
      if (data.error?.message) return data.error.message;
      if (data.message) return data.message;
      try {
        return JSON.stringify(data);
      } catch {
        return `NVIDIA API returned HTTP ${error.response.status}`;
      }
    }
    return `NVIDIA API returned HTTP ${error.response.status}`;
  }
  return error?.message || 'Unknown proxy error';
}

function sendError(res, status, message, type = 'invalid_request_error') {
  return res.status(status).json({
    error: { message, type, code: status }
  });
}

// ============================================
// HEALTH
// ============================================

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    model: PUBLIC_MODEL_NAME,
    upstream_model: NIM_MODEL_NAME,
    reasoning: false
  });
});

// ============================================
// MODELS
// ============================================

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: [
      {
        id: PUBLIC_MODEL_NAME,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'stepfun-ai'
      }
    ]
  });
});

// ============================================
// CHAT COMPLETIONS
// ============================================

app.post('/v1/chat/completions', async (req, res) => {
  const {
    model,
    messages,
    temperature,
    max_tokens,
    top_p,
    seed,
    stop,
    presence_penalty,
    frequency_penalty,
    stream
  } = req.body || {};

  if (!Array.isArray(messages)) {
    return sendError(res, 400, 'messages must be an array');
  }

  if (model && model !== PUBLIC_MODEL_NAME) {
    return sendError(
      res,
      400,
      `This proxy only serves "${PUBLIC_MODEL_NAME}". Requested model: "${model}".`
    );
  }

  const isStream = stream === true;

  // Build the upstream request. Deliberately never adds
  // chat_template_kwargs or reasoning_effort for this model.
  const nimRequest = {
    model: NIM_MODEL_NAME,
    messages,
    temperature: temperature ?? DEFAULTS.temperature,
    max_tokens: max_tokens ?? DEFAULTS.max_tokens,
    top_p: top_p ?? DEFAULTS.top_p,
    stream: isStream
  };

  if (seed !== undefined && seed !== null) nimRequest.seed = seed;
  if (stop !== undefined && stop !== null) nimRequest.stop = stop;
  if (presence_penalty !== undefined && presence_penalty !== null) {
    nimRequest.presence_penalty = presence_penalty;
  }
  if (frequency_penalty !== undefined && frequency_penalty !== null) {
    nimRequest.frequency_penalty = frequency_penalty;
  }

  console.log(`[Request] ${PUBLIC_MODEL_NAME} -> ${NIM_MODEL_NAME} (stream=${isStream})`);

  // ============================================
  // NON-STREAMING
  // ============================================

  if (!isStream) {
    try {
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          Accept: 'application/json'
        },
        validateStatus: () => true
      });

      if (response.status < 200 || response.status >= 300) {
        const data = response.data;
        console.error(`[NVIDIA Error] HTTP ${response.status}:`, data);
        return sendError(
          res,
          response.status,
          data?.error?.message || data?.message || `NVIDIA API returned HTTP ${response.status}`,
          'nvidia_api_error'
        );
      }

      const body = response.data;
      for (const choice of body?.choices || []) {
        if (typeof choice?.message?.content === 'string') {
          choice.message.content = stripThinkTags(choice.message.content);
        }
        // This model has no reasoning mode; never forward reasoning fields.
        if (choice?.message) {
          delete choice.message.reasoning;
          delete choice.message.reasoning_content;
        }
      }

      return res.status(200).json(body);
    } catch (error) {
      console.error('[Proxy Error]', getSafeErrorMessage(error));
      const status = error?.response?.status || 500;
      return sendError(res, status, getSafeErrorMessage(error), 'invalid_request_error');
    }
  }

  // ============================================
  // STREAMING
  // ============================================

  let upstream;
  try {
    upstream = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream'
      },
      responseType: 'stream',
      validateStatus: () => true
    });
  } catch (error) {
    console.error('[Proxy Error]', getSafeErrorMessage(error));
    const status = error?.response?.status || 500;
    return sendError(res, status, getSafeErrorMessage(error), 'invalid_request_error');
  }

  if (upstream.status < 200 || upstream.status >= 300) {
    let errorBody = '';
    try {
      for await (const chunk of upstream.data) {
        errorBody += chunk.toString();
        if (errorBody.length > 100000) break;
      }
    } catch {
      // ignore
    }

    let parsed = errorBody;
    try {
      parsed = JSON.parse(errorBody);
    } catch {
      // keep string
    }

    console.error(`[NVIDIA Error] HTTP ${upstream.status}:`, parsed);
    return sendError(
      res,
      upstream.status,
      typeof parsed === 'string'
        ? parsed
        : parsed?.error?.message || parsed?.message || `NVIDIA API returned HTTP ${upstream.status}`,
      'nvidia_api_error'
    );
  }

  res.status(200);
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');

  let buffer = '';
  let finished = false;

  function writeSSE(data) {
    if (finished || res.writableEnded) return;
    try {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    } catch (err) {
      console.error('[SSE Write Error]', err.message);
    }
  }

  function sendDone() {
    if (finished) return;
    finished = true;
    try {
      res.write('data: [DONE]\n\n');
    } catch {
      // client likely gone
    }
    if (!res.writableEnded) res.end();
  }

  function processLine(line) {
    line = line.replace(/\r$/, '');
    if (!line.trim() || line.startsWith(':')) return;
    if (!line.startsWith('data:')) return;

    const raw = line.slice(5).trim();
    if (raw === '[DONE]') {
      sendDone();
      return;
    }

    let data;
    try {
      data = JSON.parse(raw);
    } catch (err) {
      console.error('[SSE Parse Error]', err.message);
      return;
    }

    const delta = data?.choices?.[0]?.delta;
    if (delta) {
      if (typeof delta.content === 'string') {
        delta.content = stripThinkTags(delta.content);
      }
      // No reasoning mode for this model — never forward these.
      delete delta.reasoning;
      delete delta.reasoning_content;
    }

    writeSSE(data);
  }

  upstream.data.on('data', (chunk) => {
    if (finished || res.writableEnded) return;
    buffer += chunk.toString('utf8');
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (finished) break;
      processLine(line);
    }
  });

  upstream.data.on('end', () => {
    if (buffer.trim()) processLine(buffer);
    sendDone();
  });

  upstream.data.on('error', (error) => {
    console.error('[NVIDIA Stream Error]', error.message);
    if (!finished && !res.writableEnded) {
      writeSSE({ error: { message: error.message || 'NVIDIA stream error', type: 'stream_error' } });
      if (!res.writableEnded) res.end();
      finished = true;
    }
  });

  req.on('close', () => {
    if (!finished && upstream?.data?.destroy) upstream.data.destroy();
    finished = true;
  });
});

// ============================================
// 404
// ============================================

app.all('*', (req, res) => {
  if (res.headersSent) return res.end();
  sendError(res, 404, `Endpoint ${req.path} not found`);
});

// ============================================
// START
// ============================================

app.listen(PORT, () => {
  console.log('==============================================');
  console.log('NVIDIA NIM proxy — step-3.7-flash only');
  console.log(`Port: ${PORT}`);
  console.log(`API base: ${NIM_API_BASE}`);
  console.log(`Public model name: ${PUBLIC_MODEL_NAME} -> ${NIM_MODEL_NAME}`);
  console.log(`API key set: ${NIM_API_KEY ? 'yes' : 'NO (requests will fail)'}`);
  console.log('==============================================');
});
