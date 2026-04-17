// server.js - OpenAI to NVIDIA NIM Proxy (Plugin Architecture)

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// Config
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false; // toggle reasoning output

// Model mapping (OpenAI -> NIM)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2.5',
  'gpt-4o': 'deepseek-ai/deepseek-v3.2',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

//
// 🧠 STRATEGIES
//

const deepseekStrategy = {
  buildRequest: ({ model, messages, temperature, max_tokens, stream }) => ({
    model,
    messages,
    temperature: temperature ?? 0.6,
    max_tokens: max_tokens ?? 9024,
    stream: stream ?? false,
    chat_template_kwargs: { thinking: true } // ✅ critical fix
  }),

  parseDelta: (delta) => ({
    reasoning: delta.reasoning_content ?? null,
    content: delta.content ?? null
  }),

  formatOutput: ({ reasoning, content, state }) => {
    if (!SHOW_REASONING) return content || '';

    let output = '';

    if (reasoning && !state.reasoningStarted) {
      output += '<think>\n' + reasoning;
      state.reasoningStarted = true;
    } else if (reasoning) {
      output += reasoning;
    }

    if (content && state.reasoningStarted) {
      output += '</think>\n\n' + content;
      state.reasoningStarted = false;
    } else if (content) {
      output += content;
    }

    return output;
  }
};

const kimiStrategy = {
  buildRequest: ({ model, messages, temperature, max_tokens, stream }) => ({
    model,
    messages,
    temperature: temperature ?? 1.0,
    max_tokens: max_tokens ?? 16384,
    stream: stream ?? true
  }),

  parseDelta: (delta) => ({
    reasoning: null,
    content: delta.content ?? null
  }),

  formatOutput: ({ content }) => content || ''
};

const defaultStrategy = {
  buildRequest: ({ model, messages, temperature, max_tokens, stream }) => ({
    model,
    messages,
    temperature: temperature ?? 0.7,
    max_tokens: max_tokens ?? 4096,
    stream: stream ?? false
  }),

  parseDelta: (delta) => ({
    reasoning: null,
    content: delta.content ?? null
  }),

  formatOutput: ({ content }) => content || ''
};

//
// 🔀 STRATEGY REGISTRY
//

const MODEL_STRATEGIES = {
  'deepseek-ai/deepseek-v3.2': deepseekStrategy,
  'moonshotai/kimi-k2.5': kimiStrategy,
  'default': defaultStrategy
};

function getStrategy(model) {
  return MODEL_STRATEGIES[model] || MODEL_STRATEGIES['default'];
}

//
// 🌐 ROUTES
//

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'NIM Proxy (Plugin आधारित)',
    reasoning_display: SHOW_REASONING
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

//
// 🚀 MAIN ENDPOINT
//

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel = MODEL_MAPPING[model] || model;
    const strategy = getStrategy(nimModel);

    const nimRequest = strategy.buildRequest({
      model: nimModel,
      messages,
      temperature,
      max_tokens,
      stream
    });

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          ...(stream && { 'Accept': 'text/event-stream' })
        },
        responseType: stream ? 'stream' : 'json'
      }
    );

    //
    // 🔥 STREAMING HANDLER
    //
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      const state = { reasoningStarted: false };

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            res.write(line + '\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta || {};

            const { reasoning, content } = strategy.parseDelta(delta);

            const output = strategy.formatOutput({
              reasoning,
              content,
              state
            });

            data.choices[0].delta = { content: output };

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', () => res.end());

    } else {
      //
      // 🔹 NON-STREAM RESPONSE
      //
      res.json(response.data);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);

    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'proxy_error'
      }
    });
  }
});

//
// ❌ FALLBACK
//

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error'
    }
  });
});

//
// ▶️ START SERVER
//

app.listen(PORT, () => {
  console.log(`🚀 NIM Proxy running on port ${PORT}`);
  console.log(`🔍 Reasoning display: ${SHOW_REASONING ? 'ON' : 'OFF'}`);
});
