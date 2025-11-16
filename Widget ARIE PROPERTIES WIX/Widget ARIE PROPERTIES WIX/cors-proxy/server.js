// server.js - Proxy CORS simples para Windows
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const app = express();
const PORT = process.env.PORT || 3000;

// Configurar CORS
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Middleware para logging
app.use((req, res, next) => {
  console.log(`[${new Date().toLocaleTimeString()}] ${req.method} ${req.url}`);
  next();
});

// Rota principal do proxy
app.get('/proxy', async (req, res) => {
  try {
    const targetUrl = req.query.url;
    
    if (!targetUrl) {
      return res.status(400).json({ error: 'URL parameter is required' });
    }
    
    console.log(`Proxying request to: ${targetUrl}`);
    
    // Copiar headers da requisição original
    const headers = {};
    if (req.headers.authorization) {
      headers.Authorization = req.headers.authorization;
    }
    if (req.headers['content-type']) {
      headers['Content-Type'] = req.headers['content-type'];
    }
    
    // Fazer requisição para o alvo
    const response = await axios({
      method: req.method,
      url: targetUrl,
      headers: headers,
      data: req.method !== 'GET' ? req.body : undefined,
      responseType: 'arraybuffer'
    });
    
    // Copiar headers da resposta
    Object.entries(response.headers).forEach(([key, value]) => {
      res.setHeader(key, value);
    });
    
    // Enviar resposta
    res.status(response.status).send(response.data);
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(500).json({
      error: 'Proxy error',
      message: error.message
    });
  }
});

// Rota de teste
app.get('/', (req, res) => {
  res.send(`
    <html>
      <head>
        <title>CORS Proxy</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
          h1 { color: #0066cc; }
          .code { background: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace; }
          .success { color: #28a745; }
        </style>
      </head>
      <body>
        <h1>Proxy CORS Local</h1>
        <p class="success">✅ Servidor funcionando na porta ${PORT}!</p>
        <p>Use a seguinte URL no widget:</p>
        <p class="code">http://localhost:${PORT}/proxy?url=URL_ENCODED_TARGET</p>
        <p>Exemplo:</p>
        <p class="code">http://localhost:${PORT}/proxy?url=https%3A%2F%2Fapi.sienge.com.br%2Farieproperties%2Fpublic%2Fapi%2Fv1%2Fcurrent-debit-balance%3Fcpf%3D37455407866</p>
      </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log(`
========================================================
  🚀 Proxy CORS rodando em http://localhost:${PORT}
  
  Use esta URL no widget:
  http://localhost:${PORT}/proxy?url=
  
  Pressione Ctrl+C para encerrar o servidor
========================================================
  `);
});