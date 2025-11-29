require('dotenv').config();
const express = require('express');
const cors = require('cors');
const connectDB = require('./src/config/database');
const orderRoutes = require('./src/routes/orderRoutes');
const errorHandler = require('./src/middleware/errorHandler');
const notFound = require('./src/middleware/notFound');
const requestLogger = require('./src/middleware/requestLogger');

const app = express();

// Conectar ao banco de dados
connectDB();

// Middlewares globais
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(requestLogger);

// Rota de health check
app.get('/', (req, res) => {
  res.status(200).json({
    message: 'API de Gerenciamento de Pedidos - Jitterbit',
    version: '1.0.0',
    status: 'online',
    endpoints: {
      'POST /order': 'Criar novo pedido',
      'GET /order/:numeroPedido': 'Obter pedido específico',
      'GET /order/list': 'Listar todos os pedidos',
      'PUT /order/:numeroPedido': 'Atualizar pedido',
      'DELETE /order/:numeroPedido': 'Deletar pedido'
    }
  });
});

// Rotas da API
app.use('/order', orderRoutes);

// Middlewares de erro
app.use(notFound);
app.use(errorHandler);

// Iniciar servidor
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`\nServidor rodando na porta ${PORT}`);
  console.log(`URL: http://localhost:${PORT}`);
  console.log(`Documentação: http://localhost:${PORT}\n`);
});

module.exports = app;
