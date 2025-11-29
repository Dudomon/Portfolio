/**
 * Middleware para rotas não encontradas
 */
const notFound = (req, res, next) => {
  res.status(404).json({
    error: 'Rota não encontrada',
    message: `A rota ${req.method} ${req.originalUrl} não existe`,
    availableRoutes: {
      'POST /order': 'Criar novo pedido',
      'GET /order/:numeroPedido': 'Obter pedido específico',
      'GET /order/list': 'Listar todos os pedidos',
      'PUT /order/:numeroPedido': 'Atualizar pedido',
      'DELETE /order/:numeroPedido': 'Deletar pedido'
    }
  });
};

module.exports = notFound;
