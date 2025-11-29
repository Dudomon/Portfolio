/**
 * Middleware para tratamento centralizado de erros
 */
const errorHandler = (err, req, res, next) => {
  console.error('Error:', err);

  // Erro de validação do Mongoose
  if (err.name === 'ValidationError') {
    return res.status(400).json({
      error: 'Erro de validação',
      message: err.message,
      details: Object.values(err.errors).map(e => e.message)
    });
  }

  // Erro de cast do Mongoose (ID inválido)
  if (err.name === 'CastError') {
    return res.status(400).json({
      error: 'Dado inválido',
      message: `Formato inválido para ${err.path}: ${err.value}`
    });
  }

  // Erro de chave duplicada
  if (err.code === 11000) {
    const field = Object.keys(err.keyPattern)[0];
    return res.status(409).json({
      error: 'Conflito',
      message: `O valor para ${field} já existe`
    });
  }

  // Erro padrão
  res.status(err.status || 500).json({
    error: err.name || 'Erro interno do servidor',
    message: err.message || 'Ocorreu um erro inesperado'
  });
};

module.exports = errorHandler;
