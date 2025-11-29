/**
 * Middleware para log de requisições
 */
const requestLogger = (req, res, next) => {
  const start = Date.now();

  // Log quando a resposta é finalizada
  res.on('finish', () => {
    const duration = Date.now() - start;
    const timestamp = new Date().toISOString();

    console.log(
      `[${timestamp}] ${req.method} ${req.originalUrl} - Status: ${res.statusCode} - ${duration}ms`
    );
  });

  next();
};

module.exports = requestLogger;
