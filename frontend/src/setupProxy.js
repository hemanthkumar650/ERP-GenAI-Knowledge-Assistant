const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:5000',
      changeOrigin: true,
      // CRA mount at '/api' strips that prefix before proxying.
      // Backend expects '/api/*', so prepend it back.
      pathRewrite: (path) => `/api${path}`,
    })
  );
};