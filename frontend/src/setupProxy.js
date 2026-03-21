const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function setupProxy(app) {
  const target = process.env.BACKEND_PROXY_TARGET || "http://localhost:5000";

  app.use(
    "/api",
    createProxyMiddleware({
      target,
      changeOrigin: true,
    }),
  );
};
