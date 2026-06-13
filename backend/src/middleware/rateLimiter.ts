import type { RequestHandler } from "express";
import rateLimit, { type RateLimitExceededEventHandler } from "express-rate-limit";

import { environment } from "../config/environment";

const rateLimitJsonHandler: RateLimitExceededEventHandler = (req, res, _next, optionsUsed) => {
  const msg = optionsUsed.message;
  const errorText =
    typeof msg === "object" &&
    msg !== null &&
    "error" in msg &&
    typeof (msg as { error: unknown }).error === "string"
      ? (msg as { error: string }).error
      : "Too many requests. Please slow down and try again in a few moments.";
  res.status(optionsUsed.statusCode).json({
    error: errorText,
    requestId: req.requestId,
  });
};

function normalizedPath(originalUrl: string): string {
  const path = originalUrl.split("?")[0] || "/";
  return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

export function createApiRateLimiter(): RequestHandler {
  if (environment.rateLimit.disabled) {
    return (_req, _res, next) => next();
  }

  return rateLimit({
    windowMs: environment.rateLimit.windowMs,
    max: environment.rateLimit.max,
    standardHeaders: "draft-7",
    legacyHeaders: false,
    message: {
      error:
        "Too many requests. Please slow down and try again in a few moments.",
    },
    handler: rateLimitJsonHandler,
    // Cheap status probe; the UI polls this often. Also skip CORS preflight traffic.
    skip: (req) => {
      const path = normalizedPath(req.originalUrl);
      const isHealthProbe = (req.method === "GET" || req.method === "HEAD") && path === "/api/health";
      return req.method === "OPTIONS" || isHealthProbe;
    },
  });
}
