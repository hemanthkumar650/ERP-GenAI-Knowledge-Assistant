import type { RequestHandler } from "express";
import rateLimit from "express-rate-limit";

import { environment } from "../config/environment";

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
    // Cheap status probe; the UI polls this often. LLM routes stay limited.
    skip: (req) => {
      const path = req.originalUrl.split("?")[0] ?? "";
      return req.method === "GET" && path === "/api/health";
    },
  });
}
