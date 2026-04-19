import type { RequestHandler } from "express";
import morgan from "morgan";

import { environment } from "../config/environment";

export function createRequestLogger(): RequestHandler {
  if (environment.logging.disabled) {
    return (_req, _res, next) => next();
  }

  return morgan(environment.logging.format, {
    skip: (req) => req.path === "/health",
  });
}
