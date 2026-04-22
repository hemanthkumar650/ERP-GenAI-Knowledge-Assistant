import type { Request, RequestHandler } from "express";
import morgan from "morgan";

import { environment } from "../config/environment";

const DEV_WITH_REQ_ID =
  ":method :url :status :response-time ms - :res[content-length] :req-id";
const COMBINED_WITH_REQ_ID =
  ':remote-addr - :remote-user [:date[clf]] ":method :url HTTP/:http-version" :status :res[content-length] ":referrer" ":user-agent" :req-id';

morgan.token("req-id", (req: Request) => req.requestId ?? "-");

function resolveMorganFormat(): string {
  if (process.env.HTTP_LOG_FORMAT) {
    return environment.logging.format;
  }
  return environment.nodeEnv === "production" ? COMBINED_WITH_REQ_ID : DEV_WITH_REQ_ID;
}

export function createRequestLogger(): RequestHandler {
  if (environment.logging.disabled) {
    return (_req, _res, next) => next();
  }

  return morgan(resolveMorganFormat(), {
    skip: (req) => req.path === "/health",
  });
}
