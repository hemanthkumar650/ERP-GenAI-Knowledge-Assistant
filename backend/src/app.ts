import cors from "cors";
import express from "express";
import helmet from "helmet";

import "./types/expressRequestContext";

import { environment } from "./config/environment";
import { createApiRateLimiter } from "./middleware/rateLimiter";
import { assignRequestId } from "./middleware/requestId";
import { createRequestLogger } from "./middleware/requestLogger";
import createChatRouter from "./routes/chat";
import createSearchRouter from "./routes/search";
import { ragService, RagService } from "./services/ragService";

function normalizeChunkLimit(rawLimit: unknown, fallback = 12, max = 50): number {
  if (rawLimit === null || rawLimit === undefined) {
    return fallback;
  }

  if (Array.isArray(rawLimit)) {
    return fallback;
  }

  if (typeof rawLimit === "string" && rawLimit.trim() === "") {
    return fallback;
  }

  const parsed = Number(rawLimit);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }

  return Math.min(Math.max(Math.trunc(parsed), 1), max);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }

  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function normalizeChunks(rawChunks: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(rawChunks)) {
    return [];
  }

  return rawChunks.filter((chunk): chunk is Record<string, unknown> => isPlainObject(chunk));
}

function normalizeStatus(rawStatus: unknown, fallback = "ok"): string {
  if (typeof rawStatus !== "string") {
    return fallback;
  }

  const normalized = rawStatus.trim();
  return normalized || fallback;
}

function normalizeIndexedChunks(rawCount: unknown, fallback = 0): number {
  if (typeof rawCount !== "number" || !Number.isFinite(rawCount)) {
    return fallback;
  }

  return Math.max(Math.trunc(rawCount), 0);
}

function normalizeBoolean(rawValue: unknown, fallback = false): boolean {
  return typeof rawValue === "boolean" ? rawValue : fallback;
}

function normalizeErrorMessage(
  rawMessage: unknown,
  fallback = "Unexpected backend error while calling the Python RAG service.",
): string {
  if (typeof rawMessage !== "string") {
    return fallback;
  }

  const normalized = rawMessage.trim();
  return normalized || fallback;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return normalizeErrorMessage(error.message);
  }

  if (typeof error === "object" && error !== null && "message" in error) {
    return normalizeErrorMessage(error.message);
  }

  return normalizeErrorMessage(undefined);
}

function errorStatus(error: unknown): number {
  if (typeof error !== "object" || error === null) {
    return 500;
  }

  const candidate =
    "status" in error ? error.status : "statusCode" in error ? error.statusCode : undefined;
  return typeof candidate === "number" && candidate >= 400 && candidate < 600 ? candidate : 500;
}

export function createApp(service: RagService = ragService) {
  const app = express();

  app.use(assignRequestId());
  app.use(createRequestLogger());
  app.use(helmet());
  app.use(cors());
  app.use(express.json({ limit: "1mb" }));

  app.get("/health", (_req, res) => {
    res.json({
      status: "ok",
      service: "backend",
      pythonRagUrl: environment.pythonRagUrl,
      timestamp: new Date().toISOString(),
    });
  });

  app.use("/api", createApiRateLimiter());

  app.get("/api/health", async (req, res, next) => {
    try {
      const data = await service.getHealth(req.requestId);
      const payload = isPlainObject(data) ? data : {};
      res.json({
        status: normalizeStatus(payload.status),
        vector_db_loaded: normalizeBoolean(payload.vector_db_loaded),
        indexed_chunks: normalizeIndexedChunks(payload.indexed_chunks),
      });
    } catch (error) {
      next(error);
    }
  });

  app.get("/api/chunks", async (req, res, next) => {
    try {
      const limit = normalizeChunkLimit(req.query.limit);
      const data = await service.getChunks(limit, req.requestId);
      const chunks = isPlainObject(data) ? normalizeChunks(data.chunks) : [];
      res.json({ chunks });
    } catch (error) {
      next(error);
    }
  });

  app.post("/api/reindex", async (req, res, next) => {
    try {
      const data = await service.reindexDocuments(req.requestId);
      const payload = isPlainObject(data) ? data : {};
      res.json({
        status: normalizeStatus(payload.status),
        indexed_chunks: normalizeIndexedChunks(payload.indexed_chunks),
      });
    } catch (error) {
      next(error);
    }
  });

  app.use("/api/chat", createChatRouter(service));
  app.use("/api/search", createSearchRouter(service));

  app.use((req, res, next) => {
    if (res.headersSent) {
      return next();
    }

    return res.status(404).json({
      error: "Route not found",
      requestId: req.requestId,
    });
  });

  app.use((error: unknown, req: express.Request, res: express.Response, _next: express.NextFunction) => {
    const message = errorMessage(error);
    res.status(errorStatus(error)).json({ error: message, requestId: req.requestId });
  });

  return app;
}
