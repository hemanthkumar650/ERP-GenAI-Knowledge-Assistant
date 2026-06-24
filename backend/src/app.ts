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
      res.json(data);
    } catch (error) {
      next(error);
    }
  });

  app.get("/api/chunks", async (req, res, next) => {
    try {
      const limit = normalizeChunkLimit(req.query.limit);
      const data = await service.getChunks(limit, req.requestId);
      res.json(data);
    } catch (error) {
      next(error);
    }
  });

  app.post("/api/reindex", async (req, res, next) => {
    try {
      const data = await service.reindexDocuments(req.requestId);
      res.json(data);
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
    const message =
      error instanceof Error
        ? error.message
        : "Unexpected backend error while calling the Python RAG service.";
    res.status(errorStatus(error)).json({ error: message, requestId: req.requestId });
  });

  return app;
}
