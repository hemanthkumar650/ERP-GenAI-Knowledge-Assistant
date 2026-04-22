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

  app.get("/api/health", async (_req, res, next) => {
    try {
      const data = await service.getHealth();
      res.json(data);
    } catch (error) {
      next(error);
    }
  });

  app.get("/api/chunks", async (req, res, next) => {
    try {
      const limit = Number(req.query.limit ?? 12);
      const data = await service.getChunks(limit);
      res.json(data);
    } catch (error) {
      next(error);
    }
  });

  app.post("/api/reindex", async (_req, res, next) => {
    try {
      const data = await service.reindexDocuments();
      res.json(data);
    } catch (error) {
      next(error);
    }
  });

  app.use("/api/chat", createChatRouter(service));
  app.use("/api/search", createSearchRouter(service));

  app.use((error: unknown, req: express.Request, res: express.Response, _next: express.NextFunction) => {
    const message =
      error instanceof Error
        ? error.message
        : "Unexpected backend error while calling the Python RAG service.";
    res.status(500).json({ error: message, requestId: req.requestId });
  });

  return app;
}
