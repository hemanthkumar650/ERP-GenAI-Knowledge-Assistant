import cors from "cors";
import express from "express";
import helmet from "helmet";

import { environment } from "./config/environment";
import chatRouter from "./routes/chat";
import searchRouter from "./routes/search";
import { getChunks, getHealth, reindexDocuments } from "./services/ragService";

const app = express();

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

app.get("/api/health", async (_req, res, next) => {
  try {
    const data = await getHealth();
    res.json(data);
  } catch (error) {
    next(error);
  }
});

app.get("/api/chunks", async (req, res, next) => {
  try {
    const limit = Number(req.query.limit ?? 12);
    const data = await getChunks(limit);
    res.json(data);
  } catch (error) {
    next(error);
  }
});

app.post("/api/reindex", async (_req, res, next) => {
  try {
    const data = await reindexDocuments();
    res.json(data);
  } catch (error) {
    next(error);
  }
});

app.use("/api/chat", chatRouter);
app.use("/api/search", searchRouter);

app.use((error: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  const message =
    error instanceof Error
      ? error.message
      : "Unexpected backend error while calling the Python RAG service.";
  res.status(500).json({ error: message });
});

app.listen(environment.port, () => {
  console.log(`Backend API listening on http://localhost:${environment.port}`);
});
