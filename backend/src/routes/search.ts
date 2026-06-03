import { Router } from "express";

import { RagService } from "../services/ragService";
import { SearchRequest, SearchResponse } from "../types";

function normalizeText(rawText: unknown): string | undefined {
  if (typeof rawText !== "string") {
    return undefined;
  }

  const normalized = rawText.trim();
  return normalized || undefined;
}

function normalizeTopK(rawTopK: unknown, fallback = 3, max = 10): number {
  const parsed = Number(rawTopK);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }

  return Math.min(Math.max(Math.trunc(parsed), 1), max);
}

export default function createSearchRouter(ragService: RagService) {
  const router = Router();

  router.post("/", async (req, res, next) => {
    try {
      const body = req.body as SearchRequest;
      const query = normalizeText(body.query);

      if (!query) {
        return res.status(400).json({ error: "query is required" });
      }

      const topK = normalizeTopK(body.topK);
      const data = await ragService.searchDocuments(query, topK, req.requestId);
      const response: SearchResponse = {
        results: data.results ?? [],
        count: data.count ?? 0,
      };

      return res.json(response);
    } catch (error) {
      return next(error);
    }
  });

  return router;
}
