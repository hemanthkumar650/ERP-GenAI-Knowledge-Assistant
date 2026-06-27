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
  if (rawTopK === null || rawTopK === undefined) {
    return fallback;
  }

  if (Array.isArray(rawTopK)) {
    return fallback;
  }

  if (typeof rawTopK === "string" && rawTopK.trim() === "") {
    return fallback;
  }

  const parsed = Number(rawTopK);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }

  return Math.min(Math.max(Math.trunc(parsed), 1), max);
}

function normalizeResultCount(rawCount: unknown, fallback: number): number {
  if (typeof rawCount !== "number" || !Number.isFinite(rawCount)) {
    return fallback;
  }

  return Math.max(Math.trunc(rawCount), 0);
}

function normalizeResults(rawResults: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(rawResults)) {
    return [];
  }

  return rawResults.filter(
    (result): result is Record<string, unknown> =>
      typeof result === "object" && result !== null && !Array.isArray(result),
  );
}

export default function createSearchRouter(ragService: RagService) {
  const router = Router();

  router.post("/", async (req, res, next) => {
    try {
      const body = (req.body ?? {}) as Partial<SearchRequest>;
      const query = normalizeText(body.query);

      if (!query) {
        return res.status(400).json({ error: "query is required" });
      }

      const topK = normalizeTopK(body.topK);
      const data = await ragService.searchDocuments(query, topK, req.requestId);
      const results = normalizeResults(data.results);
      const response: SearchResponse = {
        results,
        count: normalizeResultCount(data.count, results.length),
      };

      return res.json(response);
    } catch (error) {
      return next(error);
    }
  });

  return router;
}
