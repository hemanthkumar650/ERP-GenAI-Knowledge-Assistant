import { Router } from "express";

import { RagService } from "../services/ragService";
import { SearchRequest, SearchResponse } from "../types";

export default function createSearchRouter(ragService: RagService) {
  const router = Router();

  router.post("/", async (req, res, next) => {
    try {
      const body = req.body as SearchRequest;
      const query = body.query?.trim();

      if (!query) {
        return res.status(400).json({ error: "query is required" });
      }

      const data = await ragService.searchDocuments(query, body.topK ?? 3, req.requestId);
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
