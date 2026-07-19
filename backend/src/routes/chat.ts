import { Router } from "express";

import { RagService } from "../services/ragService";
import { ChatRequest, ChatResponse } from "../types";

function normalizeText(rawText: unknown): string | undefined {
  if (typeof rawText !== "string") {
    return undefined;
  }

  const normalized = rawText.trim();
  return normalized || undefined;
}

function normalizeConversationId(rawConversationId: unknown): string | undefined {
  if (typeof rawConversationId !== "string") {
    return undefined;
  }

  const normalized = rawConversationId.trim();
  return normalized || undefined;
}

function normalizeSources(rawSources: unknown): string[] {
  if (!Array.isArray(rawSources)) {
    return [];
  }

  return rawSources
    .map((source) => (typeof source === "string" ? source.trim() : ""))
    .filter((source): source is string => source.length > 0);
}

function normalizeChunks(rawChunks: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(rawChunks)) {
    return [];
  }

  return rawChunks.filter(
    (chunk): chunk is Record<string, unknown> =>
      typeof chunk === "object" && chunk !== null && !Array.isArray(chunk),
  );
}

function normalizeAnswer(rawAnswer: unknown): string {
  return typeof rawAnswer === "string" ? rawAnswer : "";
}

export default function createChatRouter(ragService: RagService) {
  const router = Router();

  router.post("/", async (req, res, next) => {
    try {
      const body = (req.body ?? {}) as Partial<ChatRequest>;
      const message = normalizeText(body.message);
      const conversationId = normalizeConversationId(body.conversationId);

      if (!message) {
        return res.status(400).json({ error: "message is required" });
      }

      const data = await ragService.askQuestion(message, conversationId, req.requestId);
      const response: ChatResponse = {
        response: normalizeAnswer(data.answer),
        sources: normalizeSources(data.sources),
        chunks: normalizeChunks(data.chunks),
        conversationId: normalizeConversationId(data.session_id) ?? conversationId,
        timestamp: new Date().toISOString(),
      };

      return res.json(response);
    } catch (error) {
      return next(error);
    }
  });

  return router;
}
