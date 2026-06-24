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
        response: data.answer,
        sources: data.sources ?? [],
        chunks: data.chunks ?? [],
        conversationId: data.session_id ?? conversationId,
        timestamp: new Date().toISOString(),
      };

      return res.json(response);
    } catch (error) {
      return next(error);
    }
  });

  return router;
}
