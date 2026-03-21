import { Router } from "express";

import { askQuestion } from "../services/ragService";
import { ChatRequest, ChatResponse } from "../types";

const router = Router();

router.post("/", async (req, res, next) => {
  try {
    const body = req.body as ChatRequest;
    const message = body.message?.trim();

    if (!message) {
      return res.status(400).json({ error: "message is required" });
    }

    const data = await askQuestion(message, body.conversationId);
    const response: ChatResponse = {
      response: data.answer,
      sources: data.sources ?? [],
      chunks: data.chunks ?? [],
      conversationId: data.session_id ?? body.conversationId,
      timestamp: new Date().toISOString(),
    };

    return res.json(response);
  } catch (error) {
    return next(error);
  }
});

export default router;
