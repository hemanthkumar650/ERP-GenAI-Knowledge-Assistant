import axios from "axios";

import { environment } from "../config/environment";

const ragClient = axios.create({
  baseURL: environment.pythonRagUrl,
  timeout: 30000,
});

function outgoingRequestOptions(requestId?: string): { headers?: Record<string, string> } {
  if (!requestId) {
    return {};
  }
  return { headers: { "X-Request-Id": requestId } };
}

export interface RagService {
  getHealth(requestId?: string): Promise<unknown>;
  getChunks(limit?: number, requestId?: string): Promise<unknown>;
  reindexDocuments(requestId?: string): Promise<unknown>;
  askQuestion(message: string, conversationId?: string, requestId?: string): Promise<any>;
  searchDocuments(query: string, topK?: number, requestId?: string): Promise<any>;
}

export async function getHealth(requestId?: string) {
  const { data } = await ragClient.get("/health", outgoingRequestOptions(requestId));
  return data;
}

export async function getChunks(limit = 12, requestId?: string) {
  const { data } = await ragClient.get("/chunks", {
    params: { limit },
    ...outgoingRequestOptions(requestId),
  });
  return data;
}

export async function reindexDocuments(requestId?: string) {
  const { data } = await ragClient.post("/reindex", undefined, outgoingRequestOptions(requestId));
  return data;
}

export async function askQuestion(message: string, conversationId?: string, requestId?: string) {
  const { data } = await ragClient.post(
    "/ask",
    {
      question: message,
      session_id: conversationId,
    },
    outgoingRequestOptions(requestId),
  );
  return data;
}

export async function searchDocuments(query: string, topK = 3, requestId?: string) {
  const { data } = await ragClient.post(
    "/search",
    {
      query,
      topK,
    },
    outgoingRequestOptions(requestId),
  );
  return data;
}

export const ragService: RagService = {
  getHealth,
  getChunks,
  reindexDocuments,
  askQuestion,
  searchDocuments,
};
