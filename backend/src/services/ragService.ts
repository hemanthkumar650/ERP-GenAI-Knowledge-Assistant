import axios from "axios";

import { environment } from "../config/environment";

const ragClient = axios.create({
  baseURL: environment.pythonRagUrl,
  timeout: 30000,
});

export interface RagService {
  getHealth(): Promise<unknown>;
  getChunks(limit?: number): Promise<unknown>;
  reindexDocuments(): Promise<unknown>;
  askQuestion(message: string, conversationId?: string): Promise<any>;
  searchDocuments(query: string, topK?: number): Promise<any>;
}

export async function getHealth() {
  const { data } = await ragClient.get("/health");
  return data;
}

export async function getChunks(limit = 12) {
  const { data } = await ragClient.get("/chunks", { params: { limit } });
  return data;
}

export async function reindexDocuments() {
  const { data } = await ragClient.post("/reindex");
  return data;
}

export async function askQuestion(message: string, conversationId?: string) {
  const { data } = await ragClient.post("/ask", {
    question: message,
    session_id: conversationId,
  });
  return data;
}

export async function searchDocuments(query: string, topK = 3) {
  const { data } = await ragClient.post("/search", {
    query,
    topK,
  });
  return data;
}

export const ragService: RagService = {
  getHealth,
  getChunks,
  reindexDocuments,
  askQuestion,
  searchDocuments,
};
