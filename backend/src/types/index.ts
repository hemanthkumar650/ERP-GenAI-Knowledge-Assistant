export interface ChatRequest {
  message: string;
  conversationId?: string;
}

export interface ChatResponse {
  response: string;
  sources: string[];
  chunks: Array<Record<string, unknown>>;
  conversationId?: string;
  timestamp: string;
}

export interface SearchRequest {
  query: string;
  topK?: number;
}

export interface SearchResponse {
  results: Array<Record<string, unknown>>;
  count: number;
}
