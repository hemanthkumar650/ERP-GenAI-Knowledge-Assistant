import { FormEvent, useEffect, useState } from "react";

import "./App.css";

type Chunk = {
  source: string;
  chunk_id: string;
  similarity?: number;
  text?: string;
  text_preview?: string;
  retrieval?: string;
  policy_type?: string;
  effective_date?: string;
  department?: string;
  version?: string;
};

function isMeaningfulMeta(value: string | undefined): value is string {
  const v = value?.trim().toLowerCase();
  return Boolean(v && v !== "unknown");
}

function ChunkEvidenceMeta({ chunk }: { chunk: Chunk }) {
  const pairs: { label: string; value: string }[] = [];
  if (isMeaningfulMeta(chunk.policy_type)) {
    pairs.push({ label: "Policy type", value: chunk.policy_type });
  }
  if (isMeaningfulMeta(chunk.effective_date)) {
    pairs.push({ label: "Effective", value: chunk.effective_date });
  }
  if (isMeaningfulMeta(chunk.department)) {
    pairs.push({ label: "Department", value: chunk.department });
  }
  if (isMeaningfulMeta(chunk.version)) {
    pairs.push({ label: "Version", value: chunk.version });
  }

  const mode = chunk.retrieval?.trim().toLowerCase();
  const showRetrieval = mode === "hybrid" || mode === "vector";

  if (!showRetrieval && pairs.length === 0) {
    return null;
  }

  return (
    <div className="chunk-enriched" aria-label="Chunk metadata">
      {showRetrieval ? (
        <span className={`chip chip-retrieval chip-retrieval--${mode}`} title="How this chunk was retrieved">
          {mode}
        </span>
      ) : null}
      {pairs.map(({ label, value }) => (
        <span key={label} className="chunk-meta-pair" title={label}>
          <span className="chunk-meta-label">{label}</span>
          {value}
        </span>
      ))}
    </div>
  );
}

type ChatResponse = {
  response: string;
  sources: string[];
  chunks: Chunk[];
  conversationId?: string;
  timestamp: string;
};

type HealthResponse = {
  status: string;
  indexed_chunks?: number;
};

async function readApiPayload(response: Response) {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return response.json();
  }

  const text = await response.text();
  if (text.trim().startsWith("<!DOCTYPE") || text.trim().startsWith("<html")) {
    throw new Error("The frontend reached an HTML page instead of the API. Make sure the backend on port 5000 is running.");
  }

  throw new Error(text || "Unexpected non-JSON response from the API.");
}

class HttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "HttpError";
  }
}

function createRequestId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  return `req-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function withRequestId(init?: RequestInit): RequestInit {
  const headers = new Headers(init?.headers);
  if (!headers.has("X-Request-Id")) {
    headers.set("X-Request-Id", createRequestId());
  }

  return { ...init, headers };
}

function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  return fetch(input, withRequestId(init));
}

function throwIfNotOk(response: Response, data: unknown): void {
  if (response.ok) {
    return;
  }

  const body = data as { error?: string; detail?: string };
  const fromServer = body.error?.trim() || body.detail?.trim();
  const message =
    response.status === 429
      ? fromServer || "Too many requests. Please wait a moment and try again."
      : fromServer || `Request failed (${response.status}).`;

  throw new HttpError(message, response.status);
}

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("Ask a grounded ERP policy question to begin.");
  const [sources, setSources] = useState<string[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [health, setHealth] = useState<HealthResponse>({ status: "checking" });
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();

  async function loadHealth() {
    try {
      const response = await apiFetch("/api/health");
      const data = (await readApiPayload(response)) as HealthResponse;
      throwIfNotOk(response, data);
      setHealth(data);
    } catch (error) {
      if (error instanceof HttpError && error.status === 429) {
        setHealth({ status: "rate_limited" });
      } else {
        setHealth({ status: "error" });
      }
    }
  }

  async function loadChunks() {
    try {
      const response = await apiFetch("/api/chunks?limit=6");
      const data = await readApiPayload(response);
      throwIfNotOk(response, data);
      setChunks(Array.isArray(data.chunks) ? data.chunks : []);
    } catch {
      setChunks([]);
    }
  }

  useEffect(() => {
    void loadHealth();
    void loadChunks();
  }, []);

  async function handleAsk(event: FormEvent) {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt) {
      return;
    }

    setLoading(true);
    setAnswer("Thinking through retrieved policy evidence...");

    try {
      const response = await apiFetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: prompt,
          conversationId,
        }),
      });
      const data = (await readApiPayload(response)) as ChatResponse | { error?: string };
      throwIfNotOk(response, data);
      if (!("response" in data)) {
        throw new Error("Chat request failed.");
      }

      setAnswer(data.response);
      setSources(data.sources ?? []);
      setChunks(data.chunks ?? []);
      setConversationId(data.conversationId);
      await loadHealth();
    } catch (error) {
      setAnswer(error instanceof Error ? error.message : "Chat request failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleReindex() {
    setLoading(true);
    setAnswer("Reindexing policy documents...");
    try {
      const response = await apiFetch("/api/reindex", { method: "POST" });
      const data = await readApiPayload(response);
      throwIfNotOk(response, data);
      await loadHealth();
      await loadChunks();
      setAnswer("Reindex completed. Ask a fresh question to use the updated knowledge base.");
    } catch (error) {
      setAnswer(error instanceof Error ? error.message : "Reindex failed.");
    } finally {
      setLoading(false);
    }
  }

  function handleNewConversation() {
    setConversationId(undefined);
    setQuestion("");
    setSources([]);
    setAnswer("Ask a grounded ERP policy question to begin.");
  }

  return (
    <div className="app">
      <div className="backdrop backdrop-left" />
      <div className="backdrop backdrop-right" />

      <header className="hero">
        <div>
          <p className="eyebrow">ERP-GenAI-Knowledge-Assistant</p>
          <h1>ERP-GenAI-Knowledge-Assistant</h1>
          <p className="lede">
            React on the front, Node in the middle, Python for retrieval, and a background worker keeping
            policy data fresh.
          </p>
        </div>
        <div className="hero-actions">
          <button type="button" className="ghost-button" onClick={() => void loadHealth()}>
            Refresh Health
          </button>
          <button type="button" className="primary-button" onClick={() => void handleReindex()} disabled={loading}>
            Reindex Policies
          </button>
        </div>
      </header>

      <section className="stats-grid">
        <article className="stat-card">
          <span className="stat-label">System</span>
          <strong title={health.status}>
            {health.status === "rate_limited" ? "Rate limited" : health.status}
          </strong>
        </article>
        <article className="stat-card">
          <span className="stat-label">Indexed Chunks</span>
          <strong>{health.indexed_chunks ?? "-"}</strong>
        </article>
        <article className="stat-card">
          <span className="stat-label">Conversation</span>
          <strong>{conversationId ? "active" : "new"}</strong>
        </article>
      </section>

      <main className="main-grid">
        <section className="panel ask-panel">
          <div className="panel-heading">
            <h2>Ask A Policy Question</h2>
            <span>{loading ? "Working..." : "Ready"}</span>
          </div>
          <div className="ask-actions">
            <button type="button" className="ghost-button" onClick={handleNewConversation} disabled={loading}>
              New Conversation
            </button>
          </div>
          <form onSubmit={handleAsk} className="ask-form">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What is the company policy on expense reimbursement for remote staff?"
            />
            <button type="submit" className="primary-button" disabled={loading || !question.trim()}>
              Get Answer
            </button>
          </form>
        </section>

        <section className="panel answer-panel">
          <div className="panel-heading">
            <h2>Grounded Answer</h2>
            <span>Azure OpenAI + Chroma</span>
          </div>
          <pre>{answer}</pre>
        </section>

        <section className="panel">
          <div className="panel-heading">
            <h2>Sources</h2>
            <span>{sources.length} file(s)</span>
          </div>
          <div className="chip-list">
            {sources.length ? sources.map((source) => <span key={source} className="chip">{source}</span>) : <p>No sources yet.</p>}
          </div>
        </section>

        <section className="panel">
          <div className="panel-heading">
            <h2>Retrieved Evidence</h2>
            <span>{chunks.length} chunk(s)</span>
          </div>
          <div className="chunk-list">
            {chunks.length ? (
              chunks.map((chunk) => (
                <article key={`${chunk.source}-${chunk.chunk_id}`} className="chunk-card">
                  <div className="chunk-meta">
                    <span className="chip">{chunk.source}</span>
                    <code>{chunk.chunk_id}</code>
                    {typeof chunk.similarity === "number" ? <span>sim {chunk.similarity.toFixed(3)}</span> : null}
                  </div>
                  <ChunkEvidenceMeta chunk={chunk} />
                  <p>{chunk.text_preview ?? chunk.text ?? "No preview available."}</p>
                </article>
              ))
            ) : (
              <p>No chunks loaded.</p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
