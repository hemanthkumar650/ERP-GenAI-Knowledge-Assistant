import { FormEvent, useEffect, useState } from "react";

import "./App.css";

type Chunk = {
  source: string;
  chunk_id: string;
  similarity?: number;
  text?: string;
  text_preview?: string;
};

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

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("Ask a grounded ERP policy question to begin.");
  const [sources, setSources] = useState<string[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [health, setHealth] = useState<HealthResponse>({ status: "checking" });
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();

  async function loadHealth() {
    const response = await fetch("/api/health");
    const data = (await readApiPayload(response)) as HealthResponse;
    setHealth(data);
  }

  async function loadChunks() {
    const response = await fetch("/api/chunks?limit=6");
    const data = await readApiPayload(response);
    setChunks(Array.isArray(data.chunks) ? data.chunks : []);
  }

  useEffect(() => {
    loadHealth().catch(() => setHealth({ status: "error" }));
    loadChunks().catch(() => setChunks([]));
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
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: prompt,
          conversationId,
        }),
      });
      const data = (await readApiPayload(response)) as ChatResponse | { error?: string };
      if (!response.ok || !("response" in data)) {
        throw new Error("error" in data ? data.error ?? "Chat request failed." : "Chat request failed.");
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
      const response = await fetch("/api/reindex", { method: "POST" });
      const data = await readApiPayload(response);
      if (!response.ok) {
        throw new Error(data.error ?? data.detail ?? "Reindex failed.");
      }
      await loadHealth();
      await loadChunks();
      setAnswer("Reindex completed. Ask a fresh question to use the updated knowledge base.");
    } catch (error) {
      setAnswer(error instanceof Error ? error.message : "Reindex failed.");
    } finally {
      setLoading(false);
    }
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
          <strong>{health.status}</strong>
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
          <form onSubmit={handleAsk} className="ask-form">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What is the company policy on expense reimbursement for remote staff?"
            />
            <button type="submit" className="primary-button" disabled={loading}>
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
