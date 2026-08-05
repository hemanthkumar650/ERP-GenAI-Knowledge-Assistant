const assert = require("node:assert/strict");
const { once } = require("node:events");

process.env.HTTP_LOG_DISABLED = "true";

require("ts-node/register");

const { createApp } = require("../src/app");

function createFakeService() {
  return {
    getHealth: async () => ({ status: "ok", vector_db_loaded: true, indexed_chunks: 14 }),
    getChunks: async (limit = 12) => ({
      chunks: [{ id: "chunk-1", text: `limit:${limit}` }],
    }),
    reindexDocuments: async () => ({ status: "ok", indexed_chunks: 14 }),
    askQuestion: async (message, conversationId) => ({
      answer: `Echo: ${message}`,
      sources: ["policy.pdf"],
      chunks: [{ id: "chunk-1" }],
      session_id: conversationId,
    }),
    searchDocuments: async (query, topK = 3) => ({
      results: [{ id: "chunk-1", query, topK }],
      count: 1,
    }),
  };
}

async function withServer(service, run) {
  const app = createApp(service);
  const server = app.listen(0);
  await once(server, "listening");

  const address = server.address();
  const baseUrl = `http://127.0.0.1:${address.port}`;

  try {
    await run(baseUrl);
  } finally {
    server.close();
    await once(server, "close");
  }
}

async function runTest(name, fn) {
  try {
    await fn();
    console.log(`PASS ${name}`);
  } catch (error) {
    console.error(`FAIL ${name}`);
    console.error(error);
    process.exitCode = 1;
  }
}

async function main() {
  await runTest("GET /health returns backend metadata", async () => {
  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/health`);
    assert.equal(response.status, 200);

    const requestId = response.headers.get("x-request-id");
    assert.ok(requestId && requestId.length > 0);

    const body = await response.json();
    assert.equal(body.status, "ok");
    assert.equal(body.service, "backend");
    assert.equal(typeof body.pythonRagUrl, "string");
    assert.ok(Date.parse(body.timestamp));
  });
  });

  await runTest("echoes a valid incoming X-Request-Id header", async () => {
  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/health`, {
      headers: { "X-Request-Id": "trace-from-gateway-1" },
    });
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("x-request-id"), "trace-from-gateway-1");
  });
  });

  await runTest("trims safe incoming X-Request-Id headers before echoing", async () => {
  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/health`, {
      headers: { "X-Request-Id": "  trace-from-gateway-2  " },
    });

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("x-request-id"), "trace-from-gateway-2");
  });
  });

  await runTest("replaces unsafe incoming X-Request-Id headers", async () => {
  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/health`, {
      headers: { "X-Request-Id": "bad id with spaces" },
    });

    assert.equal(response.status, 200);
    const requestId = response.headers.get("x-request-id");
    assert.ok(requestId && requestId.length > 0);
    assert.notEqual(requestId, "bad id with spaces");
    assert.match(requestId, /^[a-zA-Z0-9._-]+$/);
  });
  });

  await runTest("replaces overlong incoming X-Request-Id headers", async () => {
  const overlongRequestId = "a".repeat(129);

  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/health`, {
      headers: { "X-Request-Id": overlongRequestId },
    });

    assert.equal(response.status, 200);
    const requestId = response.headers.get("x-request-id");
    assert.ok(requestId && requestId.length > 0);
    assert.notEqual(requestId, overlongRequestId);
    assert.match(requestId, /^[a-zA-Z0-9._-]+$/);
  });
  });

  await runTest("GET /api/health proxies the RAG health payload", async () => {
  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/health`);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      status: "ok",
      vector_db_loaded: true,
      indexed_chunks: 14,
    });
  });
  });

  await runTest("GET /api/health passes backend X-Request-Id to the RAG service", async () => {
  let seenId;
  const service = createFakeService();
  service.getHealth = async (requestId) => {
    seenId = requestId;
    return { status: "ok", vector_db_loaded: true, indexed_chunks: 14 };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/health`, {
      headers: { "X-Request-Id": "health-trace-123" },
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      status: "ok",
      vector_db_loaded: true,
      indexed_chunks: 14,
    });
    assert.equal(seenId, "health-trace-123");
  });
  });

  await runTest("GET /api/chunks forwards the requested limit", async () => {
  let receivedLimit;
  const service = createFakeService();
  service.getChunks = async (limit = 12, _requestId) => {
    receivedLimit = limit;
    return { chunks: [] };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chunks?limit=7`);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { chunks: [] });
    assert.equal(receivedLimit, 7);
  });
  });

  await runTest("GET /api/chunks passes backend X-Request-Id to the RAG service", async () => {
  let seenArgs;
  const service = createFakeService();
  service.getChunks = async (limit = 12, requestId) => {
    seenArgs = { limit, requestId };
    return { chunks: [] };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chunks?limit=5`, {
      headers: { "X-Request-Id": "chunks-trace-123" },
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { chunks: [] });
    assert.deepEqual(seenArgs, { limit: 5, requestId: "chunks-trace-123" });
  });
  });

  await runTest("GET /api/chunks normalizes invalid and oversized limits", async () => {
  const receivedLimits = [];
  const service = createFakeService();
  service.getChunks = async (limit = 12) => {
    receivedLimits.push(limit);
    return { chunks: [] };
  };

  await withServer(service, async (baseUrl) => {
    const invalidResponse = await fetch(`${baseUrl}/api/chunks?limit=abc`);
    assert.equal(invalidResponse.status, 200);

    const blankResponse = await fetch(`${baseUrl}/api/chunks?limit=   `);
    assert.equal(blankResponse.status, 200);

    const repeatedResponse = await fetch(`${baseUrl}/api/chunks?limit=7&limit=9`);
    assert.equal(repeatedResponse.status, 200);

    const zeroResponse = await fetch(`${baseUrl}/api/chunks?limit=0`);
    assert.equal(zeroResponse.status, 200);

    const hugeResponse = await fetch(`${baseUrl}/api/chunks?limit=999`);
    assert.equal(hugeResponse.status, 200);

    const decimalResponse = await fetch(`${baseUrl}/api/chunks?limit=4.9`);
    assert.equal(decimalResponse.status, 200);
  });

  assert.deepEqual(receivedLimits, [12, 12, 12, 1, 50, 4]);
  });

  await runTest("GET /api/chunks filters malformed chunk payloads", async () => {
  const service = createFakeService();
  service.getChunks = async () => ({
    chunks: [{ id: "chunk-1" }, null, "bad", ["nested"], { id: "chunk-2" }],
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chunks`);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      chunks: [{ id: "chunk-1" }, { id: "chunk-2" }],
    });
  });
  });

  await runTest("POST /api/reindex passes backend X-Request-Id to the RAG service", async () => {
  let seenId;
  const service = createFakeService();
  service.reindexDocuments = async (requestId) => {
    seenId = requestId;
    return { status: "ok", indexed_chunks: 21 };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/reindex`, {
      method: "POST",
      headers: { "X-Request-Id": "reindex-trace-123" },
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { status: "ok", indexed_chunks: 21 });
    assert.equal(seenId, "reindex-trace-123");
  });
  });

  await runTest("POST /api/reindex normalizes malformed payloads", async () => {
  const service = createFakeService();
  service.reindexDocuments = async () => ({
    status: "  complete  ",
    indexed_chunks: -7.8,
    extra: "ignored",
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/reindex`, {
      method: "POST",
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      status: "complete",
      indexed_chunks: 0,
    });
  });
  });

  await runTest("POST /api/chat rejects a blank message", async () => {
  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "   " }),
    });

    assert.equal(response.status, 400);
    assert.deepEqual(await response.json(), { error: "message is required" });
  });
  });

  await runTest("POST /api/chat rejects a non-string message", async () => {
  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: 42 }),
    });

    assert.equal(response.status, 400);
    assert.deepEqual(await response.json(), { error: "message is required" });
  });
  });

  await runTest("POST /api/chat rejects a missing body", async () => {
  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
    });

    assert.equal(response.status, 400);
    assert.deepEqual(await response.json(), { error: "message is required" });
  });
  });

  await runTest("POST /api/chat trims the message and returns the normalized response", async () => {
  let receivedArgs;
  const service = createFakeService();
  service.askQuestion = async (message, conversationId, requestId) => {
    receivedArgs = { message, conversationId, requestId };
    return {
      answer: "Approved",
      sources: ["expense-policy.pdf"],
      chunks: [{ id: "chunk-7" }],
      session_id: "session-123",
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "  What is the policy?  ", conversationId: "session-123" }),
    });

    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.response, "Approved");
    assert.deepEqual(body.sources, ["expense-policy.pdf"]);
    assert.deepEqual(body.chunks, [{ id: "chunk-7" }]);
    assert.equal(body.conversationId, "session-123");
    assert.ok(Date.parse(body.timestamp));
    assert.equal(receivedArgs.message, "What is the policy?");
    assert.equal(receivedArgs.conversationId, "session-123");
    assert.ok(typeof receivedArgs.requestId === "string" && receivedArgs.requestId.length > 0);
  });
  });

  await runTest("POST /api/chat normalizes conversationId before forwarding", async () => {
  const seenConversationIds = [];
  const service = createFakeService();
  service.askQuestion = async (_message, conversationId) => {
    seenConversationIds.push(conversationId);
    return {
      answer: "ok",
      sources: [],
      chunks: [],
      session_id: conversationId,
    };
  };

  await withServer(service, async (baseUrl) => {
    const trimmedResponse = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello", conversationId: "  session-42  " }),
    });
    assert.equal(trimmedResponse.status, 200);
    assert.equal((await trimmedResponse.json()).conversationId, "session-42");

    const blankResponse = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello", conversationId: "   " }),
    });
    assert.equal(blankResponse.status, 200);
    assert.equal((await blankResponse.json()).conversationId, undefined);
  });

  assert.deepEqual(seenConversationIds, ["session-42", undefined]);
  });

  await runTest("POST /api/chat normalizes returned session_id before responding", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: "ok",
    sources: [],
    chunks: [],
    session_id: "  session-from-rag  ",
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello", conversationId: "fallback-session" }),
    });

    assert.equal(response.status, 200);
    assert.equal((await response.json()).conversationId, "session-from-rag");
  });
  });

  await runTest("POST /api/chat falls back when returned session_id is not a string", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: "ok",
    sources: [],
    chunks: [],
    session_id: null,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello", conversationId: "fallback-session" }),
    });

    assert.equal(response.status, 200);
    assert.equal((await response.json()).conversationId, "fallback-session");
  });
  });

  await runTest("POST /api/chat falls back when returned session_id is blank", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: "ok",
    sources: [],
    chunks: [],
    session_id: "   ",
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello", conversationId: "fallback-session" }),
    });

    assert.equal(response.status, 200);
    assert.equal((await response.json()).conversationId, "fallback-session");
  });
  });

  await runTest("POST /api/chat leaves conversationId undefined when returned session_id is blank", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: "ok",
    sources: [],
    chunks: [],
    session_id: "   ",
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello" }),
    });

    assert.equal(response.status, 200);
    assert.equal((await response.json()).conversationId, undefined);
  });
  });

  await runTest("POST /api/chat ignores non-string returned session_id values", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: "ok",
    sources: [],
    chunks: [],
    session_id: { value: "bad-session" },
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello", conversationId: "fallback-session" }),
    });

    assert.equal(response.status, 200);
    assert.equal((await response.json()).conversationId, "fallback-session");
  });
  });

  await runTest("POST /api/chat normalizes malformed sources and chunks", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: "ok",
    sources: ["policy.pdf", 42, null],
    chunks: [{ id: "chunk-1" }, "bad-chunk", null, ["nested"]],
    session_id: undefined,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello" }),
    });

    assert.equal(response.status, 200);
    const body = await response.json();
    assert.deepEqual(body.sources, ["policy.pdf"]);
    assert.deepEqual(body.chunks, [{ id: "chunk-1" }]);
  });
  });

  await runTest("POST /api/chat drops non-plain object chunks", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: "ok",
    sources: ["policy.pdf"],
    chunks: [{ id: "chunk-1" }, new Date("2026-07-28T00:00:00.000Z")],
    session_id: undefined,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello" }),
    });

    assert.equal(response.status, 200);
    const body = await response.json();
    assert.deepEqual(body.chunks, [{ id: "chunk-1" }]);
  });
  });

  await runTest("POST /api/chat trims and drops blank sources", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: "ok",
    sources: ["  policy.pdf  ", "   ", "", null, 42],
    chunks: [],
    session_id: undefined,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello" }),
    });

    assert.equal(response.status, 200);
    const body = await response.json();
    assert.deepEqual(body.sources, ["policy.pdf"]);
  });
  });

  await runTest("POST /api/chat deduplicates trimmed sources", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: "ok",
    sources: ["  policy.pdf  ", "policy.pdf", " other.pdf ", "policy.pdf"],
    chunks: [],
    session_id: undefined,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello" }),
    });

    assert.equal(response.status, 200);
    const body = await response.json();
    assert.deepEqual(body.sources, ["policy.pdf", "other.pdf"]);
  });
  });

  await runTest("POST /api/chat normalizes a non-string answer", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: { text: "Approved" },
    sources: [],
    chunks: [],
    session_id: undefined,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello" }),
    });

    assert.equal(response.status, 200);
    assert.equal((await response.json()).response, "");
  });
  });

  await runTest("POST /api/chat trims and blanks whitespace answers", async () => {
  const service = createFakeService();
  service.askQuestion = async () => ({
    answer: "  Approved  ",
    sources: [],
    chunks: [],
    session_id: undefined,
  });

  await withServer(service, async (baseUrl) => {
    const trimmedResponse = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello" }),
    });

    assert.equal(trimmedResponse.status, 200);
    assert.equal((await trimmedResponse.json()).response, "Approved");
  });

  service.askQuestion = async () => ({
    answer: "   ",
    sources: [],
    chunks: [],
    session_id: undefined,
  });

  await withServer(service, async (baseUrl) => {
    const blankResponse = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello" }),
    });

    assert.equal(blankResponse.status, 200);
    assert.equal((await blankResponse.json()).response, "");
  });
  });

  await runTest("POST /api/chat ignores non-string conversationId values", async () => {
  let seenConversationId = "not-called";
  const service = createFakeService();
  service.askQuestion = async (_message, conversationId) => {
    seenConversationId = conversationId;
    return {
      answer: "ok",
      sources: [],
      chunks: [],
      session_id: conversationId,
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: "hello", conversationId: 42 }),
    });

    assert.equal(response.status, 200);
    assert.equal((await response.json()).conversationId, undefined);
    assert.equal(seenConversationId, undefined);
  });
  });

  await runTest("POST /api/search uses the default topK and returns mapped results", async () => {
  let receivedArgs;
  const service = createFakeService();
  service.searchDocuments = async (query, topK = 3, requestId) => {
    receivedArgs = { query, topK, requestId };
    return {
      results: [{ id: "chunk-3" }],
      count: 1,
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: " expense reimbursement " }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-3" }],
      count: 1,
    });
    assert.equal(receivedArgs.query, "expense reimbursement");
    assert.equal(receivedArgs.topK, 3);
    assert.ok(typeof receivedArgs.requestId === "string" && receivedArgs.requestId.length > 0);
  });
  });

  await runTest("POST /api/search falls back to the result length when count is missing", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, { id: "chunk-2" }],
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }, { id: "chunk-2" }],
      count: 2,
    });
  });
  });

  await runTest("POST /api/search clamps negative count values to zero", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, { id: "chunk-2" }],
    count: -7,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }, { id: "chunk-2" }],
      count: 0,
    });
  });
  });

  await runTest("POST /api/search falls back when count is not finite", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, { id: "chunk-2" }],
    count: Number.POSITIVE_INFINITY,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }, { id: "chunk-2" }],
      count: 2,
    });
  });
  });

  await runTest("POST /api/search falls back when count is a string", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, { id: "chunk-2" }, { id: "chunk-3" }],
    count: "3",
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }, { id: "chunk-2" }, { id: "chunk-3" }],
      count: 3,
    });
  });
  });

  await runTest("POST /api/search falls back when count is a numeric string", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, { id: "chunk-2" }],
    count: "0",
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }, { id: "chunk-2" }],
      count: 2,
    });
  });
  });

  await runTest("POST /api/search falls back when count is boolean", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, { id: "chunk-2" }, { id: "chunk-3" }],
    count: true,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }, { id: "chunk-2" }, { id: "chunk-3" }],
      count: 3,
    });
  });
  });

  await runTest("POST /api/search falls back when count is an object", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, { id: "chunk-2" }],
    count: { value: 2 },
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }, { id: "chunk-2" }],
      count: 2,
    });
  });
  });

  await runTest("POST /api/search falls back when count is an array", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, { id: "chunk-2" }, { id: "chunk-3" }],
    count: [3],
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }, { id: "chunk-2" }, { id: "chunk-3" }],
      count: 3,
    });
  });
  });

  await runTest("POST /api/search truncates fractional count values", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, { id: "chunk-2" }, { id: "chunk-3" }],
    count: 4.9,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }, { id: "chunk-2" }, { id: "chunk-3" }],
      count: 4,
    });
  });
  });

  await runTest("POST /api/search normalizes malformed results", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, "bad-result", null, ["nested"]],
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }],
      count: 1,
    });
  });
  });

  await runTest("POST /api/search drops non-plain object results", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: [{ id: "chunk-1" }, new Date("2026-07-28T00:00:00.000Z")],
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1" }],
      count: 1,
    });
  });
  });

  await runTest("POST /api/search preserves count when results is not an array", async () => {
  const service = createFakeService();
  service.searchDocuments = async () => ({
    results: { id: "chunk-1" },
    count: 5,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement" }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [],
      count: 5,
    });
  });
  });

  await runTest("POST /api/search passes backend X-Request-Id to the RAG service", async () => {
  let seenArgs;
  const service = createFakeService();
  service.searchDocuments = async (query, topK = 3, requestId) => {
    seenArgs = { query, topK, requestId };
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "X-Request-Id": "search-trace-123",
      },
      body: JSON.stringify({ query: " expense reimbursement ", topK: 4 }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { results: [], count: 0 });
    assert.deepEqual(seenArgs, {
      query: "expense reimbursement",
      topK: 4,
      requestId: "search-trace-123",
    });
  });
  });

  await runTest("POST /api/search rejects a non-string query", async () => {
  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: 42 }),
    });

    assert.equal(response.status, 400);
    assert.deepEqual(await response.json(), { error: "query is required" });
  });
  });

  await runTest("POST /api/search rejects a missing body", async () => {
  await withServer(createFakeService(), async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
    });

    assert.equal(response.status, 400);
    assert.deepEqual(await response.json(), { error: "query is required" });
  });
  });

  await runTest("POST /api/search normalizes invalid and oversized topK values", async () => {
  const receivedTopKs = [];
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => {
    receivedTopKs.push(topK);
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const invalidResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: "abc" }),
    });
    assert.equal(invalidResponse.status, 200);

    const nullResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: null }),
    });
    assert.equal(nullResponse.status, 200);

    const blankResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: "   " }),
    });
    assert.equal(blankResponse.status, 200);

    const arrayResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: [7] }),
    });
    assert.equal(arrayResponse.status, 200);

    const zeroResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: 0 }),
    });
    assert.equal(zeroResponse.status, 200);

    const hugeResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: 99 }),
    });
    assert.equal(hugeResponse.status, 200);

    const decimalResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: 4.8 }),
    });
    assert.equal(decimalResponse.status, 200);
  });

  assert.deepEqual(receivedTopKs, [3, 3, 3, 3, 1, 10, 4]);
  });

  await runTest("POST /api/search falls back when topK is an object", async () => {
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => ({
    results: [{ id: "chunk-1", topK }],
    count: 1,
  });

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: { value: 4 } }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      results: [{ id: "chunk-1", topK: 3 }],
      count: 1,
    });
  });
  });

  await runTest("POST /api/search falls back for ISO date string topK values", async () => {
  const receivedTopKs = [];
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => {
    receivedTopKs.push(topK);
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: "2026-07-13T00:00:00.000Z" }),
    });

    assert.equal(response.status, 200);
  });

  assert.deepEqual(receivedTopKs, [3]);
  });

  await runTest("POST /api/search normalizes boolean topK values", async () => {
  const receivedTopKs = [];
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => {
    receivedTopKs.push(topK);
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const trueResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: true }),
    });
    assert.equal(trueResponse.status, 200);

    const falseResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: false }),
    });
    assert.equal(falseResponse.status, 200);
  });

  assert.deepEqual(receivedTopKs, [1, 1]);
  });

  await runTest("POST /api/search parses string topK values", async () => {
  const receivedTopKs = [];
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => {
    receivedTopKs.push(topK);
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: "4" }),
    });

    assert.equal(response.status, 200);
  });

  assert.deepEqual(receivedTopKs, [4]);
  });

  await runTest("POST /api/search falls back for non-finite topK values", async () => {
  const receivedTopKs = [];
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => {
    receivedTopKs.push(topK);
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const infinityResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: Number.POSITIVE_INFINITY }),
    });
    assert.equal(infinityResponse.status, 200);

    const nanResponse = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: Number.NaN }),
    });
    assert.equal(nanResponse.status, 200);
  });

  assert.deepEqual(receivedTopKs, [3, 3]);
  });

  await runTest("POST /api/search truncates decimal string topK values", async () => {
  const receivedTopKs = [];
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => {
    receivedTopKs.push(topK);
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: "4.8" }),
    });

    assert.equal(response.status, 200);
  });

  assert.deepEqual(receivedTopKs, [4]);
  });

  await runTest("POST /api/search parses padded string topK values", async () => {
  const receivedTopKs = [];
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => {
    receivedTopKs.push(topK);
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: " 4 " }),
    });

    assert.equal(response.status, 200);
  });

  assert.deepEqual(receivedTopKs, [4]);
  });

  await runTest("POST /api/search parses scientific notation topK values", async () => {
  const receivedTopKs = [];
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => {
    receivedTopKs.push(topK);
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: "1e2" }),
    });

    assert.equal(response.status, 200);
  });

  assert.deepEqual(receivedTopKs, [10]);
  });

  await runTest("POST /api/search clamps negative string topK values", async () => {
  const receivedTopKs = [];
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => {
    receivedTopKs.push(topK);
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: "-5" }),
    });

    assert.equal(response.status, 200);
  });

  assert.deepEqual(receivedTopKs, [1]);
  });

  await runTest("POST /api/search parses hex string topK values", async () => {
  const receivedTopKs = [];
  const service = createFakeService();
  service.searchDocuments = async (_query, topK = 3) => {
    receivedTopKs.push(topK);
    return {
      results: [],
      count: 0,
    };
  };

  await withServer(service, async (baseUrl) => {
    const response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "expense reimbursement", topK: "0x10" }),
    });

    assert.equal(response.status, 200);
  });

  assert.deepEqual(receivedTopKs, [10]);
  });

  await runTest("passes backend X-Request-Id through to the RAG service on /api/chat", async () => {
    let seenId;
    const service = createFakeService();
    service.askQuestion = async (_message, _conversationId, requestId) => {
      seenId = requestId;
      return {
        answer: "ok",
        sources: [],
        chunks: [],
        session_id: undefined,
      };
    };

    await withServer(service, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/api/chat`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "X-Request-Id": "upstream-trace-abc",
        },
        body: JSON.stringify({ message: "hello" }),
      });

      assert.equal(response.status, 200);
      assert.equal(seenId, "upstream-trace-abc");
    });
  });

  await runTest("unknown routes return a JSON 404 with requestId", async () => {
    await withServer(createFakeService(), async (baseUrl) => {
      const response = await fetch(`${baseUrl}/api/does-not-exist`, {
        headers: { "X-Request-Id": "missing-route-test" },
      });

      assert.equal(response.status, 404);
      assert.match(response.headers.get("content-type") ?? "", /application\/json/);
      assert.deepEqual(await response.json(), {
        error: "Route not found",
        requestId: "missing-route-test",
      });
    });
  });

  await runTest("malformed JSON requests return a JSON 400 with requestId", async () => {
    await withServer(createFakeService(), async (baseUrl) => {
      const response = await fetch(`${baseUrl}/api/chat`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "X-Request-Id": "bad-json-test",
        },
        body: '{"message":',
      });

      assert.equal(response.status, 400);
      assert.match(response.headers.get("content-type") ?? "", /application\/json/);
      const body = await response.json();
      assert.match(body.error, /json/i);
      assert.equal(body.requestId, "bad-json-test");
    });
  });

  if (!process.exitCode) {
    console.log("All backend API tests passed.");
  }
}

main().catch((error) => {
  console.error("FAIL test runner");
  console.error(error);
  process.exit(1);
});
