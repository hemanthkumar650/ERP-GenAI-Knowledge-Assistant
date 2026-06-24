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

    const zeroResponse = await fetch(`${baseUrl}/api/chunks?limit=0`);
    assert.equal(zeroResponse.status, 200);

    const hugeResponse = await fetch(`${baseUrl}/api/chunks?limit=999`);
    assert.equal(hugeResponse.status, 200);

    const decimalResponse = await fetch(`${baseUrl}/api/chunks?limit=4.9`);
    assert.equal(decimalResponse.status, 200);
  });

  assert.deepEqual(receivedLimits, [12, 12, 1, 50, 4]);
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

  assert.deepEqual(receivedTopKs, [3, 3, 3, 1, 10, 4]);
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
