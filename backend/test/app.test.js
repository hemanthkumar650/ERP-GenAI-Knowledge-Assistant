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

    const body = await response.json();
    assert.equal(body.status, "ok");
    assert.equal(body.service, "backend");
    assert.equal(typeof body.pythonRagUrl, "string");
    assert.ok(Date.parse(body.timestamp));
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

  await runTest("GET /api/chunks forwards the requested limit", async () => {
  let receivedLimit;
  const service = createFakeService();
  service.getChunks = async (limit = 12) => {
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

  await runTest("POST /api/chat trims the message and returns the normalized response", async () => {
  let receivedArgs;
  const service = createFakeService();
  service.askQuestion = async (message, conversationId) => {
    receivedArgs = { message, conversationId };
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
    assert.deepEqual(receivedArgs, {
      message: "What is the policy?",
      conversationId: "session-123",
    });
  });
  });

  await runTest("POST /api/search uses the default topK and returns mapped results", async () => {
  let receivedArgs;
  const service = createFakeService();
  service.searchDocuments = async (query, topK = 3) => {
    receivedArgs = { query, topK };
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
    assert.deepEqual(receivedArgs, {
      query: "expense reimbursement",
      topK: 3,
    });
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
