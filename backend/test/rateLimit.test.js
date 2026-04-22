const assert = require("node:assert/strict");
const { once } = require("node:events");

process.env.RATE_LIMIT_DISABLED = "false";
process.env.RATE_LIMIT_MAX = "3";
process.env.RATE_LIMIT_WINDOW_MS = "60000";
process.env.HTTP_LOG_DISABLED = "true";

require("ts-node/register");

const { createApp } = require("../src/app");

function createFakeService() {
  return {
    getHealth: async () => ({ status: "ok", vector_db_loaded: true, indexed_chunks: 0 }),
    getChunks: async () => ({ chunks: [] }),
    reindexDocuments: async () => ({ status: "ok" }),
    askQuestion: async () => ({ answer: "", sources: [], chunks: [], session_id: "" }),
    searchDocuments: async () => ({ results: [], count: 0 }),
  };
}

async function withServer(service, run) {
  const app = createApp(service);
  const server = app.listen(0);
  await once(server, "listening");
  const { port } = server.address();
  const baseUrl = `http://127.0.0.1:${port}`;
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
  await runTest("/api routes return 429 once the configured limit is exceeded", async () => {
    await withServer(createFakeService(), async (baseUrl) => {
      for (let i = 0; i < 3; i += 1) {
        const ok = await fetch(`${baseUrl}/api/chunks?limit=1`);
        assert.equal(ok.status, 200, `request ${i + 1} should succeed`);
      }

      const limited = await fetch(`${baseUrl}/api/chunks?limit=1`);
      assert.equal(limited.status, 429);
      const body = await limited.json();
      assert.match(body.error, /too many requests/i);
      assert.ok(typeof body.requestId === "string" && body.requestId.length > 0);
    });
  });

  await runTest("GET /api/health is not counted toward the rate limit", async () => {
    await withServer(createFakeService(), async (baseUrl) => {
      for (let i = 0; i < 10; i += 1) {
        const response = await fetch(`${baseUrl}/api/health`);
        assert.equal(response.status, 200, `health request ${i + 1} should succeed`);
      }
    });
  });

  await runTest("/health (non-API) is not subject to the rate limiter", async () => {
    await withServer(createFakeService(), async (baseUrl) => {
      for (let i = 0; i < 6; i += 1) {
        const response = await fetch(`${baseUrl}/health`);
        assert.equal(response.status, 200, `health request ${i + 1} should succeed`);
      }
    });
  });

  if (!process.exitCode) {
    console.log("All rate limit tests passed.");
  }
}

main().catch((error) => {
  console.error("FAIL test runner");
  console.error(error);
  process.exit(1);
});
