import dotenv from "dotenv";

dotenv.config({ path: "../.env" });
dotenv.config();

function parsePositiveNumber(raw: string | undefined, fallback: number): number {
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export const environment = {
  port: Number(process.env.PORT ?? 5000),
  nodeEnv: process.env.NODE_ENV ?? "development",
  pythonRagUrl: process.env.PYTHON_RAG_URL ?? "http://localhost:8000",
  rateLimit: {
    windowMs: parsePositiveNumber(process.env.RATE_LIMIT_WINDOW_MS, 60_000),
    max: parsePositiveNumber(process.env.RATE_LIMIT_MAX, 60),
    disabled:
      (process.env.RATE_LIMIT_DISABLED ?? "").toLowerCase() === "true" ||
      process.env.NODE_ENV === "test",
  },
  logging: {
    format: process.env.HTTP_LOG_FORMAT ?? "dev",
    disabled:
      (process.env.HTTP_LOG_DISABLED ?? "").toLowerCase() === "true" ||
      process.env.NODE_ENV === "test",
  },
};
