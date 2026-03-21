import dotenv from "dotenv";

dotenv.config({ path: "../.env" });
dotenv.config();

export const environment = {
  port: Number(process.env.PORT ?? 5000),
  nodeEnv: process.env.NODE_ENV ?? "development",
  pythonRagUrl: process.env.PYTHON_RAG_URL ?? "http://localhost:8000",
};
