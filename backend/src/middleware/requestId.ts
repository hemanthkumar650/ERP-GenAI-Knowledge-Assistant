import { randomUUID } from "crypto";
import type { RequestHandler } from "express";

const MAX_LEN = 128;
const SAFE_INCOMING = /^[a-zA-Z0-9._-]+$/;

function normalizeIncoming(raw: string | undefined): string | null {
  if (raw === undefined) {
    return null;
  }
  const trimmed = raw.trim();
  if (trimmed.length === 0 || trimmed.length > MAX_LEN || !SAFE_INCOMING.test(trimmed)) {
    return null;
  }
  return trimmed;
}

export function assignRequestId(): RequestHandler {
  return (req, res, next) => {
    const id = normalizeIncoming(req.get("x-request-id")) ?? randomUUID();
    req.requestId = id;
    res.setHeader("X-Request-Id", id);
    next();
  };
}
