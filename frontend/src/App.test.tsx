import { act } from "react";
import ReactDOMClient, { Root } from "react-dom/client";

import App from "./App";

declare global {
  interface Window {
    IS_REACT_ACT_ENVIRONMENT?: boolean;
  }
}

function createJsonResponse(payload: unknown, init?: { ok?: boolean; status?: number }): Response {
  return {
    ok: init?.ok ?? true,
    status: init?.status ?? 200,
    headers: {
      get: (name: string) => (name.toLowerCase() === "content-type" ? "application/json" : null),
    } as Headers,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  } as Response;
}

async function flushUi() {
  await Promise.resolve();
  await Promise.resolve();
}

function setTextareaValue(element: HTMLTextAreaElement, value: string) {
  const descriptor = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
  descriptor?.set?.call(element, value);
}

function findButton(container: HTMLElement, label: string): HTMLButtonElement {
  const button = Array.from(container.querySelectorAll("button")).find(
    (candidate) => candidate.textContent?.trim() === label
  );

  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`Unable to find button: ${label}`);
  }

  return button;
}

function getHeaderValue(headers: HeadersInit | undefined, name: string): string | null {
  if (!headers) {
    return null;
  }

  if (headers instanceof Headers) {
    return headers.get(name);
  }

  if (Array.isArray(headers)) {
    const pair = headers.find(([key]) => key.toLowerCase() === name.toLowerCase());
    return pair ? pair[1] : null;
  }

  const record = headers as Record<string, string>;
  for (const [key, value] of Object.entries(record)) {
    if (key.toLowerCase() === name.toLowerCase()) {
      return value;
    }
  }
  return null;
}

describe("App", () => {
  let container: HTMLDivElement;
  let root: Root | null;
  let fetchMock: jest.MockedFunction<typeof fetch>;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = null;
    fetchMock = jest.fn() as jest.MockedFunction<typeof fetch>;
    window.IS_REACT_ACT_ENVIRONMENT = true;
    global.fetch = fetchMock;
  });

  afterEach(async () => {
    if (root) {
      await act(async () => {
        root?.unmount();
      });
    }

    container.remove();
    jest.restoreAllMocks();
  });

  async function renderApp() {
    root = ReactDOMClient.createRoot(container);

    await act(async () => {
      root?.render(<App />);
      await flushUi();
    });
  }

  it("loads health and retrieved chunks on startup", async () => {
    fetchMock.mockImplementation(async (input) => {
      const url = String(input);
      if (url === "/api/health") {
        return createJsonResponse({ status: "ok", indexed_chunks: 14 });
      }

      if (url === "/api/chunks?limit=6") {
        return createJsonResponse({
          chunks: [
            {
              source: "travel-policy.pdf",
              chunk_id: "chunk-1",
              similarity: 0.1234,
              policy_type: "Travel",
              effective_date: "2023-01-01",
              department: "Finance",
              version: "2023",
              text_preview: "Managers must approve travel in advance.",
            },
          ],
        });
      }

      throw new Error(`Unexpected request: ${url}`);
    });

    await renderApp();

    expect(container.textContent).toContain("ok");
    expect(container.textContent).toContain("14");
    expect(container.textContent).toContain("travel-policy.pdf");
    expect(container.textContent).toContain("chunk-1");
    expect(container.textContent).toContain("Managers must approve travel in advance.");
    expect(container.textContent).toContain("Policy type");
    expect(container.textContent).toContain("Travel");
    expect(container.textContent).toContain("Finance");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("shows a clear message when chat is rate limited (429)", async () => {
    fetchMock.mockImplementation(async (input) => {
      const url = String(input);
      if (url === "/api/health") {
        return createJsonResponse({ status: "ok", indexed_chunks: 5 });
      }
      if (url === "/api/chunks?limit=6") {
        return createJsonResponse({ chunks: [] });
      }
      if (url === "/api/chat") {
        return createJsonResponse(
          { error: "Too many requests. Please slow down and try again in a few moments." },
          { ok: false, status: 429 }
        );
      }
      throw new Error(`Unexpected request: ${url}`);
    });

    await renderApp();

    const textarea = container.querySelector("textarea");
    const form = container.querySelector("form");
    if (!(textarea instanceof HTMLTextAreaElement) || !(form instanceof HTMLFormElement)) {
      throw new Error("Expected ask form controls to exist.");
    }

    await act(async () => {
      setTextareaValue(textarea, "Test question");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
      await flushUi();
      await flushUi();
    });

    expect(container.textContent).toContain("Too many requests");
  });

  it("submits a trimmed question and renders the answer state", async () => {
    let healthCalls = 0;

    fetchMock.mockImplementation(async (input, init) => {
      const url = String(input);

      if (url === "/api/health") {
        healthCalls += 1;
        return createJsonResponse({
          status: "ok",
          indexed_chunks: healthCalls === 1 ? 14 : 15,
        });
      }

      if (url === "/api/chunks?limit=6") {
        return createJsonResponse({ chunks: [] });
      }

      if (url === "/api/chat") {
        expect(init?.method).toBe("POST");
        expect(getHeaderValue(init?.headers as HeadersInit | undefined, "x-request-id")).toBeTruthy();
        expect(init?.body).toBe(
          JSON.stringify({
            message: "What is the travel policy?",
            conversationId: undefined,
          })
        );

        return createJsonResponse({
          response: "Travel must be approved by a manager.",
          sources: ["travel-policy.pdf"],
          chunks: [
            {
              source: "travel-policy.pdf",
              chunk_id: "chunk-2",
              retrieval: "hybrid",
              policy_type: "Travel",
              text_preview: "Approval is required.",
            },
          ],
          conversationId: "session-99",
          timestamp: "2026-03-22T12:00:00.000Z",
        });
      }

      throw new Error(`Unexpected request: ${url}`);
    });

    await renderApp();

    const textarea = container.querySelector("textarea");
    const form = container.querySelector("form");

    if (!(textarea instanceof HTMLTextAreaElement) || !(form instanceof HTMLFormElement)) {
      throw new Error("Expected ask form controls to exist.");
    }

    await act(async () => {
      setTextareaValue(textarea, "  What is the travel policy?  ");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new Event("change", { bubbles: true }));
      form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
      await flushUi();
      await flushUi();
    });

    expect(container.textContent).toContain("Travel must be approved by a manager.");
    expect(container.textContent).toContain("travel-policy.pdf");
    expect(container.textContent).toContain("Approval is required.");
    expect(container.textContent).toContain("hybrid");
    expect(container.textContent).toContain("active");
    expect(container.textContent).toContain("15");
  });

  it("reindexes documents and refreshes dashboard data", async () => {
    let healthCalls = 0;

    fetchMock.mockImplementation(async (input, init) => {
      const url = String(input);

      if (url === "/api/health") {
        healthCalls += 1;
        return createJsonResponse({
          status: "ok",
          indexed_chunks: healthCalls === 1 ? 10 : 18,
        });
      }

      if (url === "/api/chunks?limit=6") {
        return createJsonResponse({
          chunks: healthCalls === 0 ? [] : [{ source: "erp-policy.pdf", chunk_id: "chunk-8", text_preview: "Updated chunk." }],
        });
      }

      if (url === "/api/reindex") {
        expect(init?.method).toBe("POST");
        return createJsonResponse({ status: "ok", indexed_chunks: 18 });
      }

      throw new Error(`Unexpected request: ${url}`);
    });

    await renderApp();

    await act(async () => {
      findButton(container, "Reindex Policies").click();
      await flushUi();
    });

    expect(container.textContent).toContain("Reindex completed. Ask a fresh question to use the updated knowledge base.");
    expect(container.textContent).toContain("18");
    expect(container.textContent).toContain("erp-policy.pdf");
    expect(container.textContent).toContain("Updated chunk.");
  });
});
