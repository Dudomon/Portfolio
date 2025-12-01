export type IngestPayload = {
  tenant_id: string;
  documents: { id?: string; text: string; metadata?: Record<string, unknown> }[];
};

export type ChatPayload = {
  tenant_id: string;
  question: string;
};

export type ChatResponse = {
  answer: string;
  sources: { id?: string; score: number; text: string; metadata: Record<string, unknown> }[];
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const apiKey = process.env.NEXT_PUBLIC_API_KEY ?? "";

async function postJson<TReq, TRes>(path: string, body: TReq): Promise<TRes> {
  const res = await fetch(`${apiBase}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { "X-API-Key": apiKey } : {}),
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Request failed with status ${res.status}`);
  }
  return res.json() as Promise<TRes>;
}

export async function ingestDocuments(payload: IngestPayload) {
  return postJson<IngestPayload, { ingested: number }>("/ingest", payload);
}

export async function chat(payload: ChatPayload) {
  return postJson<ChatPayload, ChatResponse>("/chat", payload);
}
