"use client";

import { FormEvent, useMemo, useState } from "react";
import { chat, ingestDocuments } from "../lib/api";

type Source = { id?: string; score: number; text: string; metadata: Record<string, unknown> };

export default function Home() {
  const [tenant, setTenant] = useState("tenant-demo");
  const [docText, setDocText] = useState("Support hours: 9am to 6pm. SLA: 4h response.");
  const [question, setQuestion] = useState("What is the SLA?");
  const [loadingIngest, setLoadingIngest] = useState(false);
  const [loadingChat, setLoadingChat] = useState(false);
  const [chatAnswer, setChatAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState<string | null>(null);

  const sourcesView = useMemo(
    () =>
      sources.map((s, idx) => (
        <div key={`${s.id ?? idx}`} className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="pill">Relevância {(s.score * 100).toFixed(1)}%</span>
            <span style={{ color: "#9ca3af", fontSize: 13 }}>{s.id ?? "doc"}</span>
          </div>
          <p style={{ marginTop: 12, lineHeight: 1.5 }}>{s.text}</p>
        </div>
      )),
    [sources],
  );

  async function handleIngest(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoadingIngest(true);
    try {
      await ingestDocuments({
        tenant_id: tenant,
        documents: [{ text: docText }],
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro inesperado");
    } finally {
      setLoadingIngest(false);
    }
  }

  async function handleChat(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoadingChat(true);
    try {
      const res = await chat({ tenant_id: tenant, question });
      setChatAnswer(res.answer);
      setSources(res.sources);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro inesperado");
    } finally {
      setLoadingChat(false);
    }
  }

  return (
    <div className="grid" style={{ gap: 24 }}>
      <section className="grid" style={{ gap: 12 }}>
        <div className="pill">RAG SaaS + Observabilidade</div>
        <h1 style={{ margin: 0, fontSize: 36, letterSpacing: -0.5 }}>
          Central de suporte com ingestão de base de conhecimento e tracing pronto para produção.
        </h1>
        <p style={{ color: "#cbd5e1", maxWidth: 780 }}>
          Use este protótipo para demonstrar RAG multi-tenant, ingestão rápida de documentos e monitoramento via
          OpenTelemetry. Configure o endpoint OTLP e um provedor de LLM para trocar o fallback determinístico por
          respostas geradas.
        </p>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <span className="pill">FastAPI</span>
          <span className="pill">Next.js</span>
          <span className="pill">OpenTelemetry</span>
          <span className="pill">RAG</span>
        </div>
        <p style={{ color: "#94a3b8", fontSize: 14 }}>
          Autenticação simples por X-API-Key (configure NEXT_PUBLIC_API_KEY no frontend e API_KEYS no backend).
        </p>
      </section>

      <section className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <form className="card grid" style={{ gap: 12 }} onSubmit={handleIngest}>
          <h2 style={{ margin: 0 }}>Ingestão de documentos</h2>
          <label>Tenant ID</label>
          <input value={tenant} onChange={(e) => setTenant(e.target.value)} />
          <label>Texto</label>
          <textarea rows={5} value={docText} onChange={(e) => setDocText(e.target.value)} />
          <button className="button" type="submit" disabled={loadingIngest}>
            {loadingIngest ? "Enviando..." : "Ingerir documento"}
          </button>
        </form>

        <form className="card grid" style={{ gap: 12 }} onSubmit={handleChat}>
          <h2 style={{ margin: 0 }}>Chat com RAG</h2>
          <label>Pergunta</label>
          <input value={question} onChange={(e) => setQuestion(e.target.value)} />
          <button className="button" type="submit" disabled={loadingChat}>
            {loadingChat ? "Consultando..." : "Perguntar"}
          </button>
          {chatAnswer && (
            <div style={{ marginTop: 8, lineHeight: 1.5 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Resposta</div>
              <div>{chatAnswer}</div>
            </div>
          )}
        </form>
      </section>

      {error && (
        <div className="card" style={{ borderColor: "#ef4444", color: "#fecaca" }}>
          {error}
        </div>
      )}

      {sources.length > 0 && (
        <section className="grid" style={{ gap: 10 }}>
          <h2 style={{ margin: 0 }}>Fontes recuperadas</h2>
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
            {sourcesView}
          </div>
        </section>
      )}
    </div>
  );
}
