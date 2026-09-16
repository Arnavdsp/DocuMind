import type {
  AskResponse,
  ApiErrorBody,
  DocumentPage,
  DocumentRecord,
  DocumentUploadResponse,
  JobRecord,
  SummarizeResponse,
  TranslateResponse,
} from "../types/api";
import { ApiError } from "../types/api";

// Same-origin by default: in the Colab/production build, FastAPI serves the
// built frontend itself, so relative paths just work. VITE_API_BASE_URL
// overrides this for local `vite dev` against a separately-running backend.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let body: ApiErrorBody | null = null;
    try {
      body = await response.json();
    } catch {
      // response body wasn't JSON; fall through to generic error below
    }
    throw new ApiError(
      response.status,
      body?.detail?.error_code ?? "unknown_error",
      body?.detail?.message ?? "Something went wrong. Please try again."
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<{ status: string; version: string }>("/health"),

  uploadDocument: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<DocumentUploadResponse>("/api/documents", { method: "POST", body: formData });
  },

  listDocuments: () => request<{ documents: DocumentRecord[] }>("/api/documents"),

  getDocument: (documentId: string) => request<DocumentRecord>(`/api/documents/${documentId}`),

  deleteDocument: (documentId: string) =>
    request<void>(`/api/documents/${documentId}`, { method: "DELETE" }),

  getDocumentPages: (documentId: string) =>
    request<{ document_id: string; pages: DocumentPage[] }>(`/api/documents/${documentId}/pages`),

  getJob: (jobId: string) => request<JobRecord>(`/api/jobs/${jobId}`),

  ask: (documentId: string, question: string, conversationId?: string) =>
    request<AskResponse>(`/api/documents/${documentId}/ask`, {
      method: "POST",
      body: JSON.stringify({ question, conversation_id: conversationId }),
    }),

  summarize: (documentId: string, forceRefresh = false) =>
    request<SummarizeResponse>(`/api/documents/${documentId}/summarize`, {
      method: "POST",
      body: JSON.stringify({ force_refresh: forceRefresh }),
    }),

  translate: (documentId: string, targetLanguage: string, sourceLanguage?: string) =>
    request<TranslateResponse>(`/api/documents/${documentId}/translate`, {
      method: "POST",
      body: JSON.stringify({ target_language: targetLanguage, source_language: sourceLanguage }),
    }),
};

export { ApiError };
