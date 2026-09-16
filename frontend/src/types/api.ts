export type ProcessingStage =
  | "uploading"
  | "validating"
  | "extracting"
  | "ocr"
  | "chunking"
  | "embedding"
  | "indexing"
  | "ready"
  | "failed";

export type ExtractionMethod = "native_text" | "ocr" | "mixed" | "plain_text";

export interface PageInfo {
  page_number: number;
  char_count: number;
  extraction_method: ExtractionMethod;
  ocr_confidence: number | null;
  is_low_quality: boolean;
}

export interface DocumentSummaryMetrics {
  word_count: number;
  character_count: number;
  page_count: number;
  estimated_reading_minutes: number;
}

export interface DocumentRecord {
  document_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: ProcessingStage;
  created_at: string;
  updated_at: string;
  metrics: DocumentSummaryMetrics | null;
  pages: PageInfo[];
  error_message: string | null;
}

export interface DocumentUploadResponse {
  document: DocumentRecord;
  job_id: string;
}

export type JobStatus = "pending" | "running" | "succeeded" | "failed";

export interface JobRecord {
  job_id: string;
  document_id: string;
  status: JobStatus;
  stage: ProcessingStage;
  progress: number;
  error_message: string | null;
}

export type GroundingLevel = "none" | "weak" | "moderate" | "strong";

export interface Citation {
  chunk_id: string;
  page_number: number | null;
  section: string | null;
  snippet: string;
  relevance_score: number;
}

export interface AskResponse {
  conversation_id: string;
  question: string;
  answer: string;
  abstained: boolean;
  grounding: GroundingLevel;
  relevance_score: number;
  citations: Citation[];
  model_used: string;
}

export interface StructuredSummary {
  executive_summary: string;
  key_findings: string[];
  important_numbers: string[];
  methodology: string | null;
  limitations: string | null;
}

export interface SummarizeResponse {
  document_id: string;
  summary: StructuredSummary;
  strategy: string;
  model_used: string;
  cached: boolean;
}

export interface TranslateResponse {
  document_id: string;
  source_language: string;
  target_language: string;
  translated_text: string;
  provider: string;
  truncated: boolean;
}

export interface DocumentPage {
  page_number: number;
  text: string;
  extraction_method: ExtractionMethod;
  ocr_confidence: number | null;
  is_low_quality: boolean;
}

export interface ApiErrorBody {
  detail: {
    error_code: string;
    message: string;
    request_id: string | null;
  };
}

export class ApiError extends Error {
  errorCode: string;
  status: number;

  constructor(status: number, errorCode: string, message: string) {
    super(message);
    this.status = status;
    this.errorCode = errorCode;
  }
}
