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
  /** TRUE chunk count from the vector index, recorded at indexing time.
   *  Null for documents indexed before this was recorded — render as an
   *  em-dash or fall back to a clearly-labelled estimate, never `?? 0`. */
  chunk_count: number | null;
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

/** Measured wall-clock milliseconds per pipeline stage.
 *
 *  `null` means the stage did not run. It is never 0: a zero would be
 *  indistinguishable from an instantaneous stage and would quietly corrupt
 *  any percentile computed from these. Render null as an em-dash — never
 *  coerce it with `?? 0`. */
export interface StageTimings {
  embed_ms: number | null;
  lexical_ms: number | null;
  fuse_ms: number | null;
  rerank_ms: number | null;
  generate_ms: number | null;
  total_ms: number;
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
  timings_ms?: StageTimings | null;
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
  /** True only when content was genuinely lost. Segmentation alone is not
   *  truncation — the UI states "not the complete document" on this, so it
   *  must never fire merely because the input was split. */
  truncated: boolean;
  /** True when the caller declared the source language; false when it was
   *  detected or could not be determined. */
  source_language_detected?: boolean;
  segments_total?: number | null;
  segments_translated?: number | null;
  content_dropped?: boolean;
  /** "segment_missing" | "output_length_implausible" | null */
  dropped_reason?: string | null;
  /** Output characters per input character. Null when there was no input. */
  length_ratio?: number | null;
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
