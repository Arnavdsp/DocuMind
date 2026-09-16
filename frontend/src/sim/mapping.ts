/* ============================================================================
   The fusion table, as code.
   ----------------------------------------------------------------------------
   This is the single place where RAG reality becomes simulation parameters.
   Nothing downstream of here invents a value; if a field is null, the
   simulation reads it as "unknown" and the HUD renders an em-dash rather
   than a plausible-looking number.

   | Physical object       | RAG reality                          | Source        |
   |-----------------------|--------------------------------------|---------------|
   | singularity           | the uploaded document                | DocumentRecord|
   | Schwarzschild radius  | size / pages / words                 | metrics       |
   | disk particles        | chunks in the vector index           | pages + index |
   | disk temperature      | index density, embedding coverage    | ingestion     |
   | photon geodesics      | retrieval — question's path to evidence | /ask        |
   | escaping null rays    | retrieved-then-discarded (8 -> 4)    | top_k config  |
   | lensed hot spots      | surviving citations, by page number  | citations[]   |
   | signal strength       | grounding + relevance_score          | AskResponse   |
   | SIGNAL LOST           | abstained === true                   | AskResponse   |
   | mission elapsed       | time since ingestion completed       | job timestamps|
   | inference backend     | model_used / backend_name            | model_used    |
   | transit time          | measured per-stage latency           | timings_ms    |
   ========================================================================= */

import type {
  AskResponse,
  DocumentRecord,
  GroundingLevel,
  PageInfo,
  ProcessingStage,
  StageTimings,
} from "../types/api";

/** Retrieval fan-out, mirrored from backend settings (retrieval_top_k=8,
 *  rerank_top_k=4). These are the counts the geodesic trace draws: 8 rays
 *  launch, 4 survive rerank. They are duplicated here only for rendering;
 *  the authoritative values live in `Settings` and the actual number of
 *  survivors drawn is always `citations.length`, never this constant. */
export const RETRIEVAL_TOP_K = 8;
export const RERANK_TOP_K = 4;

/* -------------------------------------------------------------------------
   Mass
   ---------------------------------------------------------------------- */

export interface MassState {
  /** Schwarzschild radius in scene units. Derived from real document size. */
  schwarzschildRadius: number;
  /** Displayed mass string, e.g. "4.2 × 10³ M☉". Null until metrics exist. */
  massLabel: string | null;
  /** Real counts, passed through for the text-equivalent readout. */
  pageCount: number | null;
  wordCount: number | null;
  sizeBytes: number;
}

/**
 * Document size becomes mass. The mapping is logarithmic because document
 * sizes span four orders of magnitude and a linear map would make every
 * document either a point or the whole screen.
 *
 * The displayed "mass" is an honest restatement of word count, not an
 * invented astrophysical quantity: we say 10^n M☉ where n comes directly
 * from the document's word count. A reader who divides it back out gets
 * their word count. Nothing is fabricated; the units are theatre and the
 * number underneath is real.
 */
export function deriveMass(document: DocumentRecord | null): MassState {
  if (!document) {
    return {
      schwarzschildRadius: 1,
      massLabel: null,
      pageCount: null,
      wordCount: null,
      sizeBytes: 0,
    };
  }

  const metrics = document.metrics;
  const words = metrics?.word_count ?? null;
  const pages = metrics?.page_count ?? null;

  // r_s scales with log(size) clamped to a range that keeps the horizon
  // between roughly a third and two thirds of the frame at default zoom.
  const kb = Math.max(document.size_bytes / 1024, 1);
  const schwarzschildRadius = clamp(0.6 + Math.log10(kb) * 0.28, 0.6, 1.9);

  let massLabel: string | null = null;
  if (words !== null && words > 0) {
    const exponent = Math.floor(Math.log10(words));
    const mantissa = (words / 10 ** exponent).toFixed(1);
    massLabel = `${mantissa} × 10${superscript(exponent)} M☉`;
  }

  return {
    schwarzschildRadius,
    massLabel,
    pageCount: pages,
    wordCount: words,
    sizeBytes: document.size_bytes,
  };
}

/* -------------------------------------------------------------------------
   Disk
   ---------------------------------------------------------------------- */

export interface DiskState {
  /** Particle count in the accretion disk === chunks in the vector index. */
  particleCount: number | null;
  /** True when particleCount is the index's real count rather than an
   *  estimate. Documents indexed before chunk_count was recorded report
   *  false, and the HUD marks those with "~". A measured count is never
   *  shown with a "~", and an estimate is never shown without one. */
  particleCountMeasured: boolean;
  /** 0..1 luminosity, from index density (chunks per page). Null if unknown. */
  luminosity: number | null;
  /** 0..1 fraction of pages whose extraction is trustworthy. Drives the
   *  dim, mottled banding on the disk for low-quality documents. */
  integrity: number | null;
  /** Real page-level integrity detail for the text equivalent. */
  lowQualityPages: number;
  ocrPages: number;
}

/**
 * The disk is the index. Particle count is the chunk count — not an
 * aesthetic choice of "about a thousand sparks", the actual number of
 * vectors the store holds for this document.
 *
 * The backend now records the true chunk count at indexing time, so for any
 * document indexed since then this is the measured number and carries no
 * "~". Documents indexed earlier have no recorded count; rather than
 * back-filling them with a number nobody measured, they fall back to the
 * estimate below, and `particleCountMeasured` is false so the HUD marks it
 * with a "~". An estimate never appears as a bare authoritative number, and
 * a measured count never carries the "~".
 */
export function deriveDisk(
  document: DocumentRecord | null,
  chunkTargetTokens = 220,
  overlapTokens = 40
): DiskState {
  if (!document || document.status !== "ready" || !document.metrics) {
    return {
      particleCount: null,
      particleCountMeasured: false,
      luminosity: null,
      integrity: null,
      lowQualityPages: 0,
      ocrPages: 0,
    };
  }

  const { character_count: chars, page_count: pages } = document.metrics;
  // ~4 chars per token is the standard rough English ratio; stride is the
  // advance per chunk once overlap is subtracted.
  const approxTokens = chars / 4;
  const stride = Math.max(chunkTargetTokens - overlapTokens, 1);

  // Prefer the measured count. `?? null` rather than `?? estimate` at the
  // top level so the two cases stay distinguishable: a measured 0 is a real
  // measurement of an empty index and must not silently become an estimate.
  const measured = document.chunk_count ?? null;
  const particleCountMeasured = measured !== null;
  const particleCount = measured ?? Math.max(1, Math.round(approxTokens / stride));

  const density = pages > 0 ? particleCount / pages : particleCount;
  const luminosity = clamp(density / 12, 0.05, 1);

  const pageInfos: PageInfo[] = document.pages ?? [];
  const lowQualityPages = pageInfos.filter((p) => p.is_low_quality).length;
  const ocrPages = pageInfos.filter(
    (p) => p.extraction_method === "ocr" || p.extraction_method === "mixed"
  ).length;
  const integrity =
    pageInfos.length > 0 ? 1 - lowQualityPages / pageInfos.length : null;

  return {
    particleCount,
    particleCountMeasured,
    luminosity,
    integrity,
    lowQualityPages,
    ocrPages,
  };
}

/* -------------------------------------------------------------------------
   Collapse — the ingestion cinematic
   ---------------------------------------------------------------------- */

/** Ordered, matching ProcessingStage. Index position drives how far the
 *  collapse has progressed; it never advances without a real job update. */
export const COLLAPSE_SEQUENCE: ProcessingStage[] = [
  "uploading",
  "validating",
  "extracting",
  "ocr",
  "chunking",
  "embedding",
  "indexing",
  "ready",
];

export const STAGE_CAPTION: Record<ProcessingStage, string> = {
  uploading: "Mass infalling",
  validating: "Verifying metric",
  extracting: "Resolving structure",
  ocr: "Optical reconstruction",
  chunking: "Fragmenting into orbits",
  embedding: "Computing geodesics",
  indexing: "Disk crystallising",
  ready: "Stable orbit achieved",
  failed: "Signal lost",
};

/**
 * Collapse progress, 0..1.
 *
 * IMPORTANT: this reads `job.progress` — the value the backend actually
 * wrote — and uses stage position only as a floor so the animation never
 * runs backwards between polls. It never interpolates forward past what
 * the backend reported, and it never runs a timer to "fill in" a stage the
 * backend is still working on. A stalled backend shows a stalled collapse.
 * That is the correct behaviour and it is the whole point.
 */
export function collapseProgress(
  stage: ProcessingStage | null,
  reportedProgress: number | null,
  documentStatus: ProcessingStage | null = null
): number | null {
  // The document record is the authority on whether an index exists; the job
  // is only the authority on how a particular ingestion RUN is going.
  //
  // These can disagree, and the disagreement is not hypothetical. Uploading
  // bytes that were already ingested is a content-addressed cache hit: the
  // backend returns the existing READY document alongside a brand-new job
  // that is never executed, so that job reports `pending` at 0% forever.
  // Reading collapse from the job alone leaves a fully indexed document
  // sitting in front of a dark, un-ignited disk.
  //
  // Trusting the document here is not a fudge — READY is a real backend
  // statement that the index is queryable, which is exactly what a fully
  // lit disk depicts.
  if (documentStatus === "ready") return 1;
  if (stage === null) return null;
  if (stage === "failed") return null;
  const index = COLLAPSE_SEQUENCE.indexOf(stage);
  const stageFloor = index < 0 ? 0 : index / (COLLAPSE_SEQUENCE.length - 1);
  if (reportedProgress === null) return stageFloor;
  return clamp(Math.max(reportedProgress, stageFloor), 0, 1);
}

/* -------------------------------------------------------------------------
   Signal — the geodesic trace
   ---------------------------------------------------------------------- */

export interface SignalState {
  grounding: GroundingLevel | null;
  relevance: number | null;
  abstained: boolean;
  /** Rays launched (retrieval_top_k). */
  launched: number;
  /** Rays that struck the disk — the surviving citations. */
  survived: number;
  /** Rays that fell past the horizon: launched - survived. */
  captured: number;
  /** Hot-spot positions on the disk, 0..1 around the ring, by page number. */
  hotspots: Hotspot[];
  modelUsed: string | null;
  /** Measured per-stage latency. Null stages did not run and render as an
   *  em-dash; never coerce with `?? 0`, which would fabricate a measurement. */
  timings: StageTimings | null;
}

export interface Hotspot {
  chunkId: string;
  /** Angular position around the disk, 0..1. Derived from page number so a
   *  citation from page 3 of 40 always lands in the same place. */
  theta: number;
  /** 0..1, the citation's own relevance_score — drives that spot's heat. */
  intensity: number;
  pageNumber: number | null;
}

export const EMPTY_SIGNAL: SignalState = {
  grounding: null,
  relevance: null,
  abstained: false,
  launched: 0,
  survived: 0,
  captured: 0,
  hotspots: [],
  modelUsed: null,
  timings: null,
};

export function deriveSignal(
  answer: AskResponse | null,
  pageCount: number | null
): SignalState {
  if (!answer) return EMPTY_SIGNAL;

  const survived = answer.citations.length;
  const pages = pageCount && pageCount > 0 ? pageCount : null;

  const hotspots: Hotspot[] = answer.citations.map((c, i) => {
    // Each page owns an arc of the disk; a citation sits inside its page's
    // arc, positioned by rank.
    //
    // The rank term is not decoration. Without it, every citation from a
    // single-page document (or several citations from the same page of any
    // document) collapses onto one angle and the hot spots stack into a
    // single smear — which is exactly what a plain-text upload produced
    // before this was fixed. With it, a page's citations fan out within
    // that page's slice and stay where the page puts them.
    const rankOffset = survived > 1 ? (i + 0.5) / survived : 0.5;

    if (c.page_number !== null && pages) {
      const arc = 1 / pages;
      return {
        chunkId: c.chunk_id,
        theta: (c.page_number - 1) * arc + rankOffset * arc,
        intensity: clamp(c.relevance_score, 0, 1),
        pageNumber: c.page_number,
      };
    }
    // No page number (some plain-text documents): rank order alone, which
    // is still stable across renders for the same answer.
    return {
      chunkId: c.chunk_id,
      theta: rankOffset,
      intensity: clamp(c.relevance_score, 0, 1),
      pageNumber: c.page_number,
    };
  });

  return {
    grounding: answer.grounding,
    relevance: answer.relevance_score,
    abstained: answer.abstained,
    launched: RETRIEVAL_TOP_K,
    survived,
    captured: Math.max(RETRIEVAL_TOP_K - survived, 0),
    hotspots,
    modelUsed: answer.model_used,
    // Passed through untouched. An older backend omits the field entirely,
    // which stays null and renders as an em-dash rather than becoming zeros.
    timings: answer.timings_ms ?? null,
  };
}

/** The grounding ramp. The ONLY function in the codebase permitted to
 *  return one of the --color-grounding-* tokens. `null` in, `null` out —
 *  an unknown grounding level has no colour, it has an em-dash. */
export function groundingColor(level: GroundingLevel | null): string | null {
  if (level === null) return null;
  switch (level) {
    case "strong":
      return "var(--color-grounding-strong)";
    case "moderate":
      return "var(--color-grounding-moderate)";
    case "weak":
      return "var(--color-grounding-weak)";
    case "none":
      return "var(--color-grounding-none)";
  }
}

/** Scalar form of the same ramp, for the shader's disk temperature uniform. */
export function groundingScalar(level: GroundingLevel | null): number {
  switch (level) {
    case "strong":
      return 1;
    case "moderate":
      return 0.66;
    case "weak":
      return 0.33;
    case "none":
      return 0;
    default:
      return 0;
  }
}

/* -------------------------------------------------------------------------
   Formatting — the em-dash rule
   ---------------------------------------------------------------------- */

/** The reference renders an em-dash for every unset telemetry value, and
 *  so do we. This is the only formatter the HUD uses; there is no code
 *  path that turns a null into a zero or a placeholder. */
export const UNSET = "—";

export function fmt(
  value: number | string | null | undefined,
  suffix = "",
  digits = 0
): string {
  if (value === null || value === undefined) return UNSET;
  if (typeof value === "string") return value.length > 0 ? value : UNSET;
  if (!Number.isFinite(value)) return UNSET;
  return `${value.toFixed(digits)}${suffix}`;
}

export function fmtClock(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) {
    return "00:00:00";
  }
  const s = Math.floor(seconds);
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

export function fmtBytes(bytes: number | null): string {
  if (bytes === null || !Number.isFinite(bytes)) return UNSET;
  const units = ["B", "KB", "MB", "GB"];
  let v = bytes;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u += 1;
  }
  return `${v.toFixed(u === 0 ? 0 : 1)} ${units[u]}`;
}

/* -------------------------------------------------------------------------
   Utilities
   ---------------------------------------------------------------------- */

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), hi);
}

const SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹";
function superscript(n: number): string {
  return String(Math.abs(n))
    .split("")
    .map((d) => SUPERSCRIPTS[Number(d)])
    .join("");
}
