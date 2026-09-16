import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { NavigationPanel, type Toggles } from "./hud/navigation-panel";
import { TelemetryRow } from "./hud/telemetry";
import { TitleBlock } from "./hud/title-block";
import { useJobPolling } from "./hooks/use-job-polling";
import { useReducedMotion } from "./hooks/use-reduced-motion";
import { BriefingPanel } from "./panels/briefing-panel";
import { CollapsePanel } from "./panels/collapse-panel";
import { FrequencyPanel } from "./panels/frequency-panel";
import { IntegrityPanel } from "./panels/integrity-panel";
import { QueryPanel } from "./panels/query-panel";
import { StaticFallback } from "./scenes/static-fallback";
import { api, ApiError } from "./services/api";
import { PRESET_ORDER, type CameraPreset } from "./sim/camera";
import {
  collapseProgress,
  deriveDisk,
  deriveMass,
  deriveSignal,
  fmt,
  groundingColor,
  groundingScalar,
  UNSET,
} from "./sim/mapping";
import { DEFAULT_INPUTS, seedFromId, type SimInputs, type Telemetry } from "./sim/types";
import type {
  AskResponse,
  Citation,
  DocumentPage,
  DocumentRecord,
  SummarizeResponse,
  TranslateResponse,
} from "./types/api";

/* The scene is lazy so the shader source and the WebGL host never sit on
   the critical path. First paint is the HUD and the upload affordance; the
   render arrives when it arrives, and the app is fully usable before it
   does. This is the single most important performance decision in the
   build — a shader compile on the upload path is how a cinematic interface
   makes a document tool feel broken. */
const GargantuaCanvas = lazy(() => import("./scenes/gargantua-canvas"));

type PanelId = "collapse" | "query" | "briefing" | "frequency" | "integrity";

const PANELS: Array<{ id: PanelId; label: string }> = [
  { id: "collapse", label: "Mass" },
  { id: "query", label: "Query" },
  { id: "briefing", label: "Brief" },
  { id: "frequency", label: "Shift" },
  { id: "integrity", label: "Source" },
];

export default function App() {
  const reducedMotion = useReducedMotion();

  // --- Document / ingestion state ----------------------------------------
  const [doc, setDoc] = useState<DocumentRecord | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploadName, setUploadName] = useState<string | null>(null);
  const [uploadSize, setUploadSize] = useState<number | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const { job, error: pollError } = useJobPolling(jobId);

  // --- Answer state -------------------------------------------------------
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [traceProgress, setTraceProgress] = useState(0);

  // --- Other capabilities --------------------------------------------------
  const [summary, setSummary] = useState<SummarizeResponse | null>(null);
  const [summaryPending, setSummaryPending] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  const [translation, setTranslation] = useState<TranslateResponse | null>(null);
  const [translatePending, setTranslatePending] = useState(false);
  const [translateError, setTranslateError] = useState<string | null>(null);
  const [redshift, setRedshift] = useState(0);

  const [pages, setPages] = useState<DocumentPage[]>([]);
  const [pagesPending, setPagesPending] = useState(false);
  const [pagesError, setPagesError] = useState<string | null>(null);

  // --- Simulation / HUD ----------------------------------------------------
  const [preset, setPreset] = useState<CameraPreset>("poster");
  const [toggles, setToggles] = useState<Toggles>({
    auto: true,
    cinematic: false,
    params: false,
    hud: true,
    sound: false,
  });
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [renderFailed, setRenderFailed] = useState<null | "unsupported" | "context-lost">(null);
  const [canvasKey, setCanvasKey] = useState(0);
  const [activePanel, setActivePanel] = useState<PanelId>("collapse");
  const lowerQualityRef = useRef<(() => void) | null>(null);
  const [missionStart, setMissionStart] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  // --- Derived, entirely from the fusion table ----------------------------
  const mass = useMemo(() => deriveMass(doc), [doc]);
  const disk = useMemo(() => deriveDisk(doc), [doc]);
  const signal = useMemo(() => deriveSignal(answer, mass.pageCount), [answer, mass.pageCount]);
  const collapse = useMemo(
    () => collapseProgress(job?.stage ?? null, job?.progress ?? null, doc?.status ?? null),
    [job?.stage, job?.progress, doc?.status]
  );

  const ready = doc?.status === "ready";
  const jobStatus = job?.status;
  const jobDocumentId = job?.document_id;

  /* Mission elapsed starts when the backend reports the job succeeded, not
     when the upload began. It measures the age of a usable index, which is
     the only interval that means anything to the person using this. */
  useEffect(() => {
    if (jobStatus === "succeeded" && missionStart === null) setMissionStart(Date.now());
  }, [jobStatus, missionStart]);

  useEffect(() => {
    if (missionStart === null) return;
    setElapsed((Date.now() - missionStart) / 1000);
    const id = window.setInterval(() => setElapsed((Date.now() - missionStart) / 1000), 1000);
    return () => window.clearInterval(id);
  }, [missionStart]);

  /* On a terminal job, re-read the document so the mass and disk come from
     the record's real metrics rather than being inferred from the job. */
  useEffect(() => {
    if (!jobDocumentId) return;
    if (jobStatus !== "succeeded" && jobStatus !== "failed") return;
    api
      .getDocument(jobDocumentId)
      .then(setDoc)
      .catch(() => {
        /* the collapse panel already surfaces the job's own failure */
      });
  }, [jobStatus, jobDocumentId]);

  useEffect(() => {
    if (ready) setActivePanel((p) => (p === "collapse" ? "query" : p));
  }, [ready]);

  // --- Shader inputs -------------------------------------------------------
  const simInputs: SimInputs = useMemo(
    () => ({
      ...DEFAULT_INPUTS,
      schwarzschildRadius: mass.schwarzschildRadius,
      // In units of r_s, matching uDiskInner (3 r_s, the ISCO). Grows with
      // the chunk count so a densely-indexed document has a visibly larger
      // disk — but over a deliberately narrow range (8..12 r_s). The camera
      // distance is a fixed number of r_s, so letting this vary widely
      // makes a large document overflow the frame and a small one vanish
      // into it. The mapping stays legible; the framing stays composed.
      diskOuter:
        mass.schwarzschildRadius * (8 + Math.min(disk.particleCount ?? 0, 400) / 100),
      diskLuminosity: disk.luminosity ?? 0,
      diskIntegrity: disk.integrity ?? 1,
      particleSeed: doc ? seedFromId(doc.document_id) : 0,
      collapse: collapse ?? 0,
      grounding: groundingScalar(signal.grounding),
      abstained: signal.abstained,
      traceProgress,
      hotspots: signal.hotspots,
      redshift,
    }),
    [mass, disk, doc, collapse, signal, traceProgress, redshift]
  );

  // --- Actions -------------------------------------------------------------
  const handleFile = useCallback(async (file: File) => {
    setUploading(true);
    setUploadError(null);
    setUploadName(file.name);
    setUploadSize(file.size);
    setAnswer(null);
    setSummary(null);
    setTranslation(null);
    setPages([]);
    setMissionStart(null);
    setElapsed(null);
    try {
      const res = await api.uploadDocument(file);
      setDoc(res.document);
      setJobId(res.job_id);
      setActivePanel("collapse");
    } catch (err) {
      setUploadError(messageOf(err));
      setUploadName(null);
    } finally {
      setUploading(false);
    }
  }, []);

  const handleAsk = useCallback(
    async (question: string) => {
      if (!doc) return;
      setAsking(true);
      setAskError(null);
      setSelectedCitation(null);
      setTraceProgress(0);
      try {
        setAnswer(await api.ask(doc.document_id, question, answer?.conversation_id));
      } catch (err) {
        setAskError(messageOf(err));
      } finally {
        setAsking(false);
      }
    },
    [doc, answer?.conversation_id]
  );

  /* The geodesic trace reveal. It runs AFTER the response arrives and
     reveals the citations that actually came back — it is a reveal of real
     results, never a progress indicator standing in for the request. Under
     reduced motion it completes instantly, so nothing is gated on it. */
  useEffect(() => {
    if (!answer || answer.abstained) {
      setTraceProgress(0);
      return;
    }
    if (reducedMotion) {
      setTraceProgress(1);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min((now - start) / 1400, 1);
      setTraceProgress(t);
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [answer, reducedMotion]);

  const handleSummarize = useCallback(
    async (force: boolean) => {
      if (!doc) return;
      setSummaryPending(true);
      setSummaryError(null);
      try {
        setSummary(await api.summarize(doc.document_id, force));
      } catch (err) {
        setSummaryError(messageOf(err));
      } finally {
        setSummaryPending(false);
      }
    },
    [doc]
  );

  const handleTranslate = useCallback(
    async (target: string) => {
      if (!doc) return;
      setTranslatePending(true);
      setTranslateError(null);
      try {
        setTranslation(await api.translate(doc.document_id, target));
      } catch (err) {
        setTranslateError(messageOf(err));
      } finally {
        setTranslatePending(false);
      }
    },
    [doc]
  );

  useEffect(() => {
    if (activePanel !== "integrity" || !doc || pages.length > 0) return;
    setPagesPending(true);
    setPagesError(null);
    api
      .getDocumentPages(doc.document_id)
      .then((r) => setPages(r.pages))
      .catch((err) => setPagesError(messageOf(err)))
      .finally(() => setPagesPending(false));
  }, [activePanel, doc, pages.length]);

  const reinitialise = useCallback(() => {
    setRenderFailed(null);
    setCanvasKey((k) => k + 1);
  }, []);

  const resetMass = useCallback(() => {
    setDoc(null);
    setJobId(null);
    setUploadName(null);
    setUploadSize(null);
    setAnswer(null);
    setSummary(null);
    setTranslation(null);
    setPages([]);
    setMissionStart(null);
    setElapsed(null);
    setActivePanel("collapse");
  }, []);

  const toggle = useCallback((key: keyof Toggles) => {
    setToggles((t) => ({ ...t, [key]: !t[key] }));
  }, []);

  const onCanvasReady = useCallback((handle: { lowerQuality: () => void }) => {
    lowerQualityRef.current = handle.lowerQuality;
  }, []);

  const onUnavailable = useCallback(() => {
    setRenderFailed((prev) => prev ?? "unsupported");
  }, []);

  // --- Keyboard: full operation, no exceptions ----------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      // Never hijack a key the user is typing into a field.
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) {
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const k = e.key.toLowerCase();
      if (k >= "1" && k <= "4") setPreset(PRESET_ORDER[Number(k) - 1]);
      else if (k === "c") {
        if (reducedMotion) return;
        toggle("cinematic");
      } else if (k === "r") {
        if (reducedMotion) return;
        toggle("auto");
      } else if (k === "p") toggle("params");
      else if (k === "m") toggle("sound");
      else if (k === "h") toggle("hud");
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, reducedMotion]);

  /* CINEMATIC SEQUENCE: a scripted walk through the four presets. Disabled
     outright under reduced motion — not slowed, disabled. */
  useEffect(() => {
    if (!toggles.cinematic || reducedMotion) return;
    let i = PRESET_ORDER.indexOf(preset);
    const id = window.setInterval(() => {
      i = (i + 1) % PRESET_ORDER.length;
      setPreset(PRESET_ORDER[i]);
    }, 7000);
    return () => window.clearInterval(id);
    // `preset` is read once to pick a starting point; including it would
    // restart the sequence on every step it takes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toggles.cinematic, reducedMotion]);

  const groundingTone = groundingColor(signal.grounding);

  return (
    <main className="relative h-[100dvh] w-full overflow-hidden bg-[color:var(--color-void)]">
      {renderFailed ? (
        <StaticFallback
          grounding={signal.grounding}
          abstained={signal.abstained}
          luminosity={disk.luminosity}
          hotspots={signal.hotspots}
          reason={renderFailed}
          onReinitialise={reinitialise}
        />
      ) : (
        <Suspense fallback={<LoadingState />}>
          <GargantuaCanvas
            key={canvasKey}
            inputs={simInputs}
            preset={preset}
            autoOrbit={toggles.auto}
            onTelemetry={setTelemetry}
            onUnavailable={onUnavailable}
            onReady={onCanvasReady}
          />
        </Suspense>
      )}

      {/* Viewport framing: two L-brackets, as in the reference. */}
      <div aria-hidden="true" className="bracket bracket--tl pointer-events-none absolute inset-4" />
      <div aria-hidden="true" className="bracket bracket--br pointer-events-none absolute inset-4" />

      {toggles.hud && (
        <>
          <TitleBlock
            documentName={doc?.filename ?? uploadName}
            massLabel={mass.massLabel}
            metricLine="Schwarzschild metric // null-geodesic retrieval"
            elapsedSeconds={elapsed}
          />

          {/* Bottom-left telemetry. Every row is a measured value or an
              em-dash; there is no third option anywhere in this block. */}
          <div className="pointer-events-none absolute bottom-6 left-6 select-none sm:bottom-8 sm:left-8">
            <TelemetryRow
              label="Observer distance"
              value={telemetry ? fmt(telemetry.observerDistance, " RS", 2) : null}
            />
            <TelemetryRow
              label="Disk inclination"
              value={telemetry ? fmt(telemetry.diskInclination, "°", 1) : null}
            />
            <TelemetryRow
              label="Geodesic steps"
              value={telemetry ? String(telemetry.geodesicSteps) : null}
            />
            <TelemetryRow label="Render profile" value={telemetry?.profile ?? null} />
            <TelemetryRow
              label="Frame rate"
              value={
                telemetry?.frameRate != null ? `${Math.round(telemetry.frameRate)} FPS` : null
              }
            />
            <div className="hairline my-2 w-[240px]" />
            <TelemetryRow
              label="Disk particles"
              value={disk.particleCount !== null ? `~${disk.particleCount}` : null}
            />
            <TelemetryRow
              label="Signal strength"
              value={signal.grounding ? signal.grounding.toUpperCase() : null}
              tone={groundingTone}
            />
            <TelemetryRow label="Inference backend" value={signal.modelUsed} />
          </div>

          {/* Placed before the camera controls in DOM order: keyboard
              focus should reach the document work before the chrome. */}
          <section
            aria-label="Document intelligence"
            /* Bottom bound leaves room for the telemetry block in the lower
               left; without it the panel runs over the readouts. */
            className="panel pointer-events-auto absolute bottom-[264px] left-6 top-[172px] flex w-[min(420px,calc(100vw-3rem))] flex-col sm:left-8"
          >
            <div
              role="tablist"
              aria-label="Capabilities"
              className="flex shrink-0 border-b border-[color:var(--color-chrome-rule)]"
            >
              {PANELS.map((p) => (
                <button
                  key={p.id}
                  role="tab"
                  type="button"
                  aria-selected={activePanel === p.id}
                  aria-controls={`panel-${p.id}`}
                  id={`tab-${p.id}`}
                  disabled={p.id !== "collapse" && !ready}
                  data-active={activePanel === p.id}
                  className="control flex-1"
                  style={{
                    border: 0,
                    borderBottom: "1px solid",
                    borderBottomColor:
                      activePanel === p.id ? "var(--color-accent)" : "transparent",
                    borderRadius: 0,
                  }}
                  onClick={() => setActivePanel(p.id)}
                >
                  {p.label}
                </button>
              ))}
            </div>

            <div
              id={`panel-${activePanel}`}
              role="tabpanel"
              aria-labelledby={`tab-${activePanel}`}
              tabIndex={0}
              className="min-h-0 flex-1 overflow-y-auto p-4"
            >
              {activePanel === "collapse" && (
                <CollapsePanel
                  job={job}
                  fileName={uploadName}
                  fileSize={uploadSize}
                  pollError={pollError ?? uploadError}
                  onFile={handleFile}
                  onReinitialise={resetMass}
                  busy={uploading}
                />
              )}
              {activePanel === "query" && (
                <QueryPanel
                  answer={answer}
                  pending={asking}
                  error={askError}
                  onAsk={handleAsk}
                  onSelectCitation={setSelectedCitation}
                  selectedChunkId={selectedCitation?.chunk_id ?? null}
                />
              )}
              {activePanel === "briefing" && (
                <BriefingPanel
                  summary={summary}
                  pending={summaryPending}
                  error={summaryError}
                  onGenerate={handleSummarize}
                />
              )}
              {activePanel === "frequency" && (
                <FrequencyPanel
                  result={translation}
                  pending={translatePending}
                  error={translateError}
                  onTranslate={handleTranslate}
                  onShiftPreview={setRedshift}
                />
              )}
              {activePanel === "integrity" && (
                <IntegrityPanel
                  pages={pages}
                  pending={pagesPending}
                  error={pagesError}
                  highlightPage={selectedCitation?.page_number ?? null}
                />
              )}
            </div>
          </section>

          <NavigationPanel
            preset={preset}
            toggles={toggles}
            onPreset={setPreset}
            onToggle={toggle}
            onCinematic={() => toggle("cinematic")}
            reducedMotion={reducedMotion}
          />

          {toggles.params && (
            <ParamsPanel
              onLowerQuality={() => lowerQualityRef.current?.()}
              onReset={() => {
                setPreset("poster");
                setToggles({ auto: true, cinematic: false, params: true, hud: true, sound: false });
                setRedshift(0);
              }}
              onResetMass={resetMass}
              hasDocument={doc !== null}
              telemetry={telemetry}
              diskParticles={disk.particleCount}
              lowQualityPages={disk.lowQualityPages}
              ocrPages={disk.ocrPages}
            />
          )}

        </>
      )}

      {/* The parallel text truth. Everything the canvas depicts, in the DOM,
          from the same source values — not a description of the picture. */}
      <div className="sr-only" aria-live="polite">
        <h2>Simulation state</h2>
        <p>
          Mass: {doc?.filename ?? "no document loaded"}.{" "}
          {mass.wordCount !== null
            ? `${mass.wordCount} words across ${mass.pageCount ?? UNSET} pages.`
            : "Document metrics not yet available."}
        </p>
        <p>
          Accretion disk: {disk.particleCount ?? "unknown"} indexed chunks.
          {disk.lowQualityPages > 0
            ? ` ${disk.lowQualityPages} pages flagged low quality.`
            : ""}
        </p>
        <p>
          {signal.grounding
            ? `Signal: ${signal.grounding} grounding at relevance ${signal.relevance}. ${signal.survived} of ${signal.launched} retrieved candidates survived reranking.`
            : "No query has been traced."}
        </p>
        {signal.abstained && (
          <p>
            No grounded signal. The document does not support this query and no answer was
            generated.
          </p>
        )}
      </div>
    </main>
  );
}

/* ------------------------------------------------------------------------ */

function ParamsPanel({
  onLowerQuality,
  onReset,
  onResetMass,
  hasDocument,
  telemetry,
  diskParticles,
  lowQualityPages,
  ocrPages,
}: {
  onLowerQuality: () => void;
  onReset: () => void;
  onResetMass: () => void;
  hasDocument: boolean;
  telemetry: Telemetry | null;
  diskParticles: number | null;
  lowQualityPages: number;
  ocrPages: number;
}) {
  return (
    <aside
      aria-label="Parameters"
      className="panel pointer-events-auto absolute bottom-6 right-6 w-[260px] p-3 sm:bottom-[268px] sm:right-8"
    >
      <div className="mb-3 flex items-center justify-between">
        <h2 className="u-label m-0">Parameters</h2>
        <button type="button" className="control control--chip" onClick={onReset}>
          Reset
        </button>
      </div>

      <TelemetryRow label="Render profile" value={telemetry?.profile ?? null} />
      <TelemetryRow
        label="Indexed chunks"
        value={diskParticles !== null ? `~${diskParticles}` : null}
      />
      <TelemetryRow label="OCR pages" value={hasDocument ? String(ocrPages) : null} />
      <TelemetryRow label="Low quality" value={hasDocument ? String(lowQualityPages) : null} />

      <div className="hairline my-3" />

      <button type="button" className="control mb-2 w-full" onClick={onLowerQuality}>
        Lower quality
      </button>
      <button type="button" className="control w-full" onClick={onResetMass} disabled={!hasDocument}>
        Eject mass
      </button>
    </aside>
  );
}

function LoadingState() {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center px-6 text-center">
      <p className="u-legend m-0 mb-4">Real-time relativistic raytracing</p>
      <p
        className="m-0 font-mono font-extralight text-[color:var(--color-chrome-max)]"
        style={{ fontSize: "var(--text-wordmark)", letterSpacing: "var(--tracking-wordmark)" }}
      >
        GARGANTUA
      </p>
      <p className="prose-readout mt-8 text-center opacity-70">
        Every value on this screen is measured. Where a value is unknown, it reads as an
        em-dash.
      </p>
    </div>
  );
}

function messageOf(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Something went wrong. Please try again.";
}
