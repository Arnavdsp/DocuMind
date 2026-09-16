import { useCallback, useRef, useState } from "react";
import { Section } from "../hud/telemetry";
import { COLLAPSE_SEQUENCE, STAGE_CAPTION, fmtBytes } from "../sim/mapping";
import type { JobRecord, ProcessingStage } from "../types/api";

/**
 * Ingestion is the collapse sequence.
 *
 * The nine ProcessingStage values are the script. Each row lights as the
 * backend reports reaching it — and only then. There is no timer filling
 * in the gap between polls, no easing toward the next stage, no
 * "almost done" at 95% while a job hangs. If the backend is stuck on
 * EMBEDDING for two minutes, this panel sits on EMBEDDING for two minutes.
 *
 * That restraint is the point. The previous version of this product
 * displayed progress the backend had not reported, and the whole design
 * ethos of the rebuild is that a number on screen is a measured number.
 */
export function CollapsePanel({
  job,
  fileName,
  fileSize,
  pollError,
  onFile,
  onReinitialise,
  busy,
}: {
  job: JobRecord | null;
  fileName: string | null;
  fileSize: number | null;
  pollError: string | null;
  onFile: (file: File) => void;
  onReinitialise: () => void;
  busy: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  const failed = job?.status === "failed" || job?.stage === "failed";
  const currentIndex = job ? COLLAPSE_SEQUENCE.indexOf(job.stage) : -1;

  // --- No mass loaded: the upload path ------------------------------------
  if (!job && !busy) {
    return (
      <Section label="Infalling mass">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          className="border border-dashed p-6 text-center transition-colors"
          style={{
            borderColor: dragOver
              ? "var(--color-chrome-mid)"
              : "var(--color-chrome-rule)",
          }}
        >
          <p className="prose-readout m-0 mb-4 text-center">
            Drop a document to collapse it into a singularity. PDF, plain text
            or a scanned image, up to 25 MB.
          </p>
          <button
            type="button"
            className="control mx-auto"
            onClick={() => inputRef.current?.click()}
          >
            Select document
          </button>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.txt,.jpg,.jpeg,.png"
            className="sr-only"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      </Section>
    );
  }

  return (
    <div>
      <Section label={failed ? "Signal lost" : "Collapse sequence"}>
        {fileName && (
          <p className="u-legend mb-3">
            {fileName}
            {fileSize !== null ? ` · ${fmtBytes(fileSize)}` : ""}
          </p>
        )}

        {/* The stage ladder. aria-live so a screen reader hears each real
            transition once, rather than every 900ms poll. */}
        <ol
          className="m-0 list-none p-0"
          aria-live="polite"
          aria-label="Ingestion progress"
        >
          {COLLAPSE_SEQUENCE.map((stage, i) => {
            const state: StageState = failed
              ? i < currentIndex
                ? "done"
                : "idle"
              : i < currentIndex
                ? "done"
                : i === currentIndex
                  ? "active"
                  : "idle";
            return <StageRow key={stage} stage={stage} state={state} />;
          })}
        </ol>

        {job && !failed && (
          <p className="u-legend mt-3">
            Backend reported {Math.round(job.progress * 100)}% at stage{" "}
            {job.stage}.
          </p>
        )}
      </Section>

      {failed && (
        <Section label="Fault">
          <p className="prose-readout m-0 mb-3">
            {job?.error_message ??
              "Ingestion failed before the index was built."}
          </p>
          <button type="button" className="control" onClick={onReinitialise}>
            Reinitialise
          </button>
        </Section>
      )}

      {pollError && (
        <p role="alert" className="prose-readout text-[color:var(--color-accent-bright)]">
          {pollError}
        </p>
      )}
    </div>
  );
}

type StageState = "idle" | "active" | "done";

function StageRow({ stage, state }: { stage: ProcessingStage; state: StageState }) {
  const color =
    state === "active"
      ? "var(--color-accent-bright)"
      : state === "done"
        ? "var(--color-chrome-hot)"
        : "var(--color-chrome-dim)";

  return (
    <li className="flex items-baseline gap-3 py-[3px]">
      <span
        aria-hidden="true"
        className="w-3 shrink-0 text-center"
        style={{ color }}
      >
        {state === "done" ? "•" : state === "active" ? "›" : " "}
      </span>
      <span
        className="u-label w-[104px] shrink-0"
        style={{ color }}
      >
        {stage}
      </span>
      <span className="u-legend" style={{ opacity: state === "idle" ? 0.4 : 1 }}>
        {STAGE_CAPTION[stage]}
      </span>
      <span className="sr-only">
        {state === "active"
          ? " — in progress"
          : state === "done"
            ? " — complete"
            : " — not started"}
      </span>
    </li>
  );
}
