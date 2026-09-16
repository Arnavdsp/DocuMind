import { fmtClock, UNSET } from "../sim/mapping";

/**
 * Top-left identity block and top-right mission clock.
 *
 * In the reference the second line names the object's mass. Here it names
 * the document's — the document IS the mass — and it renders an em-dash
 * until ingestion has produced real metrics. The wordmark is the only
 * pure-white type and the only type above 22px in the interface.
 */
export function TitleBlock({
  documentName,
  massLabel,
  metricLine,
  elapsedSeconds,
}: {
  documentName: string | null;
  massLabel: string | null;
  metricLine: string;
  /** Seconds since ingestion completed. Null before the job succeeds — the
   *  clock reads 00:00:00 and is explicitly labelled as not yet started. */
  elapsedSeconds: number | null;
}) {
  return (
    <>
      <div className="pointer-events-none absolute left-6 top-6 select-none sm:left-8 sm:top-8">
        <h1
          className="m-0 font-mono font-extralight leading-none text-[color:var(--color-chrome-max)]"
          style={{
            fontSize: "var(--text-wordmark)",
            letterSpacing: "var(--tracking-wordmark)",
          }}
        >
          GARGANTUA
        </h1>
        <p
          className="m-0 mt-[10px] uppercase"
          style={{
            fontSize: "var(--text-value)",
            letterSpacing: "var(--tracking-mid)",
            color: "var(--color-accent-bright)",
          }}
        >
          {documentName ? truncate(documentName, 34) : "NO MASS LOADED"}
          {massLabel ? ` · ${massLabel}` : ""}
        </p>
        <p
          className="u-legend m-0 mt-[5px]"
          style={{ letterSpacing: "var(--tracking-tight)" }}
        >
          {metricLine}
        </p>
      </div>

      <div className="pointer-events-none absolute right-6 top-6 select-none text-right sm:right-8 sm:top-8">
        <p className="u-label m-0">Mission elapsed</p>
        <p
          className="m-0 mt-1 font-mono tabular-nums leading-none"
          style={{
            fontSize: "var(--text-readout)",
            letterSpacing: "0.1em",
            color:
              elapsedSeconds === null
                ? "var(--color-chrome-dim)"
                : "var(--color-chrome-hot)",
          }}
        >
          {fmtClock(elapsedSeconds)}
        </p>
        <p className="sr-only">
          {elapsedSeconds === null
            ? "Mission clock has not started; ingestion is not complete."
            : `Time since indexing completed: ${fmtClock(elapsedSeconds)}.`}
        </p>
      </div>
    </>
  );
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`;
}

export { UNSET };
