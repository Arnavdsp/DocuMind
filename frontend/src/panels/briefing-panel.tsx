import { Section } from "../hud/telemetry";
import type { SummarizeResponse } from "../types/api";

/**
 * Summarize, as a mission briefing.
 *
 * `important_numbers` are given the typographic weight the brief asks for:
 * they are the only body-copy element in the interface set at readout size
 * with the value-hot ink, laid out as a grid of instrument readings rather
 * than a bulleted list. Everything else in the briefing is prose and reads
 * as prose.
 */
export function BriefingPanel({
  summary,
  pending,
  error,
  onGenerate,
}: {
  summary: SummarizeResponse | null;
  pending: boolean;
  error: string | null;
  onGenerate: (force: boolean) => void;
}) {
  if (!summary && !pending) {
    return (
      <Section label="Mission briefing">
        <p className="prose-readout mb-3">
          No briefing has been compiled for this mass.
        </p>
        <button type="button" className="control" onClick={() => onGenerate(false)}>
          Compile briefing
        </button>
        {error && (
          <p role="alert" className="prose-readout mt-3 text-[color:var(--color-accent-bright)]">
            {error}
          </p>
        )}
      </Section>
    );
  }

  if (pending) {
    return (
      <Section label="Mission briefing">
        <p role="status" className="u-label">
          Compiling briefing
        </p>
      </Section>
    );
  }

  if (!summary) return null;
  const s = summary.summary;

  return (
    <div>
      <Section
        label="Mission briefing"
        action={
          <button
            type="button"
            className="control control--chip"
            onClick={() => onGenerate(true)}
          >
            Recompile
          </button>
        }
      >
        <p className="prose-readout m-0">{s.executive_summary}</p>
        <p className="u-legend mt-3">
          Strategy {summary.strategy} · backend {summary.model_used}
          {summary.cached ? " · served from cache" : ""}
        </p>
      </Section>

      {s.important_numbers.length > 0 && (
        <Section label={`Instrument readings · ${s.important_numbers.length}`}>
          <ul className="m-0 grid list-none grid-cols-1 gap-px bg-[color:var(--color-chrome-rule)] p-0 sm:grid-cols-2">
            {s.important_numbers.map((n, i) => (
              <li
                key={`${n}-${i}`}
                className="bg-[color:var(--color-void)] px-3 py-3"
              >
                <span
                  className="block font-mono tabular-nums"
                  style={{
                    fontSize: "var(--text-body)",
                    color: "var(--color-chrome-hot)",
                    letterSpacing: "0.02em",
                  }}
                >
                  {n}
                </span>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {s.key_findings.length > 0 && (
        <Section label={`Key findings · ${s.key_findings.length}`}>
          <ol className="m-0 list-none p-0">
            {s.key_findings.map((f, i) => (
              <li key={i} className="mb-3 flex gap-3">
                <span className="u-label shrink-0 tabular-nums">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="prose-readout m-0">{f}</span>
              </li>
            ))}
          </ol>
        </Section>
      )}

      {s.methodology && (
        <Section label="Methodology">
          <p className="prose-readout m-0">{s.methodology}</p>
        </Section>
      )}

      {/* Limitations are surfaced, never collapsed behind a disclosure.
          A summary's stated limits are part of the summary. */}
      {s.limitations && (
        <Section label="Limitations">
          <p
            className="prose-readout m-0 border-l-2 pl-3"
            style={{ borderColor: "var(--color-accent)" }}
          >
            {s.limitations}
          </p>
        </Section>
      )}
    </div>
  );
}
