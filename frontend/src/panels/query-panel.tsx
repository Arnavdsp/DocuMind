import { useState } from "react";
import { Section, TelemetryRow } from "../hud/telemetry";
import { fmt, groundingColor, RERANK_TOP_K, RETRIEVAL_TOP_K } from "../sim/mapping";
import type { AskResponse, Citation } from "../types/api";

/**
 * Asking is a geodesic trace, and this panel is its readout.
 *
 * The answer renders as decoded telemetry rather than a chat bubble: a
 * monospace transmission block under a hairline, with the retrieval
 * accounting stated plainly above it. Every number here is a value the
 * backend returned. There is no confidence percentage, because the backend
 * does not produce one that means anything — the previous version of this
 * product displayed an uncalibrated start/end logit as "confidence: NN%"
 * and that is the specific defect this build exists to not reintroduce.
 */
export function QueryPanel({
  answer,
  pending,
  error,
  onAsk,
  onSelectCitation,
  selectedChunkId,
}: {
  answer: AskResponse | null;
  pending: boolean;
  error: string | null;
  onAsk: (question: string) => void;
  onSelectCitation: (c: Citation | null) => void;
  selectedChunkId: string | null;
}) {
  const [question, setQuestion] = useState("");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (q.length > 0 && !pending) onAsk(q);
  };

  const tone = groundingColor(answer?.grounding ?? null);

  return (
    <div>
      <Section label="Query">
        <form onSubmit={submit}>
          <label htmlFor="question" className="sr-only">
            Ask a question about this document
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
            }}
            rows={3}
            placeholder="Ask the mass a question."
            className="prose-readout w-full resize-none border border-[color:var(--color-chrome-rule)] bg-transparent p-3 outline-none placeholder:text-[color:var(--color-chrome-dim)] focus-visible:border-[color:var(--color-chrome-mid)]"
          />
          <div className="mt-2 flex items-center justify-between gap-3">
            <span className="u-legend">⌘↵ to launch</span>
            <button type="submit" className="control" disabled={pending || !question.trim()}>
              {pending ? "Tracing" : "Launch geodesics"}
            </button>
          </div>
        </form>
      </Section>

      {error && (
        <p role="alert" className="prose-readout mb-4 text-[color:var(--color-accent-bright)]">
          {error}
        </p>
      )}

      {pending && (
        <p role="status" className="u-label mb-4">
          Tracing null geodesics — {RETRIEVAL_TOP_K} candidates launched, top{" "}
          {RERANK_TOP_K} reranked.
        </p>
      )}

      {answer && !pending && (
        <>
          <Section label="Signal">
            <TelemetryRow
              label="Grounding"
              value={answer.grounding.toUpperCase()}
              tone={tone}
            />
            <TelemetryRow
              label="Relevance"
              value={fmt(answer.relevance_score, "", 3)}
              tone={tone}
            />
            <TelemetryRow
              label="Rays launched"
              value={String(RETRIEVAL_TOP_K)}
            />
            <TelemetryRow
              label="Struck disk"
              value={String(answer.citations.length)}
            />
            <TelemetryRow
              label="Past horizon"
              value={String(Math.max(RETRIEVAL_TOP_K - answer.citations.length, 0))}
            />
            <TelemetryRow label="Inference backend" value={answer.model_used} />
            {answer.model_used === "mock" && (
              <p className="u-legend mt-2 text-[color:var(--color-accent)]">
                Deterministic mock backend — answers are extractive stand-ins,
                not model generations.
              </p>
            )}
          </Section>

          {answer.abstained ? (
            <Abstention />
          ) : (
            <>
              <Section label="Transmission">
                <p className="prose-readout m-0 whitespace-pre-wrap">{answer.answer}</p>
              </Section>

              <Section label={`Lensed hot spots · ${answer.citations.length}`}>
                <ul className="m-0 list-none p-0">
                  {answer.citations.map((c) => (
                    <CitationRow
                      key={c.chunk_id}
                      citation={c}
                      selected={selectedChunkId === c.chunk_id}
                      onSelect={() =>
                        onSelectCitation(selectedChunkId === c.chunk_id ? null : c)
                      }
                    />
                  ))}
                </ul>
              </Section>
            </>
          )}
        </>
      )}
    </div>
  );
}

/**
 * Abstention is a first-class state, not an error.
 *
 * It gets the same visual weight as a successful answer — a composed
 * readout, not a red banner — because the system refusing to answer from
 * evidence it does not have is the system working. The wording says what
 * happened and why, and does not apologise for it.
 */
function Abstention() {
  return (
    <Section label="Signal">
      <div
        className="border-l-2 py-1 pl-3"
        style={{ borderColor: "var(--color-grounding-none)" }}
      >
        <p
          className="u-value m-0 mb-2"
          style={{ color: "var(--color-accent-bright)" }}
        >
          No grounded signal
        </p>
        <p className="prose-readout m-0">
          The document does not support this query. Every retrieved candidate
          fell below the relevance floor, so nothing was generated — an
          unsupported answer would be worse than none.
        </p>
      </div>
    </Section>
  );
}

function CitationRow({
  citation,
  selected,
  onSelect,
}: {
  citation: Citation;
  selected: boolean;
  onSelect: () => void;
}) {
  const label =
    citation.page_number !== null ? `Page ${citation.page_number}` : "Unpaged";
  return (
    <li className="mb-2">
      <button
        type="button"
        onClick={onSelect}
        aria-expanded={selected}
        className="w-full border border-[color:var(--color-chrome-rule)] bg-transparent p-3 text-left hover:border-[color:var(--color-chrome-dim)]"
      >
        <span className="flex items-baseline justify-between gap-3">
          <span className="u-label">
            {label}
            {citation.section ? ` · ${citation.section}` : ""}
          </span>
          <span className="u-value tabular-nums">
            {citation.relevance_score.toFixed(3)}
          </span>
        </span>
        {selected && (
          <span className="prose-readout mt-3 block whitespace-pre-wrap">
            {citation.snippet}
          </span>
        )}
      </button>
    </li>
  );
}
