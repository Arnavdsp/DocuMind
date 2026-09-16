import { Section } from "../hud/telemetry";
import { fmt } from "../sim/mapping";
import type { DocumentPage, ExtractionMethod } from "../types/api";

/**
 * Source pages, as a data-integrity audit.
 *
 * The reference has no equivalent panel, so this one is designed to the
 * same language rather than copied: each page is an instrument channel
 * with its extraction method, OCR confidence and quality flag stated on
 * the row, and the extracted text below it.
 *
 * The rule this panel exists to enforce: a page we could barely read must
 * LOOK like a page we could barely read. Nothing here is smoothed over.
 * `is_low_quality` gets a visible marker, `ocr_confidence` is printed as
 * the raw number the extractor produced, and a page with no extractable
 * text says so rather than rendering as an innocuous empty block.
 */
const METHOD_LABEL: Record<ExtractionMethod, string> = {
  native_text: "Native text",
  ocr: "OCR",
  mixed: "Mixed",
  plain_text: "Plain text",
};

export function IntegrityPanel({
  pages,
  pending,
  error,
  highlightPage,
}: {
  pages: DocumentPage[];
  pending: boolean;
  error: string | null;
  /** Page number of the currently selected citation, if any. */
  highlightPage: number | null;
}) {
  if (pending) {
    return (
      <Section label="Source integrity">
        <p role="status" className="u-label">
          Reading extracted pages
        </p>
      </Section>
    );
  }

  if (error) {
    return (
      <Section label="Source integrity">
        <p role="alert" className="prose-readout text-[color:var(--color-accent-bright)]">
          {error}
        </p>
      </Section>
    );
  }

  const lowQuality = pages.filter((p) => p.is_low_quality);
  const ocrPages = pages.filter(
    (p) => p.extraction_method === "ocr" || p.extraction_method === "mixed"
  );

  return (
    <div>
      <Section label={`Source integrity · ${pages.length} pages`}>
        {lowQuality.length > 0 ? (
          <p
            className="prose-readout m-0 mb-3 border-l-2 pl-3"
            style={{ borderColor: "var(--color-accent)" }}
          >
            {lowQuality.length} of {pages.length} pages extracted poorly (pages{" "}
            {lowQuality.map((p) => p.page_number).join(", ")}). Answers drawn
            from these pages rest on unreliable text.
          </p>
        ) : (
          <p className="prose-readout m-0 mb-3">
            All {pages.length} pages extracted cleanly.
          </p>
        )}
        {ocrPages.length > 0 && (
          <p className="u-legend">
            {ocrPages.length} page{ocrPages.length === 1 ? "" : "s"} required
            optical reconstruction.
          </p>
        )}
      </Section>

      <ol className="m-0 list-none p-0">
        {pages.map((p) => (
          <li
            key={p.page_number}
            id={`page-${p.page_number}`}
            className="mb-4 border-l-2 pl-3"
            style={{
              borderColor:
                highlightPage === p.page_number
                  ? "var(--color-chrome-mid)"
                  : p.is_low_quality
                    ? "var(--color-accent)"
                    : "var(--color-chrome-rule)",
            }}
          >
            <div className="mb-2 flex flex-wrap items-baseline gap-x-4 gap-y-1">
              <span className="u-label">Page {p.page_number}</span>
              <span className="u-legend">{METHOD_LABEL[p.extraction_method]}</span>
              <span className="u-legend">
                OCR confidence {fmt(p.ocr_confidence, "", 2)}
              </span>
              {p.is_low_quality && (
                <span
                  className="u-legend"
                  style={{ color: "var(--color-accent-bright)" }}
                >
                  Low quality
                </span>
              )}
            </div>
            {p.text.trim().length > 0 ? (
              <p className="prose-readout m-0 max-h-[26vh] overflow-y-auto whitespace-pre-wrap">
                {p.text}
              </p>
            ) : (
              <p className="prose-readout m-0 text-[color:var(--color-chrome-dim)]">
                No text could be extracted from this page.
              </p>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
