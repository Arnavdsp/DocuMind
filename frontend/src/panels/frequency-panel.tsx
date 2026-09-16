import { useState } from "react";
import { Section } from "../hud/telemetry";
import type { TranslateResponse } from "../types/api";

/**
 * Translate, as a frequency shift.
 *
 * The conceit: changing the output language shifts the disk's emitted
 * spectrum. Moving the control drives the shader's uRedshift uniform in
 * real time, so the render responds before the request completes and the
 * control feels connected to the world rather than pasted onto it.
 *
 * The honesty constraint bites here and wins: the shift amount is a
 * PRESENTATION parameter, not a measurement, so it is never displayed as a
 * telemetry value with units. It is a slider position and it is labelled
 * as one. The only measured facts on this panel are the languages the
 * backend reported and whether it truncated the text.
 */
const LANGUAGES: Array<{ code: string; label: string; shift: number }> = [
  { code: "es", label: "Spanish", shift: -0.8 },
  { code: "fr", label: "French", shift: -0.5 },
  { code: "de", label: "German", shift: -0.2 },
  { code: "hi", label: "Hindi", shift: 0.2 },
  { code: "ja", label: "Japanese", shift: 0.5 },
  { code: "ar", label: "Arabic", shift: 0.8 },
];

export function FrequencyPanel({
  result,
  pending,
  error,
  onTranslate,
  onShiftPreview,
}: {
  result: TranslateResponse | null;
  pending: boolean;
  error: string | null;
  onTranslate: (target: string) => void;
  onShiftPreview: (shift: number) => void;
}) {
  const [target, setTarget] = useState<string | null>(null);

  const select = (code: string, shift: number) => {
    setTarget(code);
    onShiftPreview(shift);
  };

  return (
    <div>
      <Section label="Frequency shift">
        <p className="prose-readout mb-3">
          Shift the emitted spectrum. Blueshift toward the left of the band,
          redshift toward the right.
        </p>

        <div
          role="radiogroup"
          aria-label="Target language"
          className="mb-3 grid grid-cols-3 gap-2"
        >
          {LANGUAGES.map((l) => (
            <button
              key={l.code}
              type="button"
              role="radio"
              aria-checked={target === l.code}
              data-active={target === l.code}
              className="control"
              onClick={() => select(l.code, l.shift)}
            >
              {l.label}
            </button>
          ))}
        </div>

        <button
          type="button"
          className="control w-full"
          disabled={!target || pending}
          onClick={() => target && onTranslate(target)}
        >
          {pending ? "Shifting" : "Apply shift"}
        </button>

        {error && (
          <p role="alert" className="prose-readout mt-3 text-[color:var(--color-accent-bright)]">
            {error}
          </p>
        )}
      </Section>

      {result && (
        <Section label="Shifted transmission">
          <p className="u-legend mb-3">
            {result.source_language} → {result.target_language} · provider{" "}
            {result.provider}
          </p>
          {/* Truncation is a data-integrity fact and gets stated, not
              hidden behind a fade-out or a "read more". */}
          {result.truncated && (
            <p
              className="prose-readout mb-3 border-l-2 pl-3"
              style={{ borderColor: "var(--color-accent)" }}
            >
              Output was truncated by the translation provider. What follows
              is not the complete document.
            </p>
          )}
          <p className="prose-readout m-0 max-h-[40vh] overflow-y-auto whitespace-pre-wrap">
            {result.translated_text}
          </p>
        </Section>
      )}
    </div>
  );
}
