import { groundingColor } from "../sim/mapping";
import type { GroundingLevel } from "../types/api";

/**
 * The non-WebGL path.
 *
 * This is not a "sorry, your browser is unsupported" dead end. Every panel
 * in this application is DOM — the canvas only ever depicted state that is
 * simultaneously rendered as text — so losing WebGL costs the render and
 * nothing else. Upload, ingestion progress, asking, citations, summary,
 * translation and source pages all keep working.
 *
 * What replaces the canvas is a CSS-drawn instrument: concentric rings
 * whose colour tracks the same grounding level the shader would have used,
 * a dark core, and the citation hot spots as positioned markers. It is
 * schematic rather than cinematic, and it is honest about being so.
 */
export function StaticFallback({
  grounding,
  abstained,
  luminosity,
  hotspots,
  reason,
  onReinitialise,
}: {
  grounding: GroundingLevel | null;
  abstained: boolean;
  luminosity: number | null;
  hotspots: Array<{ chunkId: string; theta: number; intensity: number }>;
  reason: "unsupported" | "context-lost";
  onReinitialise: () => void;
}) {
  const tone = groundingColor(grounding) ?? "var(--color-chrome-rule)";
  const lit = abstained ? 0.06 : (luminosity ?? 0.25);

  return (
    <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
      <div
        aria-hidden="true"
        className="relative h-[min(58vh,520px)] w-[min(58vh,520px)]"
      >
        {/* Accretion rings. Opacity carries index density; colour carries
            grounding and nothing else. */}
        {[0.55, 0.72, 0.88, 1].map((scale, i) => (
          <div
            key={scale}
            className="absolute left-1/2 top-1/2 rounded-full border"
            style={{
              width: `${scale * 100}%`,
              height: `${scale * 100}%`,
              transform: "translate(-50%, -50%) rotateX(72deg)",
              borderColor: tone,
              opacity: lit * (0.9 - i * 0.18),
            }}
          />
        ))}

        {/* The shadow. Genuinely black — the one part of the composition
            that is meant to be an absence. */}
        <div
          className="absolute left-1/2 top-1/2 h-[38%] w-[38%] -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{ background: "#000", boxShadow: `0 0 60px 8px ${tone}22` }}
        />

        {/* Citation hot spots, positioned by page number exactly as the
            shader positions them. */}
        {hotspots.map((h) => {
          const angle = h.theta * Math.PI * 2;
          const rx = 46;
          const ry = 14;
          return (
            <span
              key={h.chunkId}
              className="absolute h-[6px] w-[6px] rounded-full"
              style={{
                left: `calc(50% + ${Math.cos(angle) * rx}% - 3px)`,
                top: `calc(50% + ${Math.sin(angle) * ry}% - 3px)`,
                background: tone,
                opacity: 0.4 + h.intensity * 0.6,
                boxShadow: `0 0 8px ${tone}`,
              }}
            />
          );
        })}
      </div>

      <div className="panel absolute bottom-[16%] left-1/2 max-w-[380px] -translate-x-1/2 p-4 text-center">
        <p className="u-label m-0 mb-2">
          {reason === "unsupported" ? "Renderer unavailable" : "Signal lost"}
        </p>
        <p className="prose-readout m-0 mb-3 text-center">
          {reason === "unsupported"
            ? "This browser or device could not start WebGL. The schematic above stands in for the render; every reading, answer and citation remains available."
            : "The graphics context was lost — usually the GPU recovering from another process. Nothing was lost from the document or the index."}
        </p>
        <button type="button" className="control mx-auto" onClick={onReinitialise}>
          Reinitialise
        </button>
      </div>
    </div>
  );
}
