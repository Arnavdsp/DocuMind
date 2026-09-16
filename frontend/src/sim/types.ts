/* ============================================================================
   Types and constants shared between the shell and the renderer.
   ----------------------------------------------------------------------------
   These live in their own module for one specific reason: App.tsx needs the
   SimInputs shape and the default values, but must NOT pull in
   renderer.ts — which imports the GLSL source. Importing a type from the
   renderer would drag ~14 KB of shader text and the WebGL host into the
   entry chunk and quietly defeat the React.lazy boundary around the scene.

   Verified by checking that `traceGeodesic` appears only in the
   gargantua-canvas chunk after a production build.
   ========================================================================= */

import type { Hotspot } from "./mapping";
import type { QualityTier } from "./quality";

export interface SimInputs {
  schwarzschildRadius: number;
  diskOuter: number;
  diskLuminosity: number;
  diskIntegrity: number;
  particleSeed: number;
  collapse: number;
  grounding: number;
  abstained: boolean;
  traceProgress: number;
  hotspots: Hotspot[];
  redshift: number;
}

export interface Telemetry {
  observerDistance: number;
  diskInclination: number;
  geodesicSteps: number;
  profile: QualityTier;
  /** Null until the first 1-second measurement window closes, so the HUD
   *  shows an em-dash on startup rather than a fabricated 60. */
  frameRate: number | null;
}

export const DEFAULT_INPUTS: SimInputs = {
  schwarzschildRadius: 1,
  diskOuter: 12,
  diskLuminosity: 0,
  diskIntegrity: 1,
  particleSeed: 0,
  collapse: 0,
  grounding: 0,
  abstained: false,
  traceProgress: 0,
  hotspots: [],
  redshift: 0,
};

/** Stable numeric seed from a document id, so a document's disk structure
 *  is identical every time it is opened. */
export function seedFromId(id: string): number {
  let h = 2166136261;
  for (let i = 0; i < id.length; i += 1) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) / 4294967295) * 100;
}
