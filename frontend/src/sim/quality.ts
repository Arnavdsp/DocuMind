/* ============================================================================
   Render quality — measured, adaptive, and honestly reported.
   ----------------------------------------------------------------------------
   RENDER PROFILE and FRAME RATE in the HUD are driven entirely by this
   module. Both are measured values. The profile is not a user preference
   that the renderer then ignores; it is the resolution scale and geodesic
   step count the fragment shader is actually running at this instant.
   ========================================================================= */

export type QualityTier = "cinematic" | "standard" | "reduced" | "minimal";

export interface QualityProfile {
  tier: QualityTier;
  /** Backing-store scale relative to CSS pixels. 1 = native, 0.5 = quarter
   *  the fragment work. Capped below devicePixelRatio on HiDPI screens. */
  resolutionScale: number;
  /** Integration steps per null geodesic. This is the number reported as
   *  GEODESIC STEPS — it is the shader's actual loop bound. */
  geodesicSteps: number;
  /** Number of secondary lensing bounces traced for the disk. */
  diskSamples: number;
}

export const PROFILES: Record<QualityTier, QualityProfile> = {
  cinematic: { tier: "cinematic", resolutionScale: 1.0, geodesicSteps: 460, diskSamples: 3 },
  standard: { tier: "standard", resolutionScale: 0.8, geodesicSteps: 300, diskSamples: 2 },
  reduced: { tier: "reduced", resolutionScale: 0.6, geodesicSteps: 180, diskSamples: 1 },
  minimal: { tier: "minimal", resolutionScale: 0.45, geodesicSteps: 110, diskSamples: 1 },
};

const ORDER: QualityTier[] = ["cinematic", "standard", "reduced", "minimal"];

export function lowerTier(tier: QualityTier): QualityTier {
  const i = ORDER.indexOf(tier);
  return ORDER[Math.min(i + 1, ORDER.length - 1)];
}

export function raiseTier(tier: QualityTier): QualityTier {
  const i = ORDER.indexOf(tier);
  return ORDER[Math.max(i - 1, 0)];
}

export function isLowestTier(tier: QualityTier): boolean {
  return tier === ORDER[ORDER.length - 1];
}

/**
 * Starting tier. Mobile and low-core devices begin at `reduced` rather
 * than starting at `cinematic` and stuttering their way down — a shader
 * compile stall on the first frame of a phone is the single worst failure
 * mode of this kind of interface, and it happens on the upload path.
 */
export function initialTier(): QualityTier {
  if (typeof window === "undefined") return "standard";
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  const narrow = window.innerWidth < 900;
  const cores = navigator.hardwareConcurrency ?? 4;
  if (coarse || narrow) return cores <= 4 ? "minimal" : "reduced";
  if (cores <= 4) return "reduced";
  return "standard";
}

/**
 * Adaptive controller. Watches measured FPS over a rolling window and
 * moves one tier at a time, with hysteresis and a cooldown so the
 * renderer cannot oscillate between two tiers on a borderline machine.
 *
 * Downshifts are fast (2 consecutive bad windows) because a user watching
 * a 12fps render wants relief immediately. Upshifts are slow (6
 * consecutive good windows) because upshifting into a stall is worse than
 * staying one tier low.
 */
export class AdaptiveQuality {
  private tier: QualityTier;
  private pinned = false;
  private badWindows = 0;
  private goodWindows = 0;
  private lastChange = 0;

  constructor(tier: QualityTier = initialTier()) {
    this.tier = tier;
  }

  get current(): QualityProfile {
    return PROFILES[this.tier];
  }

  /** User pressed LOWER QUALITY. Drops a tier and stops automatic
   *  upshifting — an explicit choice outranks the controller's opinion. */
  forceLower(): QualityProfile {
    this.tier = lowerTier(this.tier);
    this.pinned = true;
    this.badWindows = 0;
    this.goodWindows = 0;
    return this.current;
  }

  setTier(tier: QualityTier): QualityProfile {
    this.tier = tier;
    this.pinned = true;
    return this.current;
  }

  /** Feed one measured FPS sample (one window, ~1s). Returns a new profile
   *  if the tier changed, otherwise null. */
  observe(fps: number, now: number): QualityProfile | null {
    if (this.pinned) return null;
    if (now - this.lastChange < 2500) return null;

    if (fps < 24) {
      this.badWindows += 1;
      this.goodWindows = 0;
    } else if (fps > 52) {
      this.goodWindows += 1;
      this.badWindows = 0;
    } else {
      this.badWindows = 0;
      this.goodWindows = 0;
    }

    if (this.badWindows >= 2 && !isLowestTier(this.tier)) {
      this.tier = lowerTier(this.tier);
      this.badWindows = 0;
      this.lastChange = now;
      return this.current;
    }
    if (this.goodWindows >= 6 && this.tier !== "cinematic") {
      this.tier = raiseTier(this.tier);
      this.goodWindows = 0;
      this.lastChange = now;
      return this.current;
    }
    return null;
  }
}

/**
 * Rolling FPS meter. Reports over a ~1s window rather than instantaneous
 * frame deltas, because a per-frame reciprocal is far too noisy to put on
 * screen — it produces a number that flickers between 40 and 70 and tells
 * the viewer nothing.
 */
export class FrameRateMeter {
  private frames = 0;
  private windowStart = 0;
  private latest: number | null = null;

  /** Call once per rendered frame. Returns a fresh FPS reading when a
   *  window closes, otherwise null. */
  tick(now: number): number | null {
    if (this.windowStart === 0) {
      this.windowStart = now;
      return null;
    }
    this.frames += 1;
    const elapsed = now - this.windowStart;
    if (elapsed >= 1000) {
      this.latest = (this.frames * 1000) / elapsed;
      this.frames = 0;
      this.windowStart = now;
      return this.latest;
    }
    return null;
  }

  /** Null until the first window closes — so the HUD shows an em-dash on
   *  startup instead of a fabricated 60. */
  get value(): number | null {
    return this.latest;
  }

  reset(): void {
    this.frames = 0;
    this.windowStart = 0;
    this.latest = null;
  }
}
