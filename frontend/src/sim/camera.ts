/* ============================================================================
   Observer camera.
   ----------------------------------------------------------------------------
   OBSERVER DISTANCE and DISK INCLINATION in the HUD are read straight off
   this object every frame. They are the camera's actual state, which is
   why they drift continuously while AUTO orbit is running — exactly as the
   reference does.
   ========================================================================= */

export type CameraPreset = "poster" | "edge" | "polar" | "close";

export interface CameraState {
  /** Distance from the singularity, in units of r_s. */
  distance: number;
  /** Polar angle from the disk plane, radians. 0 = edge-on. */
  inclination: number;
  /** Azimuth, radians. */
  azimuth: number;
}

/**
 * Preset framings.
 *
 * `distance` here is a MULTIPLE OF THE DISK'S OUTER RADIUS, not an absolute
 * number of r_s. Documents differ in index size, so the disk differs in
 * extent; framing against a fixed r_s distance makes a large document
 * overflow the frame and a small one disappear into it. Framing against the
 * disk means every document composes identically, and the absolute distance
 * — which is what OBSERVER DISTANCE reports — comes out as a real derived
 * number rather than a constant.
 *
 * The multipliers were measured, not guessed: `frontend/frame_probe.mjs`
 * renders the shader across a sweep of k and reports the disk's width as a
 * percentage of the frame. k = 2.15 frames the disk at roughly the width the reference shows.
 */
export const PRESETS: Record<CameraPreset, CameraState> = {
  // The hero framing: above the plane far enough to see the disk's far side
  // lensed over the top of the shadow.
  poster: { distance: 2.15, inclination: 0.29, azimuth: 0.0 },
  // Edge-on. The disk collapses to a line and the lensed images above and
  // below the shadow become the whole picture.
  edge: { distance: 2.25, inclination: 0.045, azimuth: 0.6 },
  // Down the rotation axis: the disk reads as a flat annulus and the photon
  // ring is a clean circle.
  polar: { distance: 2.5, inclination: 1.35, azimuth: 0.0 },
  // Inside the disk's outer radius. Strong lensing, low frame rate.
  close: { distance: 1.15, inclination: 0.2, azimuth: 2.1 },
};

export const PRESET_ORDER: CameraPreset[] = ["poster", "edge", "polar", "close"];

const MIN_DISTANCE = 4;
const MAX_DISTANCE = 120;
const MAX_INCLINATION = Math.PI / 2 - 0.02;

export class ObserverCamera {
  /** Live state. `distance` is in units of r_s and is what the HUD prints. */
  state: CameraState = { distance: 18, inclination: 0.29, azimuth: 0 };

  /** The disk's outer radius in r_s. Presets are multiples of this. */
  frameRadius = 10;

  private preset: CameraPreset = "poster";
  private target: CameraState = { distance: 18, inclination: 0.29, azimuth: 0 };
  private transitioning = false;

  /** Radians per second of automatic orbit. Matches the reference's drift
   *  rate, measured from the screencast: the observer distance moved from
   *  37.88 r_s to 24.57 r_s over roughly 50 seconds of a slow ellipse. */
  autoOrbitSpeed = 0.055;

  /** Called when the indexed document changes size. Re-frames without
   *  moving the camera if the user has taken manual control. */
  setFrameRadius(radius: number): void {
    if (!Number.isFinite(radius) || radius <= 0) return;
    this.frameRadius = radius;
    if (!this.manual) this.goTo(this.preset, true);
  }

  private manual = false;

  goTo(preset: CameraPreset, immediate = false): void {
    this.preset = preset;
    this.manual = false;
    const p = PRESETS[preset];
    this.target = { ...p, distance: p.distance * this.frameRadius };
    if (immediate) {
      this.state = { ...this.target };
      this.transitioning = false;
    } else {
      this.transitioning = true;
    }
  }

  /** Drag to orbit. dx/dy in CSS pixels. */
  orbit(dx: number, dy: number): void {
    this.transitioning = false;
    this.manual = true;
    this.state.azimuth += dx * 0.005;
    this.state.inclination = clamp(
      this.state.inclination + dy * 0.004,
      -MAX_INCLINATION,
      MAX_INCLINATION
    );
    this.target = { ...this.state };
  }

  /** Scroll or pinch to zoom. Multiplicative so it feels linear in log
   *  space, which is what "zoom" means to a hand on a trackpad. */
  zoom(factor: number): void {
    this.transitioning = false;
    this.manual = true;
    this.state.distance = clamp(this.state.distance * factor, MIN_DISTANCE, MAX_DISTANCE);
    this.target = { ...this.state };
  }

  /**
   * Advance one frame.
   *
   * `auto` runs the slow orbit. It is suppressed entirely under
   * prefers-reduced-motion — the caller passes auto=false in that case and
   * the camera holds a fixed pose, which is the accessible behaviour the
   * brief asks for and not a degraded one: the render is still fully
   * legible standing still.
   */
  update(dt: number, auto: boolean): void {
    if (this.transitioning) {
      // Critically-damped-ish ease toward the preset. Fast enough to feel
      // like a cut with weight, slow enough to read as a camera move.
      const k = 1 - Math.exp(-dt * 3.2);
      this.state.distance += (this.target.distance - this.state.distance) * k;
      this.state.inclination += (this.target.inclination - this.state.inclination) * k;
      this.state.azimuth += (this.target.azimuth - this.state.azimuth) * k;
      if (
        Math.abs(this.target.distance - this.state.distance) < 0.02 &&
        Math.abs(this.target.inclination - this.state.inclination) < 0.002
      ) {
        this.transitioning = false;
      }
    } else if (auto) {
      this.state.azimuth += this.autoOrbitSpeed * dt;
      // A gentle breathing motion in distance and inclination so AUTO is a
      // composed move rather than a turntable spin.
      this.state.inclination = 0.28 + 0.16 * Math.sin(this.state.azimuth * 0.37);
      this.state.distance =
        this.frameRadius * (2.2 + 0.35 * Math.sin(this.state.azimuth * 0.23));
    }
  }

  /** Cartesian position, units of r_s. */
  position(): [number, number, number] {
    const { distance: d, inclination: i, azimuth: a } = this.state;
    return [
      d * Math.cos(i) * Math.cos(a),
      d * Math.sin(i),
      d * Math.cos(i) * Math.sin(a),
    ];
  }

  /** Column-major 3x3 basis (right, up, forward) looking at the origin. */
  basis(): Float32Array {
    const [px, py, pz] = this.position();
    const len = Math.hypot(px, py, pz) || 1;
    // Forward points from the observer toward the singularity; the shader
    // negates z, so we store the outward vector.
    const fx = px / len;
    const fy = py / len;
    const fz = pz / len;

    // World up, with a guard for the polar preset where up and forward
    // become parallel and the cross product degenerates.
    let ux = 0;
    let uy = 1;
    let uz = 0;
    if (Math.abs(fy) > 0.999) {
      ux = 0;
      uy = 0;
      uz = 1;
    }

    // right = normalize(cross(up, forward))
    let rx = uy * fz - uz * fy;
    let ry = uz * fx - ux * fz;
    let rz = ux * fy - uy * fx;
    const rl = Math.hypot(rx, ry, rz) || 1;
    rx /= rl;
    ry /= rl;
    rz /= rl;

    // trueUp = cross(forward, right)
    const tx = fy * rz - fz * ry;
    const ty = fz * rx - fx * rz;
    const tz = fx * ry - fy * rx;

    return new Float32Array([rx, ry, rz, tx, ty, tz, fx, fy, fz]);
  }

  /** Inclination in degrees, for the HUD. */
  inclinationDegrees(): number {
    return (this.state.inclination * 180) / Math.PI;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), hi);
}
