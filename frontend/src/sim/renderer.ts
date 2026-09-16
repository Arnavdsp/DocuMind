/* ============================================================================
   WebGL host for the Schwarzschild shader.
   ----------------------------------------------------------------------------
   Deliberately raw WebGL rather than react-three-fiber. The scene is one
   full-screen triangle running one fragment shader; three.js contributes a
   scene graph, a material system and a render loop that this needs none
   of, at roughly 600 KB gzipped, and its internal rAF loop would have to
   be fought to implement the visibility/offscreen pausing below. See
   MIGRATION.md.
   ========================================================================= */

import fragmentSource from "./schwarzschild.frag.glsl?raw";
import vertexSource from "./schwarzschild.vert.glsl?raw";
import { ObserverCamera } from "./camera";
import { AdaptiveQuality, FrameRateMeter, type QualityProfile } from "./quality";
import { DEFAULT_INPUTS, type SimInputs, type Telemetry } from "./types";

type UniformMap = Record<string, WebGLUniformLocation | null>;

export class SchwarzschildRenderer {
  readonly camera = new ObserverCamera();
  readonly quality = new AdaptiveQuality();

  private gl: WebGLRenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private buffer: WebGLBuffer | null = null;
  private uniforms: UniformMap = {};
  private meter = new FrameRateMeter();

  private rafId = 0;
  private running = false;
  private startTime = 0;
  private lastFrame = 0;

  private inputs: SimInputs = { ...DEFAULT_INPUTS };
  private autoOrbit = true;
  private reducedMotion = false;
  private hotspotBuffer = new Float32Array(24);

  private readonly canvas: HTMLCanvasElement;
  private readonly onTelemetry: (t: Telemetry) => void;
  private readonly onContextLost: () => void;

  constructor(
    canvas: HTMLCanvasElement,
    onTelemetry: (t: Telemetry) => void,
    onContextLost: () => void
  ) {
    this.canvas = canvas;
    this.onTelemetry = onTelemetry;
    this.onContextLost = onContextLost;
  }

  /** Returns false when WebGL is unavailable or the shader fails to
   *  compile. The caller renders StaticFallback in that case — a real,
   *  usable interface, not an apology. */
  init(): boolean {
    const gl =
      (this.canvas.getContext("webgl", {
        antialias: false,
        alpha: false,
        depth: false,
        stencil: false,
        powerPreference: "high-performance",
        failIfMajorPerformanceCaveat: false,
      }) as WebGLRenderingContext | null) ??
      (this.canvas.getContext("experimental-webgl") as WebGLRenderingContext | null);

    if (!gl) return false;
    this.gl = gl;

    this.canvas.addEventListener("webglcontextlost", this.handleContextLost, false);
    this.canvas.addEventListener("webglcontextrestored", this.handleContextRestored, false);

    const program = buildProgram(gl, vertexSource, fragmentSource);
    if (!program) return false;
    this.program = program;

    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    // One oversized triangle covering clip space.
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW
    );

    const loc = gl.getAttribLocation(program, "aPosition");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    for (const name of UNIFORM_NAMES) {
      this.uniforms[name] = gl.getUniformLocation(program, name);
    }
    this.uniforms.uHotspots = gl.getUniformLocation(program, "uHotspots[0]");

    gl.useProgram(program);
    this.resize();
    return true;
  }

  setInputs(next: Partial<SimInputs>): void {
    this.inputs = { ...this.inputs, ...next };
    // Keep the camera framing the disk it is actually looking at.
    const rs = this.inputs.schwarzschildRadius;
    if (rs > 0) this.camera.setFrameRadius(this.inputs.diskOuter / rs);
  }

  setAutoOrbit(on: boolean): void {
    this.autoOrbit = on;
  }

  setReducedMotion(on: boolean): void {
    this.reducedMotion = on;
    if (on) {
      // No orbit, no cinematic sequence, static render. The camera holds
      // whatever pose it has; preset changes still work because those are
      // user-initiated and instantaneous.
      this.autoOrbit = false;
    }
  }

  lowerQuality(): QualityProfile {
    const profile = this.quality.forceLower();
    this.resize();
    return profile;
  }

  /** Backing store is sized to CSS pixels x resolutionScale, capped at 2x
   *  device pixels. On a 3x phone screen, rendering a raymarcher at native
   *  resolution is nine times the fragment work for no visible gain. */
  resize(): void {
    const gl = this.gl;
    if (!gl) return;
    const scale = this.quality.current.resolutionScale;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr * scale));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr * scale));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.startTime = performance.now();
    this.lastFrame = this.startTime;
    this.meter.reset();
    this.rafId = requestAnimationFrame(this.frame);
  }

  /** Called when the tab is hidden or the canvas scrolls offscreen. A
   *  raymarcher left running in a background tab is a battery bug. */
  stop(): void {
    this.running = false;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = 0;
    this.meter.reset();
    // Emit one last telemetry frame so FRAME RATE goes to em-dash rather
    // than freezing at a stale reading that no longer describes anything.
    this.emitTelemetry(null);
  }

  dispose(): void {
    this.stop();
    this.canvas.removeEventListener("webglcontextlost", this.handleContextLost);
    this.canvas.removeEventListener("webglcontextrestored", this.handleContextRestored);
    const gl = this.gl;
    if (gl) {
      if (this.buffer) gl.deleteBuffer(this.buffer);
      if (this.program) gl.deleteProgram(this.program);
    }
    this.gl = null;
  }

  private handleContextLost = (event: Event) => {
    event.preventDefault();
    this.stop();
    this.onContextLost();
  };

  private handleContextRestored = () => {
    if (this.init()) this.start();
  };

  private frame = (now: number) => {
    if (!this.running) return;
    const gl = this.gl;
    const program = this.program;
    if (!gl || !program) return;

    const dt = Math.min((now - this.lastFrame) / 1000, 0.1);
    this.lastFrame = now;

    this.camera.update(dt, this.autoOrbit && !this.reducedMotion);

    const fps = this.meter.tick(now);
    if (fps !== null) {
      const changed = this.quality.observe(fps, now);
      if (changed) this.resize();
      this.emitTelemetry(fps);
    }

    const profile = this.quality.current;
    const u = this.uniforms;
    const i = this.inputs;

    gl.uniform2f(u.uResolution, this.canvas.width, this.canvas.height);
    gl.uniform1f(u.uTime, (now - this.startTime) / 1000);

    // The camera reports its distance in units of r_s, and the HUD prints
    // that number next to the label "RS". So the position handed to the
    // shader must be scaled by r_s — otherwise the camera sits at a fixed
    // distance in absolute scene units, the apparent size of the hole
    // varies with document size, and OBSERVER DISTANCE is simply wrong for
    // every document that isn't exactly 1 r_s. A telemetry row that lies
    // about its own units is the exact failure this build exists to avoid.
    const rs = i.schwarzschildRadius;
    const pos = this.camera.position();
    gl.uniform3f(u.uCamPos, pos[0] * rs, pos[1] * rs, pos[2] * rs);
    gl.uniformMatrix3fv(u.uCamBasis, false, this.camera.basis());
    gl.uniform1f(u.uFov, 0.9);

    gl.uniform1i(u.uSteps, profile.geodesicSteps);
    gl.uniform1i(u.uDiskSamples, profile.diskSamples);

    gl.uniform1f(u.uRs, i.schwarzschildRadius);
    gl.uniform1f(u.uDiskInner, i.schwarzschildRadius * 3.0); // ISCO
    gl.uniform1f(u.uDiskOuter, i.diskOuter);
    gl.uniform1f(u.uDiskLuminosity, i.diskLuminosity);
    gl.uniform1f(u.uDiskIntegrity, i.diskIntegrity);
    gl.uniform1f(u.uParticleSeed, i.particleSeed);
    gl.uniform1f(u.uCollapse, i.collapse);
    gl.uniform1f(u.uGrounding, i.grounding);
    gl.uniform1f(u.uAbstained, i.abstained ? 1 : 0);
    gl.uniform1f(u.uTraceProgress, i.traceProgress);
    gl.uniform1f(u.uRedshift, i.redshift);

    const count = Math.min(i.hotspots.length, 8);
    gl.uniform1i(u.uHotspotCount, count);
    this.hotspotBuffer.fill(0);
    for (let k = 0; k < count; k += 1) {
      this.hotspotBuffer[k * 3] = i.hotspots[k].theta;
      this.hotspotBuffer[k * 3 + 1] = i.hotspots[k].intensity;
    }
    gl.uniform3fv(u.uHotspots, this.hotspotBuffer);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.rafId = requestAnimationFrame(this.frame);
  };

  private emitTelemetry(fps: number | null): void {
    this.onTelemetry({
      observerDistance: this.camera.state.distance,
      diskInclination: this.camera.inclinationDegrees(),
      geodesicSteps: this.quality.current.geodesicSteps,
      profile: this.quality.current.tier,
      frameRate: fps,
    });
  }
}

const UNIFORM_NAMES = [
  "uResolution",
  "uTime",
  "uCamPos",
  "uCamBasis",
  "uFov",
  "uSteps",
  "uDiskSamples",
  "uRs",
  "uDiskInner",
  "uDiskOuter",
  "uDiskLuminosity",
  "uDiskIntegrity",
  "uParticleSeed",
  "uCollapse",
  "uGrounding",
  "uAbstained",
  "uTraceProgress",
  "uHotspotCount",
  "uRedshift",
] as const;

function compile(
  gl: WebGLRenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    // Logged, not surfaced: a GLSL compiler log is an internal detail. The
    // user sees SIGNAL LOST with a REINITIALISE action, matching the
    // AppError discipline on the backend.
    console.error("[gargantua] shader compile failed", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function buildProgram(
  gl: WebGLRenderingContext,
  vert: string,
  frag: string
): WebGLProgram | null {
  const vs = compile(gl, gl.VERTEX_SHADER, vert);
  const fs = compile(gl, gl.FRAGMENT_SHADER, frag);
  if (!vs || !fs) return null;
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("[gargantua] program link failed", gl.getProgramInfoLog(program));
    return null;
  }
  return program;
}

