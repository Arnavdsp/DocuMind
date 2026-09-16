import { useCallback, useEffect, useRef, useState } from "react";
import { useReducedMotion } from "../hooks/use-reduced-motion";
import { useRenderActive } from "../hooks/use-render-active";
import type { CameraPreset } from "../sim/camera";
import { SchwarzschildRenderer } from "../sim/renderer";
import type { SimInputs, Telemetry } from "../sim/types";

interface Props {
  inputs: SimInputs;
  preset: CameraPreset;
  autoOrbit: boolean;
  onTelemetry: (t: Telemetry) => void;
  /** Called when WebGL is unavailable or the context is lost — the shell
   *  swaps in StaticFallback and shows SIGNAL LOST / REINITIALISE. */
  onUnavailable: () => void;
  /** Registers the LOWER QUALITY action with the shell. */
  onReady: (api: { lowerQuality: () => void }) => void;
}

/**
 * The canvas host. Everything React does here is imperative-boundary
 * plumbing: the render loop lives in SchwarzschildRenderer and never
 * re-renders a component. Uniform updates go through a ref so a new
 * `inputs` object does not cost a React commit per frame.
 */
export function GargantuaCanvas({
  inputs,
  preset,
  autoOrbit,
  onTelemetry,
  onUnavailable,
  onReady,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<SchwarzschildRenderer | null>(null);
  const [failed, setFailed] = useState(false);

  const reducedMotion = useReducedMotion();
  const active = useRenderActive(wrapRef);

  const telemetryRef = useRef(onTelemetry);
  telemetryRef.current = onTelemetry;

  const unavailableRef = useRef(onUnavailable);
  unavailableRef.current = onUnavailable;

  // --- Mount -------------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const renderer = new SchwarzschildRenderer(
      canvas,
      (t) => telemetryRef.current(t),
      () => {
        setFailed(true);
        unavailableRef.current();
      }
    );

    if (!renderer.init()) {
      setFailed(true);
      unavailableRef.current();
      return;
    }

    rendererRef.current = renderer;
    onReady({
      lowerQuality: () => {
        renderer.lowerQuality();
      },
    });

    const onResize = () => renderer.resize();
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      rendererRef.current = null;
    };
    // onReady is a stable callback from the shell; re-running init on every
    // parent render would recompile the shader.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Run / pause -------------------------------------------------------
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer || failed) return;
    if (active) renderer.start();
    else renderer.stop();
  }, [active, failed]);

  // --- Inputs ------------------------------------------------------------
  useEffect(() => {
    rendererRef.current?.setInputs(inputs);
  }, [inputs]);

  useEffect(() => {
    rendererRef.current?.setReducedMotion(reducedMotion);
  }, [reducedMotion]);

  useEffect(() => {
    rendererRef.current?.setAutoOrbit(autoOrbit && !reducedMotion);
  }, [autoOrbit, reducedMotion]);

  useEffect(() => {
    rendererRef.current?.camera.goTo(preset, reducedMotion);
  }, [preset, reducedMotion]);

  // --- Pointer: drag to orbit, scroll/pinch to zoom -----------------------
  const dragging = useRef(false);
  const last = useRef({ x: 0, y: 0 });

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    dragging.current = true;
    last.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    const dx = e.clientX - last.current.x;
    const dy = e.clientY - last.current.y;
    last.current = { x: e.clientX, y: e.clientY };
    rendererRef.current?.camera.orbit(dx, dy);
  }, []);

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    dragging.current = false;
    (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
  }, []);

  const onWheel = useCallback((e: React.WheelEvent) => {
    rendererRef.current?.camera.zoom(e.deltaY > 0 ? 1.08 : 0.93);
  }, []);

  if (failed) return null;

  return (
    <div ref={wrapRef} className="absolute inset-0">
      <canvas
        ref={canvasRef}
        className="h-full w-full touch-none"
        style={{ display: "block", background: "#000" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        /* The canvas is decorative: every value it depicts is also present
           in the DOM as text. Marking it aria-hidden is what keeps a
           screen reader from landing on an unlabelled graphics surface,
           and focus is never trapped here because it is not focusable. */
        aria-hidden="true"
      />
    </div>
  );
}

export default GargantuaCanvas;
