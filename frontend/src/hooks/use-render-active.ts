import { useEffect, useState, type RefObject } from "react";

/**
 * True only when the canvas is worth rendering: the tab is visible AND the
 * element is at least partially on screen.
 *
 * Both halves matter. `visibilitychange` alone leaves the raymarcher
 * burning a core when the user scrolls the canvas out of view in a visible
 * tab; IntersectionObserver alone leaves it running in a background tab on
 * browsers that don't throttle rAF aggressively.
 */
export function useRenderActive(ref: RefObject<HTMLElement | null>): boolean {
  const [visible, setVisible] = useState(
    () => typeof document === "undefined" || document.visibilityState === "visible"
  );
  const [onScreen, setOnScreen] = useState(true);

  useEffect(() => {
    const onChange = () => setVisible(document.visibilityState === "visible");
    document.addEventListener("visibilitychange", onChange);
    return () => document.removeEventListener("visibilitychange", onChange);
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      (entries) => setOnScreen(entries.some((e) => e.isIntersecting)),
      { threshold: 0 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);

  return visible && onScreen;
}
