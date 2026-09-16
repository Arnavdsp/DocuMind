import type { ReactNode } from "react";
import { UNSET } from "../sim/mapping";

/**
 * One telemetry row: label left, value right of it in a fixed column.
 * Matches the reference's measured layout — label at the panel's left
 * edge, value in a column at a constant offset, both left-aligned within
 * their own column (not right-aligned to the panel edge).
 *
 * `value` of null renders an em-dash. There is no other unset state and no
 * component in this interface substitutes a zero, a spinner or a
 * placeholder for a value it does not have.
 */
export function TelemetryRow({
  label,
  value,
  tone,
}: {
  label: string;
  value: string | null;
  /** Optional grounding colour. Only the signal rows pass this. */
  tone?: string | null;
}) {
  const shown = value ?? UNSET;
  const unset = value === null;
  return (
    <div className="flex items-baseline gap-4 py-[3px]">
      <span className="u-label w-[168px] shrink-0">{label}</span>
      <span
        className="u-value"
        style={
          tone && !unset
            ? { color: tone, textShadow: "var(--shadow-bloom-amber)" }
            : unset
              ? { color: "var(--color-chrome-dim)" }
              : undefined
        }
      >
        {shown}
      </span>
    </div>
  );
}

/** A labelled block used inside panels — a header hairline plus content. */
export function Section({
  label,
  children,
  action,
}: {
  label: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <section className="mb-5">
      <header className="mb-2 flex items-center justify-between gap-3">
        <h2 className="u-label m-0">{label}</h2>
        {action}
      </header>
      <div className="hairline mb-3" />
      {children}
    </section>
  );
}
