import type { CameraPreset } from "../sim/camera";

export interface Toggles {
  auto: boolean;
  cinematic: boolean;
  params: boolean;
  hud: boolean;
  sound: boolean;
}

/**
 * Bottom-right NAVIGATION panel, laid out to match the reference: a
 * full-width CINEMATIC SEQUENCE action, a 2x2 preset grid, a row of four
 * latching chips, a sound toggle, then the keyboard legend.
 *
 * Every control is a real <button> with an aria-pressed state where it
 * latches. The visible label is uppercased by CSS; the accessible name
 * stays in sentence case.
 */
export function NavigationPanel({
  preset,
  toggles,
  onPreset,
  onToggle,
  onCinematic,
  reducedMotion,
}: {
  preset: CameraPreset;
  toggles: Toggles;
  onPreset: (p: CameraPreset) => void;
  onToggle: (key: keyof Toggles) => void;
  onCinematic: () => void;
  reducedMotion: boolean;
}) {
  return (
    <nav
      aria-label="Camera and display controls"
      className="panel pointer-events-auto absolute bottom-6 right-6 w-[268px] p-3 sm:bottom-8 sm:right-8"
    >
      <h2 className="u-label m-0 mb-3">Navigation</h2>

      <button
        type="button"
        className="control mb-2 w-full"
        onClick={onCinematic}
        disabled={reducedMotion}
        aria-pressed={toggles.cinematic}
        title={
          reducedMotion
            ? "Cinematic sequence is disabled because your system requests reduced motion."
            : undefined
        }
      >
        Cinematic
      </button>

      <div className="mb-2 grid grid-cols-2 gap-2">
        {(["poster", "edge", "polar", "close"] as CameraPreset[]).map((p) => (
          <button
            key={p}
            type="button"
            className="control"
            data-active={preset === p}
            aria-pressed={preset === p}
            onClick={() => onPreset(p)}
          >
            {p}
          </button>
        ))}
      </div>

      <div className="mb-2 grid grid-cols-4 gap-1">
        <Chip
          label="Auto"
          active={toggles.auto}
          disabled={reducedMotion}
          onClick={() => onToggle("auto")}
        />
        <Chip
          label="Cinematic"
          active={toggles.cinematic}
          disabled={reducedMotion}
          onClick={() => onToggle("cinematic")}
        />
        <Chip label="Params" active={toggles.params} onClick={() => onToggle("params")} />
        <Chip label="HUD" active={toggles.hud} onClick={() => onToggle("hud")} />
      </div>

      <button
        type="button"
        className="control mb-3 w-full"
        aria-pressed={toggles.sound}
        onClick={() => onToggle("sound")}
      >
        <span aria-hidden="true">{toggles.sound ? "🔊" : "🔇"}</span>
        <span>Sound: {toggles.sound ? "on" : "off"}</span>
      </button>

      <KeyboardLegend />
    </nav>
  );
}

function Chip({
  label,
  active,
  disabled,
  onClick,
}: {
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="control control--chip"
      data-active={active}
      aria-pressed={active}
      disabled={disabled}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

/**
 * Always visible, as in the reference. It is also the discoverability
 * surface for keyboard operation, so it is a real list rather than a
 * decorative string — a screen reader reads it as six shortcuts, not as
 * one run-on sentence of punctuation.
 */
export function KeyboardLegend() {
  const rows: Array<[string, string]> = [
    ["1-4", "views"],
    ["C", "cine"],
    ["R", "orbit"],
    ["P", "params"],
    ["M", "sound"],
    ["H", "hud"],
  ];
  return (
    <div className="text-center">
      <ul className="u-legend m-0 flex list-none flex-wrap justify-center gap-x-[6px] gap-y-1 p-0">
        {rows.map(([key, action]) => (
          <li key={key} className="whitespace-nowrap">
            <kbd className="font-mono not-italic">{key}</kbd> {action}
          </li>
        ))}
      </ul>
      <p className="u-legend m-0 mt-[6px] opacity-80">Drag orbit · scroll zoom</p>
    </div>
  );
}
