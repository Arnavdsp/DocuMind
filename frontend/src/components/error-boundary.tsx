import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

/**
 * Last-resort UI boundary, rendered in the same language as the rest of the
 * interface: SIGNAL LOST with a REINITIALISE action.
 *
 * It mirrors the backend's AppError discipline — the real error text goes
 * to the console for whoever is debugging, and the person looking at the
 * screen gets a user-safe message and a way forward.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[gargantua] unhandled UI error", error, info);
  }

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <main className="flex h-[100dvh] items-center justify-center bg-[color:var(--color-void)] px-6">
        <div className="panel max-w-[420px] p-6 text-center">
          <h1 className="u-label m-0 mb-3">Signal lost</h1>
          <p className="prose-readout m-0 mb-5 text-center">
            The interface stopped unexpectedly. Nothing was lost from the
            document or the index — reloading re-establishes the link.
          </p>
          <button
            type="button"
            className="control mx-auto"
            onClick={() => window.location.reload()}
          >
            Reinitialise
          </button>
        </div>
      </main>
    );
  }
}
