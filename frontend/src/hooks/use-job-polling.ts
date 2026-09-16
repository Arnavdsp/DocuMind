import { useEffect, useRef, useState } from "react";
import { api } from "../services/api";
import type { JobRecord } from "../types/api";

/**
 * Polls a job until it reaches a terminal state. Used so upload/ingestion
 * never blocks the UI thread — the caller gets progressive stage updates
 * (Extracting -> OCR -> Chunking -> Embedding -> Indexing -> Ready) to
 * render instead of a single opaque spinner.
 */
export function useJobPolling(jobId: string | null) {
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (!jobId) return;
    setError(null);

    const poll = async () => {
      try {
        const result = await api.getJob(jobId);
        setJob(result);
        if (result.status === "succeeded" || result.status === "failed") {
          if (intervalRef.current) window.clearInterval(intervalRef.current);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not check processing status.");
        if (intervalRef.current) window.clearInterval(intervalRef.current);
      }
    };

    poll();
    intervalRef.current = window.setInterval(poll, 900);
    return () => {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
    };
  }, [jobId]);

  return { job, error };
}
