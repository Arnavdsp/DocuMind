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
  // Tagged with the job it belongs to, so a result for a previous job is never
  // returned once jobId has moved on.
  const [latest, setLatest] = useState<{ jobId: string; job: JobRecord } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    setError(null);

    const poll = async () => {
      try {
        const result = await api.getJob(jobId);
        if (cancelled) return; // response for a job we've already moved on from
        setLatest({ jobId, job: result });
        if (result.status === "succeeded" || result.status === "failed") {
          if (intervalRef.current) window.clearInterval(intervalRef.current);
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not check processing status.");
        if (intervalRef.current) window.clearInterval(intervalRef.current);
      }
    };

    poll();
    intervalRef.current = window.setInterval(poll, 900);
    return () => {
      cancelled = true;
      if (intervalRef.current) window.clearInterval(intervalRef.current);
    };
  }, [jobId]);

  const job = latest && latest.jobId === jobId ? latest.job : null;
  return { job, error };
}
