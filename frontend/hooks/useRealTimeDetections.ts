"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { DetectionResult, UseRealTimeDetectionsReturn } from "@/types/detection.types"
import {
  fetchHelmetHistory,
  subscribeToHelmetEvents,
} from "@/services/helmet-detection.service"

const MAX_DETECTIONS = 50
// Reconnect cadence for dropped SSE streams and how many consecutive failures
// before surfacing an error banner instead of silently retrying.
const SSE_RETRY_DELAY_MS = 5000
const MAX_SILENT_FAILURES = 3

interface UseRealTimeDetectionsOptions {
  /** Cap on stored/rendered detections (also caps the initial history load) */
  maxItems?: number
  /**
   * Notified with each SSE batch as it arrives — unlike the returned state it
   * fires only for live events, never for the initial history seed.
   */
  onDetections?: (detections: DetectionResult[]) => void
}

export function useRealTimeDetections(
  options: UseRealTimeDetectionsOptions = {}
): UseRealTimeDetectionsReturn {
  const { maxItems = MAX_DETECTIONS } = options

  // Keep the latest callback in a ref so SSE subscription effects do not
  // need to depend on (and restart for) a new callback identity per render.
  const onDetectionsRef = useRef(options.onDetections)
  useEffect(() => {
    onDetectionsRef.current = options.onDetections
  })

  const [detections, setDetections] = useState<DetectionResult[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [isRecording, setIsRecording] = useState(false)

  // Load initial history
  useEffect(() => {
    let cancelled = false

    const loadHistory = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const history = await fetchHelmetHistory(maxItems)
        if (!cancelled) setDetections(history)
      } catch (err) {
        const error = err instanceof Error ? err : new Error("Failed to load history")
        if (!cancelled) setError(error)
        console.error("Error loading helmet history:", error)
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    loadHistory()
    return () => {
      cancelled = true
    }
  }, [maxItems])

  // Subscribe to real-time events, auto-reconnecting while recording so a
  // transient backend restart does not require pressing start again.
  useEffect(() => {
    if (!isRecording) return

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    let failureCount = 0

    const connect = () => {
      if (cancelled) return
      return subscribeToHelmetEvents(
        (newDetections) => {
          failureCount = 0
          setError(null)
          setDetections((prev) => [...newDetections, ...prev].slice(0, maxItems))
          onDetectionsRef.current?.(newDetections)
        },
        (err) => {
          if (cancelled) return
          failureCount += 1
          console.warn(`Detection stream error (${failureCount}):`, err.message)
          if (failureCount >= MAX_SILENT_FAILURES) setError(err)
          retryTimer = setTimeout(connect, SSE_RETRY_DELAY_MS)
        }
      )
    }

    const unsubscribe = connect()

    return () => {
      cancelled = true
      if (retryTimer !== null) clearTimeout(retryTimer)
      unsubscribe?.()
    }
  }, [isRecording, maxItems])

  const handleSetIsRecording = useCallback((value: boolean) => {
    setIsRecording(value)
  }, [])

  return {
    detections,
    isLoading,
    error,
    isRecording,
    setIsRecording: handleSetIsRecording,
  }
}
