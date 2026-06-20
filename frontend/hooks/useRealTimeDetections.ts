"use client"

import { useEffect, useState, useCallback } from "react"
import { DetectionResult, UseRealTimeDetectionsReturn } from "@/types/detection.types"
import {
  fetchHelmetHistory,
  subscribeToHelmetEvents,
} from "@/services/helmet-detection.service"

const MAX_DETECTIONS = 50

export function useRealTimeDetections(): UseRealTimeDetectionsReturn {
  const [detections, setDetections] = useState<DetectionResult[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [isRecording, setIsRecording] = useState(false)

  // Load initial history
  useEffect(() => {
    const loadHistory = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const history = await fetchHelmetHistory(MAX_DETECTIONS)
        setDetections(history)
      } catch (err) {
        const error = err instanceof Error ? err : new Error("Failed to load history")
        setError(error)
        console.error("Error loading helmet history:", error)
      } finally {
        setIsLoading(false)
      }
    }

    loadHistory()
  }, [])

  // Subscribe to real-time events
  useEffect(() => {
    if (!isRecording) return

    // Subscribe to real-time events
    const handleNewDetections = (newDetections: DetectionResult[]) => {
      setDetections((prev) => [...newDetections, ...prev].slice(0, MAX_DETECTIONS))
      setError(null)
    }

    const handleError = (err: Error) => {
      console.error("Detection streaming error:", err)
      setError(err)
    }

    const unsubscribe = subscribeToHelmetEvents(handleNewDetections, handleError)
    return () => unsubscribe()
  }, [isRecording])

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
