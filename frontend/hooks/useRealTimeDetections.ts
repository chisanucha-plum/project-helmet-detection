"use client"

import { useEffect, useState, useCallback } from "react"
import { DetectionResult, UseRealTimeDetectionsReturn } from "@/types/detection.types"
import {
  fetchHelmetHistory,
  subscribeToHelmetEvents,
} from "@/services/helmet-detection.service"

/**
 * Custom hook for managing real-time helmet detection data
 * Handles:
 * - Loading historical detections on mount
 * - Subscribing to SSE events when recording is active
 * - Error handling for API failures
 * - Cleanup of event subscriptions
 *
 * @returns Object with detections, loading state, error state, and recording controls
 */
export function useRealTimeDetections(): UseRealTimeDetectionsReturn {
  const [detections, setDetections] = useState<DetectionResult[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [isRecording, setIsRecording] = useState(false)

  // Load initial history from database
  useEffect(() => {
    const loadHistory = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const history = await fetchHelmetHistory(50)
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

  // Subscribe to real-time events when recording is active
  useEffect(() => {
    if (!isRecording) return

    const handleNewDetections = (newDetections: DetectionResult[]) => {
      // Add new detections to the top and keep only last 50
      setDetections((prev) => [...newDetections, ...prev].slice(0, 50))
      setError(null)
    }

    const handleError = (err: Error) => {
      console.error("Detection streaming error:", err)
      setError(err)
    }

    // Subscribe to SSE events
    const unsubscribe = subscribeToHelmetEvents(
      handleNewDetections,
      handleError
    )

    // Cleanup subscription when recording stops or component unmounts
    return () => {
      unsubscribe()
    }
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
