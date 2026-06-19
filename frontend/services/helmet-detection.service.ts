/**
 * Centralized service for helmet detection API calls
 * Handles all communication with backend endpoints
 */

import {
  DetectionHistoryItem,
  DetectionEvent,
  DetectionResult,
} from "@/types/detection.types"

const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

/**
 * Fetch helmet detection history from the database
 * @param limit Number of records to fetch
 * @returns Array of detection results
 */
export async function fetchHelmetHistory(
  limit: number = 50
): Promise<DetectionResult[]> {
  try {
    const response = await fetch(`${BASE_URL}/helmet/history?limit=${limit}`)

    if (!response.ok) {
      throw new Error(`Failed to fetch helmet history: ${response.status}`)
    }

    const data: DetectionHistoryItem[] = await response.json()

    return data.map((item) => ({
      id: item.id,
      timestamp: item.timestamp ?? "",
      camera: "กล้องหลัก",
      helmetStatus: item.helmet_status === true ? "wearing" : "not-wearing",
      passengerCount: item.passenger_count ?? 1,
      violation: item.violation ?? false,
    }))
  } catch (error) {
    console.error("Error fetching helmet history:", error)
    throw error
  }
}

/**
 * Subscribe to real-time helmet detection events via Server-Sent Events (SSE)
 * @param onDetection Callback when new detections arrive
 * @param onError Callback on connection error
 * @returns Cleanup function to close the connection
 */
export function subscribeToHelmetEvents(
  onDetection: (detections: DetectionResult[]) => void,
  onError: (error: Error) => void
): () => void {
  try {
    const es = new EventSource(`${BASE_URL}/helmet/events`)

    es.onmessage = (event) => {
  try {
    const raw = JSON.parse(event.data)
    
    // รองรับทั้ง object เดียวและ array
    const items: DetectionEvent[] = Array.isArray(raw) ? raw : [raw]

    const mapped: DetectionResult[] = items.map((item) => ({
      id:             `trk_${item.motorcycle_track_id}_${Date.now()}`,
      timestamp:      new Date().toLocaleString("th-TH"),
      camera:         "กล้องหลัก",
      helmetStatus:   item.helmet_status === true ? "wearing" : "not-wearing",
      passengerCount: item.passenger_count ?? 1,
      violation:      item.violation ?? false,
    }))

    onDetection(mapped)
  } catch (parseError) {
    console.warn("Failed to parse SSE detection event:", parseError)
  }
}

    es.onerror = () => {
      console.warn("SSE connection error on /helmet/events")
      onError(new Error("SSE connection error"))
      es.close()
    }

    // Return cleanup function
    return () => {
      es.close()
    }
  } catch (error) {
    const err = error instanceof Error ? error : new Error("Unknown error")
    onError(err)
    return () => {}
  }
}

/**
 * Get the MJPEG stream URL for live video feed
 * @returns The stream URL
 */
export function getStreamUrl(): string {
  return `${BASE_URL}/helmet/stream`
}
