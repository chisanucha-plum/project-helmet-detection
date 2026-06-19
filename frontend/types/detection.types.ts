/**
 * Centralized types for helmet detection feature
 */

export interface DetectionResult {
  id: string
  timestamp: string
  camera: string
  helmetStatus: "wearing" | "not-wearing"
  passengerCount: number
  violation: boolean
}

export interface DetectionHistoryItem {
  id: string
  timestamp?: string
  helmet_status: boolean
  passenger_count?: number
  violation?: boolean
}

export interface DetectionEvent {
  motorcycle_track_id: string
  helmet_status: boolean
  passenger_count?: number
  violation?: boolean
}

export interface UseRealTimeDetectionsReturn {
  detections: DetectionResult[]
  isLoading: boolean
  error: Error | null
  isRecording: boolean
  setIsRecording: (value: boolean) => void
}
