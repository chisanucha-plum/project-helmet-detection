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
  framePath?: string  // Optional path to saved frame image
}

export interface DetectionHistoryItem {
  id: string
  timestamp?: string
  helmet_status: boolean
  passenger_count?: number
  violation?: boolean
  frame_path?: string  // Optional path to saved frame image
}

export interface DetectionEvent {
  motorcycle_track_id: string
  helmet_status: boolean
  passenger_count?: number
  violation?: boolean
  frame_path?: string
}

export interface UseRealTimeDetectionsReturn {
  detections: DetectionResult[]
  isLoading: boolean
  error: Error | null
  isRecording: boolean
  setIsRecording: (value: boolean) => void
}
