import { CAMERA_NAME } from "@/app/api/config"

export interface DetectionResultMock {
    id: string
    timestamp: string
    camera: string
    licensePlate: string
    helmetStatus: "wearing" | "not-wearing"
    passengerCount: number
    // confidence: number
    imageUrl?: string
}

export const mockDetections: DetectionResultMock[] = [
    {
        id: "1",
        timestamp: "2025-09-26 16:23:06",
        camera: `${CAMERA_NAME} - ตรวจจับผู้ขับขี่`,
        licensePlate: "6กฮ-4422",
        helmetStatus: "not-wearing",
        passengerCount: 2,
        // confidence: 95,
    },
    {
        id: "2",
        timestamp: "2025-09-26 16:23:06",
        camera: `${CAMERA_NAME} - ตรวจจับผู้ขับขี่`,
        licensePlate: "คง-5678",
        helmetStatus: "wearing",
        passengerCount: 1,
        // confidence: 98,
    },
    {
        id: "3",
        timestamp: "2025-09-26 16:23:06",
        camera: `${CAMERA_NAME} - ตรวจจับผู้ขับขี่`,
        licensePlate: "นม-9012",
        helmetStatus: "not-wearing",
        passengerCount: 3,
        // confidence: 92,
    },
]
