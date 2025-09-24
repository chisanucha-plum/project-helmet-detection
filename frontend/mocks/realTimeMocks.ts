export interface DetectionResultMock {
    id: string
    timestamp: string
    camera: string
    licensePlate: string
    helmetStatus: "wearing" | "not-wearing"
    passengerCount: number
    confidence: number
    imageUrl?: string
}

export const mockDetections: DetectionResultMock[] = [
    {
        id: "1",
        timestamp: "14:35:42",
        camera: "กล้องหลัก - ประตูทางเข้า",
        licensePlate: "6กฮ-4422",
        helmetStatus: "not-wearing",
        passengerCount: 2,
        confidence: 95,
    },
    {
        id: "2",
        timestamp: "14:33:15",
        camera: "กล้องรอง - ลานจอดรถ",
        licensePlate: "คง-5678",
        helmetStatus: "wearing",
        passengerCount: 1,
        confidence: 98,
    },
    {
        id: "3",
        timestamp: "14:30:28",
        camera: "กล้องหลัก - ประตูทางเข้า",
        licensePlate: "นม-9012",
        helmetStatus: "not-wearing",
        passengerCount: 3,
        confidence: 92,
    },
]
