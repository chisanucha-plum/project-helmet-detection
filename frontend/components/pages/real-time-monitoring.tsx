"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertTriangle, BikeIcon, Camera, Car, CheckCircle, Clock, Eye, EyeOff, Users } from "lucide-react"
import { useEffect, useState } from "react"

import { mockDetections } from "@/mocks/realTimeMocks"

interface DetectionResult {
  id: string
  timestamp: string
  camera: string
  licensePlate: string
  helmetStatus: "wearing" | "not-wearing"
  passengerCount: number
  confidence: number
  imageUrl?: string
}

export function RealTimeMonitoring() {
  const [detections] = useState<DetectionResult[]>(mockDetections)
  const [isRecording, setIsRecording] = useState(true)
  const [currentTime, setCurrentTime] = useState(new Date())

  // Update current time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  // Show MJPEG stream by setting the image src. Use env var if provided; otherwise use relative path.
  const [mjpegUrl, setMjpegUrl] = useState<string | undefined>(undefined)

  useEffect(() => {
    // Old fetch/reader approach (commented out):
    // const res = await fetch(url, { signal: abortController.signal })
    // const reader = res.body.getReader()
    // while (true) {
    //   const { done, value } = await reader.read()
    //   if (done) break
    //   const chunk = decoder.decode(value, { stream: true })
    //   buffer += chunk
    //   // Parse multipart boundary and extract JSON payload
    //   // if (buf.trim()) {
    //   //   try {
    //   //     const payload = JSON.parse(buf.trim())
    //   //     const mapped: DetectionResult = {
    //   //       id: String(payload.id ?? Date.now()),
    //   //       timestamp: payload.timestamp ?? new Date().toLocaleTimeString('th-TH'),
    //   //       camera: payload.camera ?? 'กล้องไม่ทราบ',
    //   //       licensePlate: payload.license_plate ?? payload.licensePlate ?? 'ไม่ทราบ',
    //   //       helmetStatus: payload.helmet_status === false ? 'not-wearing' : 'wearing',
    //   //       passengerCount: payload.person_count ?? payload.passengerCount ?? 1,
    //   //       confidence: payload.confidence ?? 0,
    //   //       imageUrl: payload.image_url ?? payload.imageUrl,
    //   //     }
    //   //     setDetections(prev => [mapped, ...prev.slice(0, 10)])
    //   //   } catch (e) { console.error('Parse error:', e) }
    //   // }
    //   // const detection = JSON.parse(payloadJson)
    //   // setDetections(prev => [detection, ...prev.slice(0, 99)])
    // }
    
    // New simple approach: use <img> with MJPEG URL
    const base = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
    const url = `${base}/helmet/detect`
    console.log('MJPEG Stream URL:', url)
    if (isRecording) setMjpegUrl(url)
    else setMjpegUrl(undefined)
  }, [isRecording])

  const todayViolations = detections.filter((d) => d.helmetStatus === "not-wearing").length
  const todayTotal = detections.length
  const complianceRate = todayTotal > 0 ? Math.round(((todayTotal - todayViolations) / todayTotal) * 100) : 0

  return (
    <div className="space-y-6">
      {/* Header with Status */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-foreground">การตรวจสอบแบบ Real-time</h2>
          <p className="text-muted-foreground">อัปเดตล่าสุด: {currentTime.toLocaleTimeString("th-TH")}</p>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-muted-foreground">สถานะ: เชื่อมต่อแล้ว</span>
          </div>

          <Button
            variant={isRecording ? "destructive" : "default"}
            size="sm"
            onClick={() => setIsRecording(!isRecording)}
            className="gap-2"
          >
            {isRecording ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            {isRecording ? "หยุดบันทึก" : "เริ่มบันทึก"}
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                <AlertTriangle className="h-5 w-5 text-red-600" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">การกระทำผิดวันนี้</p>
                <p className="text-2xl font-bold text-foreground">{todayViolations}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                <CheckCircle className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">อัตราการปฏิบัติตาม</p>
                <p className="text-2xl font-bold text-foreground">{complianceRate}%</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <BikeIcon className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">มอเตอร์ไซค์ที่ตรวจพบ</p>
                <p className="text-2xl font-bold text-foreground">{todayTotal}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-orange-100 rounded-lg flex items-center justify-center">
                <Users className="h-5 w-5 text-orange-600" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">นั่งเกิน 2 คน</p>
                <p className="text-2xl font-bold text-foreground">
                  {detections.filter((d) => d.passengerCount > 2).length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Video Feeds */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Camera className="h-5 w-5" />
              กล้องหลัก - ตรวจจับผู้ขับขี่
              <Badge variant="secondary" className="ml-auto">
                Live
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center relative overflow-hidden">
              {mjpegUrl ? (
                <img id="mjpeg-stream" src={mjpegUrl} alt="Live MJPEG" className="absolute inset-0 w-full h-full object-cover" />
              ) : (
                <div className="relative z-10 text-center">
                  <Camera className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                  <span className="text-muted-foreground">Live Video Feed</span>
                </div>
              )}
              <div className="absolute top-3 right-3 flex items-center gap-1 bg-red-500 text-white px-2 py-1 rounded text-xs">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                REC
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Camera className="h-5 w-5" />
              กล้องรอง - ตรวจจับป้ายทะเบียน
              <Badge variant="secondary" className="ml-auto">
                Live
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="aspect-video bg-muted rounded-lg flex items-center justify-center relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-green-500/20 to-blue-500/20"></div>
              <div className="relative z-10 text-center">
                <Camera className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                <span className="text-muted-foreground">Live Video Feed</span>
              </div>
              <div className="absolute top-3 right-3 flex items-center gap-1 bg-red-500 text-white px-2 py-1 rounded text-xs">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                REC
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detection Results */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            ผลการตรวจจับล่าสุด
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {detections.map((detection) => (
              <div
                key={detection.id}
                className="flex flex-col sm:flex-row sm:items-center justify-between p-4 bg-muted rounded-lg gap-3"
              >
                <div className="flex items-center gap-4">
                  <div className="text-sm text-muted-foreground min-w-[70px]">{detection.timestamp}</div>

                  <div className="flex items-center gap-2">
                    {detection.helmetStatus === "wearing" ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-red-500" />
                    )}
                    <span
                      className={`text-sm font-medium ${detection.helmetStatus === "wearing" ? "text-green-600" : "text-red-600"
                        }`}
                    >
                      {detection.helmetStatus === "wearing" ? "สวมหมวกกันน็อค" : "ไม่สวมหมวกกันน็อค"}
                    </span>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-3 text-sm">
                  <div className="flex items-center gap-1">
                    <Car className="h-4 w-4 text-muted-foreground" />
                    <span className="font-mono">{detection.licensePlate}</span>
                  </div>

                  <div className="flex items-center gap-1">
                    <Users className="h-4 w-4 text-muted-foreground" />
                    <span>{detection.passengerCount} คน</span>
                    {detection.passengerCount > 2 && (
                      <Badge variant="destructive" className="ml-1 text-xs">
                        เกินกำหนด
                      </Badge>
                    )}
                  </div>

                  <Badge variant="outline" className="text-xs">
                    {detection.confidence}% แม่นยำ
                  </Badge>

                  <span className="text-xs text-muted-foreground">{detection.camera}</span>
                </div>
              </div>
            ))}

            {detections.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <Camera className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>ไม่มีการตรวจจับในขณะนี้</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
