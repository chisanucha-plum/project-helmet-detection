"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  AlertTriangle,
  BikeIcon,
  Camera,
  CheckCircle,
  Clock,
  Eye,
  EyeOff,
  Users,
  Maximize,
  Minimize,
  AlertCircle,
  RotateCw,
} from "lucide-react"
import React, { useEffect, useState, useCallback } from "react"
import { useRealTimeDetections } from "@/hooks/useRealTimeDetections"
import { getStreamUrl } from "@/services/helmet-detection.service"

// Small clock component that updates every second. Kept isolated so the parent
// RealTimeMonitoring component does not re-render every tick.
function NowClock() {
  const [isMounted, setIsMounted] = useState(false)
  const [now, setNow] = useState<Date | null>(null)

  useEffect(() => {
    setIsMounted(true)
    setNow(new Date())
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  if (!isMounted || !now) {
    return <span suppressHydrationWarning>--:--:--</span>
  }

  return <>{now.toLocaleTimeString("th-TH")}</>
}

export function RealTimeMonitoring() {
  const { detections, isLoading, error, isRecording, setIsRecording } =
    useRealTimeDetections()
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [mjpegUrl, setMjpegUrl] = useState<string | undefined>(undefined)

  // Make toggleFullscreen stable so it doesn't change every render
  const toggleFullscreen = useCallback(() => {
    const containerElement = document.getElementById(
      "video-container"
    ) as HTMLDivElement | null
    if (!containerElement) return

    if (!document.fullscreenElement) {
      if (containerElement.requestFullscreen) {
        containerElement.requestFullscreen()
      } else if ((containerElement as any).webkitRequestFullscreen) {
        (containerElement as any).webkitRequestFullscreen()
      } else if ((containerElement as any).msRequestFullscreen) {
        (containerElement as any).msRequestFullscreen()
      }
    } else if (document.exitFullscreen) {
      document.exitFullscreen()
    } else if ((document as any).webkitExitFullscreen) {
      (document as any).webkitExitFullscreen()
    } else if ((document as any).msExitFullscreen) {
      (document as any).msExitFullscreen()
    }
  }, [])

  // Listen for fullscreen changes and keyboard shortcuts
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener("fullscreenchange", handleFullscreenChange)
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange)
    document.addEventListener("msfullscreenchange", handleFullscreenChange)

    // Add CSS for fullscreen
    const style = document.createElement("style")
    style.textContent = `
      #video-container:fullscreen {
        background: black;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      #video-container:-webkit-full-screen {
        background: black;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      #video-container:-moz-full-screen {
        background: black;
        display: flex;
        align-items: center;
        justify-content: center;
      }
    `
    document.head.appendChild(style)

    // Handle keyboard shortcuts
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (document.fullscreenElement) {
          if (document.exitFullscreen) {
            document.exitFullscreen()
          } else if ((document as any).webkitExitFullscreen) {
            (document as any).webkitExitFullscreen()
          }
        }
      }
      if (event.key.toLowerCase() === "f") {
        const mjpegEl = document.getElementById("mjpeg-stream")
        const container = document.getElementById("video-container")
        if (mjpegEl && container && !document.fullscreenElement) {
          if ((container as any).requestFullscreen) {
            (container as any).requestFullscreen()
          }
        }
      }
    }

    document.addEventListener("keydown", handleKeyPress)

    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange)
      document.removeEventListener("webkitfullscreenchange", handleFullscreenChange)
      document.removeEventListener("msfullscreenchange", handleFullscreenChange)
      document.removeEventListener("keydown", handleKeyPress)
      document.head.removeChild(style)
    }
  }, [])

  // Update stream URL when recording state changes
  useEffect(() => {
    if (isRecording) {
      setMjpegUrl(getStreamUrl())
    } else {
      setMjpegUrl(undefined)
    }
  }, [isRecording])

  const todayViolations = detections.filter(
    (d) => d.helmetStatus === "not-wearing"
  ).length
  const todayTotal = detections.length
  const complianceRate =
    todayTotal > 0
      ? Math.round(((todayTotal - todayViolations) / todayTotal) * 100)
      : 0

  return (
    <div className="space-y-6">
      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-red-900">เกิดข้อผิดพลาด</p>
              <p className="text-sm text-red-700">{error.message}</p>
            </div>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => window.location.reload()}
            className="gap-2 flex-shrink-0"
          >
            <RotateCw className="h-4 w-4" />
            ลองใหม่
          </Button>
        </div>
      )}

      {/* Header with Status */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-foreground">
            การตรวจสอบแบบ Real-time
          </h2>
          <p className="text-muted-foreground">
            อัปเดตล่าสุด: <NowClock />
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                isRecording
                  ? "bg-green-500 animate-pulse"
                  : "bg-gray-400"
              }`}
            ></div>
            <span className="text-sm text-muted-foreground">
              สถานะ: {isRecording ? "กำลังทำงาน" : "หยุดอยู่"}
            </span>
          </div>

          <Button
            variant={isRecording ? "destructive" : "default"}
            size="sm"
            onClick={() => setIsRecording(!isRecording)}
            className="gap-2"
            disabled={isLoading}
          >
            {isRecording ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
            {isRecording ? "หยุดบันทึก" : "เริ่มบันทึก"}
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <BikeIcon className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">มอเตอร์ไซค์ที่ตรวจพบ</p>
                <p className="text-2xl font-bold text-foreground">
                  {isLoading ? "-" : todayTotal}
                </p>
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
                  {isLoading
                    ? "-"
                    : detections.filter((d) => d.passengerCount > 2).length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                <AlertTriangle className="h-5 w-5 text-red-600" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">การกระทำผิดวันนี้</p>
                <p className="text-2xl font-bold text-foreground">
                  {isLoading ? "-" : todayViolations}
                </p>
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
                <p className="text-2xl font-bold text-foreground">
                  {isLoading ? "-" : complianceRate}%
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Video Feeds */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">
        {/* Camera 1 — ใช้ 2 คอลัมน์จาก 3 */}
        <div className="lg:col-span-2 flex flex-col">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Camera className="h-5 w-5" />
              Camera 1 - ตรวจจับผู้ขับขี่
              <Badge variant="secondary" className="ml-auto">
                Live
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div 
              id="video-container"
              className={`aspect-video bg-muted rounded-lg flex items-center justify-center relative overflow-hidden group ${
                isFullscreen ? 'fixed inset-0 z-50 bg-black rounded-none aspect-auto' : ''
              }`}
            >
              {mjpegUrl ? (
                <img 
                  id="mjpeg-stream" 
                  src={mjpegUrl} 
                  alt="Live MJPEG" 
                  className={`absolute inset-0 w-full h-full object-cover ${
                    isFullscreen ? 'object-contain' : 'object-cover'
                  }`} 
                />
              ) : (
                <div className="relative z-10 text-center">
                  <Camera className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                  <span className="text-muted-foreground">Live Video Feed</span>
                </div>
              )}
              
              {/* Recording indicator */}
              <div className="absolute top-3 left-3 flex items-center gap-1 bg-red-500 text-white px-2 py-1 rounded text-xs z-20">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                REC
              </div>

              {/* Fullscreen button */}
              {mjpegUrl && (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={toggleFullscreen}
                  className="absolute top-3 right-3 z-20 transition-opacity"
                  title={isFullscreen ? "ออกจากเต็มจอ" : "ดูเต็มจอ"}
                >
                  {isFullscreen ? (
                    <Minimize className="h-4 w-4" />
                  ) : (
                    <Maximize className="h-4 w-4" />
                  )}
                </Button>
              )}

              {/* Fullscreen controls */}
              {isFullscreen && (
                <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-20">
                  <div className="flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded-lg px-4 py-2 text-white">
                    <div className="flex items-center gap-2 text-sm">
                      <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                      <span>Camera 1 - ตรวจจับผู้ขับขี่</span>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={toggleFullscreen}
                      className="text-white hover:text-white hover:bg-white/20"
                    >
                      <Minimize className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
        </div>

        <div className="lg:col-span-1 flex flex-col">
        <Card className="flex flex-col flex-1">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Camera className="h-4 w-4" />
              Camera 2 - ป้ายทะเบียน
              <Badge variant="secondary" className="ml-auto text-xs">
                Live
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 flex flex-col">
            <div className="flex-1 bg-muted rounded-lg flex flex-col items-center justify-center min-h-0 overflow-hidden">
              <img
                src="/f.jpg"
                alt="Camera 2 placeholder"
                className="w-full h-full object-cover rounded-lg"
              />
            </div>
          </CardContent>
        </Card>
        </div>
      </div>

      {/* Detection Results */}
      <Card id="detection-results">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            ผลการตรวจจับล่าสุด
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {isLoading && detections.length === 0 ? (
              <div className="text-center py-8">
                <div className="inline-block">
                  <div className="w-8 h-8 border-4 border-muted border-t-foreground rounded-full animate-spin mb-3"></div>
                </div>
                <p className="text-muted-foreground">กำลังโหลดข้อมูล...</p>
              </div>
            ) : detections.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Camera className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>ไม่มีการตรวจจับในขณะนี้</p>
              </div>
            ) : (
              detections.map((detection) => (
                <div
                  key={detection.id}
                  className="flex flex-col sm:flex-row sm:items-center justify-between p-4 bg-muted rounded-lg gap-3"
                >
                  <div className="flex items-center gap-4">
                    <div>
                      <div className="text-sm text-muted-foreground">
                        {detection.timestamp}
                      </div>

                      <div className="flex items-center gap-2 mt-1">
                        {detection.helmetStatus === "wearing" ? (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-red-500" />
                        )}
                        <span
                          className={`text-sm font-medium ${
                            detection.helmetStatus === "wearing"
                              ? "text-green-600"
                              : "text-red-600"
                          }`}
                        >
                          {detection.helmetStatus === "wearing"
                            ? "สวมหมวกกันน็อค"
                            : "ไม่สวมหมวกกันน็อค"}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-3 text-sm">
                    <div className="flex items-center gap-1">
                      <Users className="h-4 w-4 text-muted-foreground" />
                      <span>{detection.passengerCount} คน</span>
                      {detection.passengerCount > 2 && (
                        <Badge variant="destructive" className="ml-1 text-xs">
                          เกินกำหนด
                        </Badge>
                      )}
                    </div>

                    {detection.violation && (
                      <Badge variant="destructive" className="text-xs">
                        ฝ่าฝืน
                      </Badge>
                    )}

                    <span className="text-xs text-muted-foreground">
                      {detection.camera}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
