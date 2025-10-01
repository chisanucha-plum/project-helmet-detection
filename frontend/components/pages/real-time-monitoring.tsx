"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertTriangle, BikeIcon, Camera, Car, CheckCircle, Clock, Eye, EyeOff, Users, Maximize, Minimize } from "lucide-react"
import React, { useEffect, useState, useCallback } from "react"

// We fetch real detection history from the backend instead of using mocks

// Small clock component that updates every second. Kept isolated so the parent
// RealTimeMonitoring component does not re-render every tick.
function NowClock() {
  const [now, setNow] = useState(() => new Date())

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  return <>{now.toLocaleTimeString('th-TH')}</>
}

interface DetectionResult {
  id: string
  timestamp: string
  camera: string
  licensePlate: string
  helmetStatus: "wearing" | "not-wearing"
  passengerCount: number
  // confidence: number
  imageUrl?: string
}

export function RealTimeMonitoring() {
  const [detections, setDetections] = useState<DetectionResult[]>([])
  const [isRecording, setIsRecording] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Handle fullscreen functionality
  // Make toggleFullscreen stable so it doesn't change every render
  const toggleFullscreen = useCallback(() => {
    const containerElement = document.getElementById('video-container') as HTMLDivElement | null
    if (!containerElement) return

    if (!document.fullscreenElement) {
      if (containerElement.requestFullscreen) {
        containerElement.requestFullscreen()
      } else if ((containerElement as any).webkitRequestFullscreen) {
        (containerElement as any).webkitRequestFullscreen()
      } else if ((containerElement as any).msRequestFullscreen) {
        (containerElement as any).msRequestFullscreen()
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
      } else if ((document as any).webkitExitFullscreen) {
        (document as any).webkitExitFullscreen()
      } else if ((document as any).msExitFullscreen) {
        (document as any).msExitFullscreen()
      }
    }
  }, [])

  // Listen for fullscreen changes and keyboard shortcuts. We avoid closing over
  // changing state (like isFullscreen or mjpegUrl) by querying the DOM/document
  // directly when handling keys.
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange)
    document.addEventListener('msfullscreenchange', handleFullscreenChange)

    // Add CSS for fullscreen
    const style = document.createElement('style')
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

    // Handle keyboard shortcuts without reading stale closures
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (document.fullscreenElement) {
          if (document.exitFullscreen) {
            document.exitFullscreen()
          } else if ((document as any).webkitExitFullscreen) {
            (document as any).webkitExitFullscreen()
          }
        }
      }
      if (event.key.toLowerCase() === 'f') {
        // If the MJPEG element exists, toggle fullscreen on its container
        const mjpegEl = document.getElementById('mjpeg-stream')
        const container = document.getElementById('video-container')
        if (mjpegEl && container && !document.fullscreenElement) {
          if ((container as any).requestFullscreen) {
            (container as any).requestFullscreen()
          }
        }
      }
    }

    document.addEventListener('keydown', handleKeyPress)

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange)
      document.removeEventListener('msfullscreenchange', handleFullscreenChange)
      document.removeEventListener('keydown', handleKeyPress)
      document.head.removeChild(style)
    }
  }, [])

  // NOTE: we move the per-second clock into a small component below so the
  // whole page doesn't re-render every second.

  // Show MJPEG stream by setting the image src. Use env var if provided; otherwise use relative path.
  const [mjpegUrl, setMjpegUrl] = useState<string | undefined>(undefined)

  // Poll backend for history records and map them to DetectionResult
  useEffect(() => {
    let mounted = true
    const controller = new AbortController()
    const base = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

    const mapHistoryToDetection = (h: any): DetectionResult => ({
      id: h.id ?? `id_${Math.random().toString(36).slice(2, 9)}`,
      timestamp: h.timestamp ?? new Date().toLocaleString('th-TH'),
      camera: 'กล้องหลัก',
      licensePlate: '',
      helmetStatus: h.helmet_status === true ? 'wearing' : 'not-wearing',
      passengerCount: typeof h.passenger_count === 'number' ? h.passenger_count : 1,
      imageUrl: undefined,
    })

    const fetchHistory = async () => {
      try {
        const res = await fetch(`${base}/helmet/history?limit=20`, { signal: controller.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        if (!mounted) return
        const items = Array.isArray(data) ? data.map(mapHistoryToDetection) : []
        setDetections(items)
      } catch (e) {
        // ignore abort errors
        if ((e as any).name === 'AbortError') return
        console.warn('Failed to fetch history:', e)
      }
    }

    fetchHistory()
    const t = setInterval(fetchHistory, 5000)
    return () => {
      mounted = false
      controller.abort()
      clearInterval(t)
    }
  }, [])

  useEffect(() => {
    
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
          <p className="text-muted-foreground">อัปเดตล่าสุด: <NowClock /></p>
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
                      <span>กล้องหลัก - ตรวจจับผู้ขับขี่</span>
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
            <div 
              id="video-container-2"
              className="aspect-video bg-muted rounded-lg flex items-center justify-center relative overflow-hidden group"
            >
              <img 
                id="static-image" 
                src="/f.jpg" 
                alt="License Plate Detection" 
                className="absolute inset-0 w-full h-full object-cover" 
              />
              
              {/* Recording indicator */}
              <div className="absolute top-3 left-3 flex items-center gap-1 bg-red-500 text-white px-2 py-1 rounded text-xs z-20">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                REC
              </div>

              {/* Fullscreen button */}
              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  const img = document.getElementById('static-image') as HTMLImageElement
                  if (img.requestFullscreen) {
                    img.requestFullscreen()
                  } else if ((img as any).webkitRequestFullscreen) {
                    (img as any).webkitRequestFullscreen()
                  } else if ((img as any).msRequestFullscreen) {
                    (img as any).msRequestFullscreen()
                  }
                }}
                className="absolute top-3 right-3 z-20 transition-opacity"
                title="ดูเต็มจอ"
              >
                <Maximize className="h-4 w-4" />
              </Button>
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

                  {/* <Badge variant="outline" className="text-xs">
                    {detection.confidence}% แม่นยำ
                  </Badge> */}

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
