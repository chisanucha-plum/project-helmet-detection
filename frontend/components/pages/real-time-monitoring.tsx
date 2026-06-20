"use client"

import { useEffect, useState, useCallback, useMemo } from "react"
import { AlertTriangle, BikeIcon, Camera, CheckCircle, Clock, Eye, EyeOff, Users, Maximize, Minimize, AlertCircle, RotateCw } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useRealTimeDetections } from "@/hooks/useRealTimeDetections"
import { useLanguage } from "@/hooks/useLanguage"
import { getStreamUrl } from "@/services/helmet-detection.service"

const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"

function NowClock() {
  const [isMounted, setIsMounted] = useState(false)
  const [now, setNow] = useState<Date | null>(null)

  useEffect(() => {
    setIsMounted(true)
    setNow(new Date())
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  if (!isMounted || !now) return <span suppressHydrationWarning>--:--:--</span>
  return <>{now.toLocaleTimeString("th-TH")}</>
}

function StatsCards({ detections, isLoading, t }: { detections: any[]; isLoading: boolean; t: (key: string) => string }) {
  const stats = useMemo(() => {
    const violations = detections.filter((d) => d.helmetStatus === "not-wearing").length
    const total = detections.length
    const compliance = total > 0 ? Math.round(((total - violations) / total) * 100) : 0
    const overCapacity = detections.filter((d) => d.passengerCount > 2).length
    return { violations, total, compliance, overCapacity }
  }, [detections])

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard icon={BikeIcon} label={t("stats.motorcyclesDetected")} value={isLoading ? "-" : stats.total} bgColor="bg-blue-100" iconColor="text-blue-600" />
      <StatCard icon={Users} label={t("stats.overCapacity")} value={isLoading ? "-" : stats.overCapacity} bgColor="bg-orange-100" iconColor="text-orange-600" />
      <StatCard icon={AlertTriangle} label={t("stats.violations")} value={isLoading ? "-" : stats.violations} bgColor="bg-red-100" iconColor="text-red-600" />
      <StatCard icon={CheckCircle} label={t("stats.complianceRate")} value={isLoading ? "-" : `${stats.compliance}%`} bgColor="bg-green-100" iconColor="text-green-600" />
    </div>
  )
}

function StatCard({ icon: Icon, label, value, bgColor, iconColor }: { icon: any; label: string; value: any; bgColor: string; iconColor: string }) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 ${bgColor} rounded-lg flex items-center justify-center`}>
            <Icon className={`h-5 w-5 ${iconColor}`} />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">{label}</p>
            <p className="text-2xl font-bold text-foreground">{value}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

const DetectionItem = ({ detection, t }: { detection: any; t: (key: string) => string }) => {
  const [showModal, setShowModal] = useState(false)

  return (
    <>
      <div className="flex flex-row gap-3 p-3 bg-muted rounded-lg border border-border/50">
        {detection.framePath ? (
          <div 
            className="flex-shrink-0 w-20 h-20 rounded-md overflow-hidden bg-background border border-border cursor-pointer hover:ring-2 hover:ring-primary transition-all"
            onClick={() => setShowModal(true)}
          >
            <img src={`${BASE_URL}/helmet/frame/${detection.framePath}`} alt={`Detection ${detection.id}`} className="w-full h-full object-cover" loading="lazy" />
          </div>
        ) : (
          <div className="flex-shrink-0 w-20 h-20 rounded-md bg-muted border border-border flex items-center justify-center">
            <Camera className="h-6 w-6 text-muted-foreground" />
          </div>
        )}
        <div className="flex-1 flex flex-col justify-between gap-2 min-w-0">
          <div>
            <div className="text-xs text-muted-foreground">{detection.timestamp}</div>
            <div className="flex items-center gap-2 mt-1">
              {detection.helmetStatus === "wearing" ? <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" /> : <AlertTriangle className="h-4 w-4 text-red-500 flex-shrink-0" />}
              <span className={`text-sm font-semibold truncate ${detection.helmetStatus === "wearing" ? "text-green-600" : "text-red-600"}`}>
                {detection.helmetStatus === "wearing" ? t("detection.wearingHelmet") : t("detection.notWearingHelmet")}
              </span>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <div className="flex items-center gap-1">
              <Users className="h-3 w-3 text-muted-foreground" />
              <span>{detection.passengerCount} {t("detection.passengers")}</span>
              {detection.passengerCount > 2 && <Badge variant="destructive" className="ml-1 text-xs px-1 py-0">{t("detection.overCapacityBadge")}</Badge>}
            </div>
            {detection.violation && <Badge variant="destructive" className="text-xs px-1.5 py-0">{t("detection.violation")}</Badge>}
            <span className="text-xs text-muted-foreground">{detection.camera}</span>
          </div>
        </div>
      </div>

      {showModal && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4" onClick={() => setShowModal(false)}>
          <div className="relative max-w-4xl max-h-[90vh] w-full">
            <button onClick={() => setShowModal(false)} className="absolute top-2 right-2 bg-black/50 text-white rounded-full p-2 hover:bg-black/70 z-10">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <img src={`${BASE_URL}/helmet/frame/${detection.framePath}`} alt="Detection Frame" className="w-full h-full object-contain rounded-lg" onClick={(e) => e.stopPropagation()} />
          </div>
        </div>
      )}
    </>
  )
}

export function RealTimeMonitoring() {
  const { detections, isLoading, error, isRecording, setIsRecording } = useRealTimeDetections()
  const { t } = useLanguage("en")
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [mjpegUrl, setMjpegUrl] = useState<string | undefined>(undefined)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)

  const toggleFullscreen = useCallback(() => {
    const container = document.getElementById("video-container") as HTMLDivElement | null
    if (!container) return
    if (!document.fullscreenElement) container.requestFullscreen?.()
    else document.exitFullscreen?.()
  }, [])

  useEffect(() => {
    const handleChange = () => setIsFullscreen(!!document.fullscreenElement)
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === "Escape" && document.fullscreenElement) document.exitFullscreen?.()
      else if (e.key.toLowerCase() === "f" && !document.fullscreenElement) toggleFullscreen()
    }
    document.addEventListener("fullscreenchange", handleChange)
    document.addEventListener("keydown", handleKeyPress)
    return () => {
      document.removeEventListener("fullscreenchange", handleChange)
      document.removeEventListener("keydown", handleKeyPress)
    }
  }, [toggleFullscreen])

  useEffect(() => {
    setMjpegUrl(isRecording ? getStreamUrl() : undefined)
  }, [isRecording])

  return (
    <div className="space-y-6">
      {selectedImage && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4" onClick={() => setSelectedImage(null)}>
          <div className="relative max-w-4xl max-h-[90vh] w-full">
            <button onClick={() => setSelectedImage(null)} className="absolute top-2 right-2 bg-black/50 text-white rounded-full p-2 hover:bg-black/70 z-10">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <img src={selectedImage} alt="Detection Frame" className="w-full h-full object-contain rounded-lg" onClick={(e) => e.stopPropagation()} />
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-red-900">{t("errors.errorOccurred")}</p>
              <p className="text-sm text-red-700">{error.message}</p>
            </div>
          </div>
          <Button size="sm" variant="outline" onClick={() => window.location.reload()} className="gap-2">
            <RotateCw className="h-4 w-4" />
            {t("buttons.retry")}
          </Button>
        </div>
      )}

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-foreground">{t("header.title")}</h2>
          <p className="text-muted-foreground">
            {t("header.lastUpdate")} <NowClock />
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isRecording ? "bg-green-500 animate-pulse" : "bg-gray-400"}`} />
            <span className="text-sm text-muted-foreground">{t("status." + (isRecording ? "running" : "stopped"))}</span>
          </div>
          <Button variant={isRecording ? "destructive" : "default"} size="sm" onClick={() => setIsRecording(!isRecording)} disabled={isLoading} className="gap-2">
            {isRecording ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            {t("buttons." + (isRecording ? "stopRecording" : "startRecording"))}
          </Button>
        </div>
      </div>

      <StatsCards detections={detections} isLoading={isLoading} t={t} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Camera className="h-5 w-5" />
                {t("camera.camera1")}
                <Badge variant="secondary" className="ml-auto">
                  {t("camera.live")}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div id="video-container" className={`aspect-video bg-muted rounded-lg flex items-center justify-center relative overflow-hidden ${isFullscreen ? "fixed inset-0 z-50 bg-black rounded-none aspect-auto" : ""}`}>
                {mjpegUrl ? (
                  <img id="mjpeg-stream" src={mjpegUrl} alt="Live MJPEG" className={`absolute inset-0 w-full h-full ${isFullscreen ? "object-contain" : "object-cover"}`} />
                ) : (
                  <div className="z-10 text-center">
                    <Camera className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                    <span className="text-muted-foreground">{t("camera.liveVideoFeed")}</span>
                  </div>
                )}
                <div className="absolute top-3 left-3 flex items-center gap-1 bg-red-500 text-white px-2 py-1 rounded text-xs z-20">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  {t("recording")}
                </div>
                {mjpegUrl && (
                  <Button size="sm" variant="secondary" onClick={toggleFullscreen} className="absolute top-3 right-3 z-20">
                    {isFullscreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
                  </Button>
                )}
                {isFullscreen && (
                  <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20">
                    <div className="flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded-lg px-4 py-2 text-white">
                      <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                      <span className="text-sm">{t("camera.camera1")}</span>
                      <Button size="sm" variant="ghost" onClick={toggleFullscreen} className="text-white hover:bg-white/20">
                        <Minimize className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
        <div className="lg:col-span-1">
          <Card className="flex flex-col h-full">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Camera className="h-4 w-4" />
                {t("camera.camera2")}
                <Badge variant="secondary" className="ml-auto text-xs">
                  {t("camera.live")}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1">
              <img src="/f.jpg" alt="Camera 2" className="w-full h-full object-cover rounded-lg" />
            </CardContent>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            {t("detection.latestResults")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {isLoading && detections.length === 0 ? (
              <div className="text-center py-8">
                <div className="w-8 h-8 border-4 border-muted border-t-foreground rounded-full animate-spin mx-auto mb-3" />
                <p className="text-muted-foreground">{t("detection.loading")}</p>
              </div>
            ) : detections.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Camera className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>{t("detection.noDetections")}</p>
              </div>
            ) : (
              detections.map((detection) => <DetectionItem key={detection.id} detection={detection} t={t} />)
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}


