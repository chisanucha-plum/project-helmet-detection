"use client"

import { useEffect, useState, useMemo } from "react"
import type React from "react"
import {
  AlertCircle,
  AlertTriangle,
  BikeIcon,
  Camera,
  CheckCircle,
  Clock,
  Eye,
  EyeOff,
  RotateCw,
  Users,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useRealTimeDetections } from "@/hooks/useRealTimeDetections"
import { useLanguage } from "@/hooks/useLanguage"
import { getStreamUrl } from "@/services/helmet-detection.service"
import { VideoStream } from "@/components/real-time/VideoStream"
import { DetectionList } from "@/components/real-time/DetectionList"
import { Badge } from "@/components/ui/badge"
import type { DetectionResult } from "@/types/detection.types"

function NowClock() {
  const [now, setNow] = useState<Date | null>(null)

  useEffect(() => {
    setNow(new Date())
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  if (!now) return <span suppressHydrationWarning>--:--:--</span>
  return <>{now.toLocaleTimeString("th-TH")}</>
}

function StatsCards({ detections, isLoading, t }: { detections: DetectionResult[]; isLoading: boolean; t: (key: string) => string }) {
  const stats = useMemo(() => {
    const violations = detections.filter((d) => d.helmetStatus === "not-wearing").length
    const total = detections.length
    const compliance = total > 0 ? Math.round(((total - violations) / total) * 100) : 0
    const overCapacity = detections.filter((d) => d.passengerCount > 2).length
    return { violations, total, compliance, overCapacity }
  }, [detections])

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard icon={BikeIcon} label={t("stats.motorcyclesDetected")} value={isLoading ? "-" : stats.total} bgColor="bg-info" iconColor="text-info-foreground" />
      <StatCard icon={Users} label={t("stats.overCapacity")} value={isLoading ? "-" : stats.overCapacity} bgColor="bg-warning" iconColor="text-warning-foreground" />
      <StatCard icon={AlertTriangle} label={t("stats.violations")} value={isLoading ? "-" : stats.violations} bgColor="bg-critical" iconColor="text-critical-foreground" />
      <StatCard icon={CheckCircle} label={t("stats.complianceRate")} value={isLoading ? "-" : `${stats.compliance}%`} bgColor="bg-success" iconColor="text-success-foreground" />
    </div>
  )
}

function StatCard({ icon: Icon, label, value, bgColor, iconColor }: { icon: React.ComponentType<{ className?: string }>; label: string; value: string | number; bgColor: string; iconColor: string }) {
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

export function RealTimeMonitoring() {
  const { detections, isLoading, error, isRecording, setIsRecording } = useRealTimeDetections()
  const { t } = useLanguage("en")
  const [mjpegUrl, setMjpegUrl] = useState<string | undefined>(undefined)

  useEffect(() => {
    setMjpegUrl(isRecording ? getStreamUrl() : undefined)
  }, [isRecording])

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-critical border border-critical-foreground/20 rounded-lg p-4 flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-critical-foreground flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-foreground">{t("errors.errorOccurred")}</p>
              <p className="text-sm text-muted-foreground">{error.message}</p>
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
            <div className={`w-3 h-3 rounded-full ${isRecording ? "bg-success-foreground animate-pulse" : "bg-muted-foreground/40"}`} />
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
              <VideoStream
                mjpegUrl={mjpegUrl}
              />
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
          {isLoading && detections.length === 0 ? (
            <div className="text-center py-8">
              <div className="w-8 h-8 border-4 border-muted border-t-foreground rounded-full animate-spin mx-auto mb-3" />
              <p className="text-muted-foreground">{t("detection.loading")}</p>
            </div>
          ) : (
            <DetectionList detections={detections} t={t} />
          )}
        </CardContent>
      </Card>
    </div>
  )
}
