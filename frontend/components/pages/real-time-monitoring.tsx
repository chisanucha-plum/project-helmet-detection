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
  MapPin,
  RotateCw,
  X,
  Users,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { toast } from "sonner"
import { useRealTimeDetections } from "@/hooks/useRealTimeDetections"
import { useLanguage } from "@/hooks/useLanguage"
import { loadDisplayPrefs } from "@/lib/app-settings"
import { playViolationBeep } from "@/lib/alert-sound"
import { getStreamUrl } from "@/services/helmet-detection.service"
import { DetectionList } from "@/components/real-time/DetectionList"
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

/** Upper bound for the SSE/history buffer regardless of display settings */
const MAX_SSE_BUFFER = 50

/** Camera location on campus_map.png, as % of image size (building S13) */
const CAMERA_LOCATION = { x: 29.1, y: 6.1, label: "S13" }

/** Pin gradient by violation rate */
const LEVEL_STYLES = {
  high: "linear-gradient(135deg,#E8543E,#C43D2C)", // > 30%
  mid: "linear-gradient(135deg,#E0A23D,#B87F26)", // 10-30%
  low: "linear-gradient(135deg,#2F8F63,#22714D)", // < 10%
} as const

type LevelKey = keyof typeof LEVEL_STYLES

/** Pin color level from violation rate (0-100) */
function levelFromRate(rate: number): LevelKey {
  if (rate > 30) return "high"
  if (rate >= 10) return "mid"
  return "low"
}

function CampusMap({ t, mjpegUrl }: { t: (key: string) => string; mjpegUrl?: string }) {
  const level = levelFromRate(0) // TODO: wire real violation rate

  // Hover preview — shows instantly on pin enter; popup is a DOM child of the
  // pin wrapper, so moving the cursor onto it keeps the popup open.
  const [showPreview, setShowPreview] = useState(false)

  const [expanded, setExpanded] = useState(false)

  // ESC closes fullscreen (stream <img> below stays mounted, so closing never refetches)
  useEffect(() => {
    if (!expanded) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(false)
    }
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [expanded])

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-lg">
          <MapPin className="h-5 w-5" />
          {t("camera.campusMap")}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <div className="relative">
          <img src="/campus_map.png" alt="KMUTT Bangmod campus map" className="w-full h-auto rounded-md" />
          <div
            className="pin-wrap"
            style={{ left: `${CAMERA_LOCATION.x}%`, top: `${CAMERA_LOCATION.y}%` }}
            onMouseEnter={() => setShowPreview(true)}
            onMouseLeave={() => setShowPreview(false)}
          >
            <button
              type="button"
              className="pin cursor-pointer"
              style={{ background: LEVEL_STYLES[level] }}
              onClick={() => setExpanded(true)}
              aria-label={t("camera.livePreview")}
            >
              <div className="pin-inner">{CAMERA_LOCATION.label}</div>
            </button>
          </div>

          {/* Single persistent stream layer — hover popup and fullscreen share one <img>,
              so toggling between them never opens a second MJPEG connection. */}
          <div
            className={
              expanded
                ? "fixed inset-0 z-50 bg-black"
                : `absolute z-30 w-80 overflow-hidden rounded-lg border bg-popover shadow-xl ${showPreview ? "visible" : "invisible"}`}
            style={expanded ? undefined : { left: `calc(${CAMERA_LOCATION.x}% + 12px)`, top: `calc(${CAMERA_LOCATION.y}% - 12px)` }}
            onMouseEnter={() => setShowPreview(true)}
            onMouseLeave={() => setShowPreview(false)}
            onClick={expanded ? () => setExpanded(false) : undefined}
          >
          {!expanded && (
            <div className="flex items-center gap-2 px-2.5 py-1.5 text-xs font-medium border-b">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              {t("camera.camera1")}
            </div>
          )}
          {mjpegUrl ? (
            <img
              src={mjpegUrl}
              alt="Live stream"
              className={expanded ? "absolute inset-0 h-full w-full object-contain" : "aspect-video w-full object-cover"}
            />
          ) : (
            <div className={`flex aspect-video w-full items-center justify-center bg-muted ${expanded ? "m-auto" : ""}`}>
              <Camera className={expanded ? "h-12 w-12" : "h-6 w-6"} />
            </div>
          )}
          {!expanded && (
            <button
              type="button"
              className="w-full py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-accent/50"
              onClick={() => setExpanded(true)}
            >
              {t("camera.expandFullscreen")}
            </button>
          )}
          {expanded && (
            <>
              <div className="absolute top-3 left-3 flex items-center gap-2 bg-red-500 text-white px-2 py-1 rounded text-xs">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                REC
              </div>
              <Button
                size="sm"
                variant="secondary"
                className="absolute top-3 right-3"
                onClick={(e) => { e.stopPropagation(); setExpanded(false) }}
              >
                <X className="h-4 w-4" />
              </Button>
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/70 backdrop-blur-sm rounded-lg px-4 py-2 text-white text-sm">
                {t("camera.camera1")}
              </div>
            </>
          )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function RealTimeMonitoring() {
  const { t } = useLanguage("en")

  // Client-effective preferences from the settings page (localStorage-backed)
  const [prefs] = useState(loadDisplayPrefs)

  const handleNewDetections = (batch: DetectionResult[]) => {
    if (!prefs.notifyInApp && !prefs.notifySound) return
    const hasViolation = batch.some((detection) => detection.violation)
    if (!hasViolation) return

    if (prefs.notifyInApp) {
      toast.error(t("alerts.newViolation"), { description: t("alerts.newViolationDesc") })
    }
    if (prefs.notifySound) {
      playViolationBeep()
    }
  }

  const { detections, isLoading, error, isRecording, setIsRecording } = useRealTimeDetections({
    maxItems: Math.min(prefs.realtimeRows, MAX_SSE_BUFFER),
    onDetections: handleNewDetections,
  })

  const visibleDetections = useMemo(
    () =>
      prefs.showOnlyViolations ? detections.filter((d) => d.violation) : detections,
    [detections, prefs.showOnlyViolations]
  )

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

      <CampusMap t={t} mjpegUrl={mjpegUrl} />

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
            <DetectionList detections={visibleDetections} t={t} />
          )}
        </CardContent>
      </Card>
    </div>
  )
}
