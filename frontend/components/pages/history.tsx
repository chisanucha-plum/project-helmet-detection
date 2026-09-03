"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { AlertCircle, Download, History as HistoryIcon, RotateCw } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { DetectionList } from "@/components/real-time/DetectionList"
import { useLanguage } from "@/hooks/useLanguage"
import { downloadCsv } from "@/lib/export-csv"
import { fetchHelmetHistory } from "@/services/helmet-detection.service"
import type { DetectionResult } from "@/types/detection.types"

const HISTORY_LIMIT = 500

type StatusFilter = "all" | "violation" | "compliant" | "overCapacity"

/** ISO date (YYYY-MM-DD) of a stored timestamp string, "" when unparseable */
function dayOf(detection: DetectionResult): string {
  const match = detection.timestamp.match(/^(\d{4}-\d{2}-\d{2})/)
  return match ? match[1] : ""
}

export function HistoryPage() {
  const { t } = useLanguage("en")
  const [detections, setDetections] = useState<DetectionResult[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all")
  const [dayFilter, setDayFilter] = useState<string>("")

  const load = useCallback(async () => {
    try {
      setIsLoading(true)
      setError(null)
      setDetections(await fetchHelmetHistory(HISTORY_LIMIT))
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Failed to load history"))
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const filtered = useMemo(() => {
    return detections.filter((detection) => {
      if (dayFilter && dayOf(detection) !== dayFilter) return false
      if (statusFilter === "violation") return detection.violation
      if (statusFilter === "compliant") return !detection.violation
      if (statusFilter === "overCapacity") return detection.passengerCount > 2
      return true
    })
  }, [detections, statusFilter, dayFilter])

  const exportCsv = () => {
    downloadCsv(
      `helmet-history-${new Date().toISOString().slice(0, 10)}.csv`,
      filtered.map((detection) => ({
        timestamp: detection.timestamp,
        helmet: detection.helmetStatus,
        passengers: detection.passengerCount,
        overCapacity: detection.passengerCount > 2 ? "yes" : "no",
        violation: detection.violation ? "yes" : "no",
        frame: detection.framePath ?? "",
      }))
    )
  }

  const violationCount = filtered.filter((detection) => detection.violation).length

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-foreground">{t("history.title")}</h2>
          <p className="text-muted-foreground">{t("history.subtitle")}</p>
        </div>
        <Button variant="outline" size="sm" className="gap-2 bg-transparent" onClick={exportCsv} disabled={filtered.length === 0}>
          <Download className="h-4 w-4" />
          {t("history.exportCsv")}
        </Button>
      </div>

      {error && (
        <div className="bg-critical border border-critical-foreground/20 rounded-lg p-4 flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-critical-foreground flex-shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">{error.message}</p>
          </div>
          <Button size="sm" variant="outline" onClick={load} className="gap-2">
            <RotateCw className="h-4 w-4" />
            {t("buttons.retry")}
          </Button>
        </div>
      )}

      <Card>
        <CardContent className="flex flex-col sm:flex-row sm:items-center gap-3 p-4">
          <Select value={statusFilter} onValueChange={(value) => setStatusFilter(value as StatusFilter)}>
            <SelectTrigger className="sm:w-[220px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{t("history.filterAll")}</SelectItem>
              <SelectItem value="violation">{t("history.filterViolations")}</SelectItem>
              <SelectItem value="compliant">{t("history.filterCompliant")}</SelectItem>
              <SelectItem value="overCapacity">{t("history.filterOverCapacity")}</SelectItem>
            </SelectContent>
          </Select>
          <Input type="date" value={dayFilter} onChange={(event) => setDayFilter(event.target.value)} className="sm:w-[180px]" aria-label={t("history.dayFilter")} />
          {(statusFilter !== "all" || dayFilter) && (
            <Button variant="ghost" size="sm" onClick={() => { setStatusFilter("all"); setDayFilter("") }}>
              {t("history.clearFilters")}
            </Button>
          )}
          <div className="sm:ml-auto flex items-center gap-2">
            <Badge variant="secondary">{filtered.length} / {detections.length}</Badge>
            <Badge variant="destructive">{t("history.violationsBadge")}: {violationCount}</Badge>
          </div>
        </CardContent>
      </Card>

      <div className="relative min-h-[200px]">
        {!isLoading && <DetectionList detections={filtered} t={t} />}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <Card>
              <CardContent className="p-10 flex flex-col items-center gap-3 text-muted-foreground">
                <HistoryIcon className="h-10 w-10 opacity-50" />
                <p>{t("common.loading")}</p>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}
