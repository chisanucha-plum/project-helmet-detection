"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { AlertTriangle, BikeIcon, Download, Shield, TrendingDown, TrendingUp, Users, type LucideIcon } from "lucide-react"
import { useState, useMemo, memo, lazy, Suspense, type ReactNode } from "react"

import { useHelmetStats } from "@/hooks/useHelmetStats"
import { useLanguage } from "@/hooks/useLanguage"
import { downloadCsv } from "@/lib/export-csv"
import type { StatsBucketSize, StatsTimeRange } from "@/types/detection.types"

// Lazy load recharts to reduce initial bundle and delay heavy SVG rendering
const DashboardCharts = lazy(() => import("@/components/dashboard/DashboardCharts"))

/** Percent change vs the previous period; null when not computable */
function percentChange(current: number, previous: number): number | null {
  if (previous === 0) return null
  return Math.round(((current - previous) / previous) * 100)
}

/** "2026-08-23 14" -> "14:00", "2026-08-23" -> "23/08" */
function formatBucketLabel(label: string, bucketSize: StatsBucketSize): string {
  if (bucketSize === "hour") return `${label.slice(11, 13)}:00`
  return `${label.slice(8, 10)}/${label.slice(5, 7)}`
}

/** Calculate compliance delta as percentage points, not percent */
function calculateComplianceDelta(current: number, previous: number): number {
  return Math.round((current - previous) * 10) / 10
}

interface TrendIndicatorProps {
  delta: number | null
  /** Direction of change considered good for this metric */
  goodWhen: "up" | "down"
  vsLabel: string
}

function TrendIndicator({ delta, goodWhen, vsLabel }: TrendIndicatorProps) {
  if (delta === null || delta === 0) return null
  const isUp = delta > 0
  const isGood = goodWhen === "up" ? isUp : !isUp
  const colorClass = isGood ? "text-success-foreground" : "text-critical-foreground"
  const Icon = isUp ? TrendingUp : TrendingDown
  return (
    <div className="flex items-center gap-1 mt-1">
      <Icon className={`h-4 w-4 ${colorClass}`} />
      <span className={`text-sm ${colorClass}`}>
        {isUp ? "+" : ""}{delta}%{" "}
        <span className="text-muted-foreground">{vsLabel}</span>
      </span>
    </div>
  )
}

interface StatCardProps {
  label: string
  value: string | number
  trend?: ReactNode
  icon: LucideIcon
  iconBgClass: string
  iconColorClass: string
}

const StatCard = memo(function StatCard({ label, value, trend, icon: Icon, iconBgClass, iconColorClass }: StatCardProps) {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-muted-foreground">{label}</p>
            <p className="text-3xl font-bold text-foreground">{value}</p>
            {trend}
          </div>
          <div className={`w-12 h-12 ${iconBgClass} rounded-lg flex items-center justify-center`}>
            <Icon className={`h-6 w-6 ${iconColorClass}`} />
          </div>
        </div>
      </CardContent>
    </Card>
  )
})

export function Dashboard() {
  const { t } = useLanguage("en")
  const [timeRange, setTimeRange] = useState<StatsTimeRange>("today")
  const { stats, previousStats, error, refetch } = useHelmetStats(timeRange)

  const summary = stats?.summary ?? null
  const prev = previousStats?.summary ?? null

  const totalViolations = summary?.total_violations ?? 0
  const totalDetections = summary?.total_detections ?? 0
  const compliancePercent = summary?.compliance_percent ?? 0
  const excessPassengers = summary?.excess_passengers ?? 0

  const violationsDelta = summary && prev ? percentChange(totalViolations, prev.total_violations) : null
  const detectionsDelta = summary && prev ? percentChange(totalDetections, prev.total_detections) : null
  const excessDelta = summary && prev ? percentChange(excessPassengers, prev.excess_passengers) : null
  const complianceDelta = summary && prev ? calculateComplianceDelta(summary.compliance_percent, prev.compliance_percent) : null

  // Localized labels - stable across renders when language doesn't change
  const labels = useMemo(() => ({
    totalViolations: t("dashboard.totalViolations"),
    totalDetections: t("dashboard.totalDetections"),
    complianceRate: t("stats.complianceRate"),
    wearingHelmet: t("detection.wearingHelmet"),
    notWearingHelmet: t("detection.notWearingHelmet"),
  }), [t])

  // Area/line chart series: display label + per-bucket compliance rate
  const chartData = useMemo(() => {
    if (!stats) return []
    return stats.series.map((bucket) => ({
      name: formatBucketLabel(bucket.label, stats.bucket_size),
      total: bucket.total,
      violations: bucket.violations,
      compliance:
        bucket.total > 0
          ? Math.round(((bucket.total - bucket.violations) / bucket.total) * 100)
          : null,
    }))
  }, [stats])

  // Pie chart shares derived from helmet on/off totals
  const helmetPieData = useMemo(() => {
    if (!summary) return []
    const denominator = summary.helmet_on + summary.helmet_off
    if (denominator === 0) return []
    return [
      {
        name: labels.wearingHelmet,
        value: Math.round((summary.helmet_on / denominator) * 100),
        color: "var(--success-foreground)",
      },
      {
        name: labels.notWearingHelmet,
        value: Math.round((summary.helmet_off / denominator) * 100),
        color: "var(--critical-foreground)",
      },
    ]
  }, [summary, labels.wearingHelmet, labels.notWearingHelmet])

  // Localized violation-type rows with share of all recorded violations
  const violationRows = useMemo(() => {
    if (!stats) return []
    const names: Record<string, string> = {
      no_helmet: t("detection.notWearingHelmet"),
      over_capacity: t("stats.overCapacity"),
    }
    const total = stats.violation_types.reduce((sum, item) => sum + item.count, 0)
    return stats.violation_types.map((item) => ({
      type: names[item.type] ?? item.type,
      count: item.count,
      percentage: total > 0 ? Math.round((item.count / total) * 100) : 0,
    }))
  }, [stats, t])

  // CSV report of the currently selected range: summary line + per-bucket series
  const exportReport = () => {
    if (!stats || !summary) return
    downloadCsv(
      `helmet-report-${stats.range_from}_to_${stats.range_to}.csv`,
      [
        {
          metric: "summary",
          total_detections: summary.total_detections,
          violations: summary.total_violations,
          helmet_on: summary.helmet_on,
          helmet_off: summary.helmet_off,
          excess_passengers: summary.excess_passengers,
          compliance_percent: summary.compliance_percent,
        },
        ...chartData.map((bucket) => ({
          metric: bucket.name,
          total_detections: bucket.total,
          violations: bucket.violations,
          helmet_on: "",
          helmet_off: "",
          excess_passengers: "",
          compliance_percent: bucket.compliance ?? "",
        })),
      ]
    )
  }

  // Full-screen fallback while the first load has not produced data yet
  if (!stats) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-foreground">{t("dashboard.title")}</h2>
        {error ? (
          <Card>
            <CardContent className="p-10 flex flex-col items-center gap-4">
              <AlertTriangle className="h-10 w-10 text-red-500" />
              <p className="text-muted-foreground">{t("errors.errorOccurred")}</p>
              <Button variant="outline" size="sm" onClick={refetch}>
                {t("buttons.retry")}
              </Button>
            </CardContent>
          </Card>
        ) : (
          <p className="text-muted-foreground">{t("common.loading")}</p>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-foreground">{t("dashboard.title")}</h2>
          <p className="text-muted-foreground">{t("dashboard.subtitle")}</p>
          {error && (
            <div className="flex items-center gap-2 mt-2">
              <Badge variant="secondary" className="text-xs text-critical-foreground">{t("errors.errorOccurred")}</Badge>
              <Button variant="ghost" size="sm" onClick={refetch}>{t("buttons.retry")}</Button>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          <Select value={timeRange} onValueChange={(value) => setTimeRange(value as StatsTimeRange)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="today">{t("dashboard.today")}</SelectItem>
              <SelectItem value="week">{t("dashboard.thisWeek")}</SelectItem>
              <SelectItem value="month">{t("dashboard.thisMonth")}</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="outline" size="sm" className="gap-2 bg-transparent" onClick={exportReport}>
            <Download className="h-4 w-4" />
            {t("dashboard.downloadReport")}
          </Button>
        </div>
      </div>

      {/* Key Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          label={t("dashboard.totalViolations")}
          value={totalViolations}
          trend={<TrendIndicator delta={violationsDelta} goodWhen="down" vsLabel={t("dashboard.vsPrevPeriod")} />}
          icon={AlertTriangle}
          iconBgClass="bg-critical"
          iconColorClass="text-critical-foreground"
        />

        <StatCard
          label={t("dashboard.helmetCompliance")}
          value={`${compliancePercent}%`}
          trend={<TrendIndicator delta={complianceDelta} goodWhen="up" vsLabel={t("dashboard.vsPrevPeriod")} />}
          icon={Shield}
          iconBgClass="bg-success"
          iconColorClass="text-success-foreground"
        />

        <StatCard
          label={t("dashboard.totalDetections")}
          value={totalDetections}
          trend={<TrendIndicator delta={detectionsDelta} goodWhen="up" vsLabel={t("dashboard.vsPrevPeriod")} />}
          icon={BikeIcon}
          iconBgClass="bg-info"
          iconColorClass="text-info-foreground"
        />

        <StatCard
          label={t("dashboard.excessPassengers")}
          value={excessPassengers}
          trend={<TrendIndicator delta={excessDelta} goodWhen="down" vsLabel={t("dashboard.vsPrevPeriod")} />}
          icon={Users}
          iconBgClass="bg-warning"
          iconColorClass="text-warning-foreground"
        />
      </div>

      {/* Charts Section */}
      <Suspense fallback={
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card><CardContent className="p-10"><div className="w-8 h-8 border-4 border-muted border-t-foreground rounded-full animate-spin mx-auto" /></CardContent></Card>
          <Card><CardContent className="p-10"><div className="w-8 h-8 border-4 border-muted border-t-foreground rounded-full animate-spin mx-auto" /></CardContent></Card>
        </div>
      }>
        <DashboardCharts 
          chartData={chartData} 
          helmetPieData={helmetPieData} 
          labels={labels} 
          complianceByDayLabel={t("dashboard.complianceByDay")}
          helmetComplianceLabel={t("dashboard.helmetCompliance")}
        />
      </Suspense>

      {/* Violation Types Breakdown */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">{t("dashboard.violationTypes")}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {violationRows.map((item) => (
              <div key={item.type} className="flex items-center justify-between p-4 bg-muted rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-4 h-4 bg-chart-3 rounded"></div>
                  <span className="font-medium">{item.type}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-2xl font-bold">{item.count}</span>
                  <Badge variant="secondary">{item.percentage}%</Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
