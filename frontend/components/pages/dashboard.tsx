"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { AlertTriangle, BikeIcon, Download, Shield, TrendingDown, TrendingUp, Users } from "lucide-react"
import { useState, useMemo, memo } from "react"
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { helmetComplianceData, hourlyData, violationTypeData, weeklyData } from "@/mocks/dashboardMocks"
import { useLanguage } from "@/hooks/useLanguage"

// Memoize tooltip style to prevent recreation on every render
const tooltipStyle = {
  backgroundColor: "hsl(var(--card))",
  border: "1px solid hsl(var(--border))",
  borderRadius: "8px",
}

// Memoized chart components to prevent unnecessary re-renders
const MemoizedAreaChart = memo(AreaChart)
const MemoizedLineChart = memo(LineChart)
const MemoizedPieChart = memo(PieChart)
const MemoizedCartesianGrid = memo(CartesianGrid)
const MemoizedXAxis = memo(XAxis)
const MemoizedYAxis = memo(YAxis)
const MemoizedTooltip = memo(Tooltip)
const MemoizedArea = memo(Area)
const MemoizedLine = memo(Line)
const MemoizedPie = memo(Pie)

export function Dashboard() {
  const { t } = useLanguage("en")
  const [timeRange, setTimeRange] = useState("today")
  const [chartType, setChartType] = useState("hourly")

  const currentData = useMemo(() => chartType === "hourly" ? hourlyData : weeklyData, [chartType])

  const totalViolations = currentData.reduce((sum, item) => sum + item.violations, 0)
  const totalDetections = currentData.reduce((sum, item) => sum + item.total, 0)
  const averageCompliance = Math.round(currentData.reduce((sum, item) => sum + item.compliance, 0) / currentData.length)
  const excessPassengers = violationTypeData.find((item) => item.type.includes("เกิน 2 คน"))?.count || 0
  const totalLicensePlates = totalDetections

  return (
    <div className="space-y-6">
      {/* Header with Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-foreground">{t("dashboard.title")}</h2>
          <p className="text-muted-foreground">{t("dashboard.subtitle")}</p>
        </div>

        <div className="flex items-center gap-3">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="today">{t("dashboard.today")}</SelectItem>
              <SelectItem value="week">{t("dashboard.thisWeek")}</SelectItem>
              <SelectItem value="month">{t("dashboard.thisMonth")}</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="outline" size="sm" className="gap-2 bg-transparent">
            <Download className="h-4 w-4" />
            {t("dashboard.downloadReport")}
          </Button>
        </div>
      </div>

      {/* Key Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">{t("dashboard.totalViolations")}</p>
                <p className="text-3xl font-bold text-foreground">{totalViolations}</p>
                <div className="flex items-center gap-1 mt-1">
                  <TrendingUp className="h-4 w-4 text-red-500" />
                  <span className="text-sm text-red-500">+8.5% {t("dashboard.yesterday")}</span>
                </div>
              </div>
              <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                <AlertTriangle className="h-6 w-6 text-red-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">{t("dashboard.helmetCompliance")}</p>
                <p className="text-3xl font-bold text-foreground">{averageCompliance}%</p>
                <div className="flex items-center gap-1 mt-1">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-green-500">+2.1% {t("dashboard.yesterday")}</span>
                </div>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <Shield className="h-6 w-6 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">{t("dashboard.totalDetections")}</p>
                <p className="text-3xl font-bold text-foreground">{totalDetections}</p>
                <div className="flex items-center gap-1 mt-1">
                  <TrendingDown className="h-4 w-4 text-blue-500" />
                  <span className="text-sm text-blue-500">-1.2% {t("dashboard.yesterday")}</span>
                </div>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <BikeIcon className="h-6 w-6 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">{t("dashboard.excessPassengers")}</p>
                <p className="text-3xl font-bold text-foreground">{excessPassengers}</p>
                <div className="flex items-center gap-1 mt-1">
                  <TrendingUp className="h-4 w-4 text-orange-500" />
                  <span className="text-sm text-orange-500">+12.3% {t("dashboard.yesterday")}</span>
                </div>
              </div>
              <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                <Users className="h-6 w-6 text-orange-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">{t("dashboard.totalDetections")}</p>
                <p className="text-3xl font-bold text-foreground">{totalLicensePlates}</p>
                <div className="flex items-center gap-1 mt-1">
                  <TrendingUp className="h-4 w-4 text-purple-500" />
                  <span className="text-sm text-purple-500">+5.7% {t("dashboard.yesterday")}</span>
                </div>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                <BikeIcon className="h-6 w-6 text-purple-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Violations Trend Chart */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">{t("dashboard.complianceByDay")}</CardTitle>
              <Select value={chartType} onValueChange={setChartType}>
                <SelectTrigger className="w-[120px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hourly">{t("dashboard.hourly")}</SelectItem>
                  <SelectItem value="weekly">{t("dashboard.weekly")}</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <MemoizedAreaChart data={currentData}>
                <MemoizedCartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <MemoizedXAxis
                  dataKey={chartType === "hourly" ? "hour" : "day"}
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <MemoizedYAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                <MemoizedTooltip contentStyle={tooltipStyle} />
                <MemoizedArea
                  type="monotone"
                  dataKey="violations"
                  stroke="hsl(var(--chart-3))"
                  fill="hsl(var(--chart-3))"
                  fillOpacity={0.3}
                  name="การกระทำผิด"
                />
                <MemoizedArea
                  type="monotone"
                  dataKey="total"
                  stroke="hsl(var(--chart-1))"
                  fill="hsl(var(--chart-1))"
                  fillOpacity={0.1}
                  name="ทั้งหมด"
                />
              </MemoizedAreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Helmet Compliance Pie Chart */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">{t("dashboard.helmetCompliance")}</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <MemoizedPieChart>
                <MemoizedPie
                  data={helmetComplianceData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {helmetComplianceData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </MemoizedPie>
                <MemoizedTooltip contentStyle={tooltipStyle} />
              </MemoizedPieChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-6 mt-4">
              {helmetComplianceData.map((item, index) => (
                <div key={index} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                  <span className="text-sm text-muted-foreground">{item.name}</span>
                  <span className="text-sm font-medium">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* License Plate Detection by Province (removed) */}

      {/* Compliance Rate Trend */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">{t("dashboard.complianceByDay")}</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <MemoizedLineChart data={currentData}>
              <MemoizedCartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <MemoizedXAxis
                dataKey={chartType === "hourly" ? "hour" : "day"}
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <MemoizedYAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[60, 100]} />
              <MemoizedTooltip contentStyle={tooltipStyle} />
              <MemoizedLine
                type="monotone"
                dataKey="compliance"
                stroke="hsl(var(--chart-1))"
                strokeWidth={3}
                dot={{ fill: "hsl(var(--chart-1))", strokeWidth: 2, r: 4 }}
                name="อัตราการปฏิบัติตาม (%)"
              />
            </MemoizedLineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Violation Types Breakdown */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">{t("dashboard.violationTypes")}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {violationTypeData.map((item, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-muted rounded-lg">
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
