import { memo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
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

const tooltipStyle = {
  backgroundColor: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: "8px",
}

const lineChartDot = { fill: "var(--chart-1)", strokeWidth: 2, r: 4 }

interface DashboardChartsProps {
  chartData: Array<{
    name: string
    total: number
    violations: number
    compliance: number | null
  }>
  helmetPieData: Array<{
    name: string
    value: number
    color: string
  }>
  labels: {
    totalViolations: string
    totalDetections: string
    complianceRate: string
  }
  complianceByDayLabel: string
  helmetComplianceLabel: string
}

const DashboardCharts = memo(function DashboardCharts({ 
  chartData, 
  helmetPieData, 
  labels,
  complianceByDayLabel,
  helmetComplianceLabel,
}: DashboardChartsProps) {
  return (
    <>
      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Violations Trend Chart */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">{complianceByDayLabel}</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="name" stroke="var(--muted-foreground)" fontSize={12} />
                <YAxis stroke="var(--muted-foreground)" fontSize={12} />
                <Tooltip contentStyle={tooltipStyle} />
                <Area
                  type="monotone"
                  dataKey="violations"
                  stroke="var(--chart-3)"
                  fill="var(--chart-3)"
                  fillOpacity={0.3}
                  name={labels.totalViolations}
                />
                <Area
                  type="monotone"
                  dataKey="total"
                  stroke="var(--chart-1)"
                  fill="var(--chart-1)"
                  fillOpacity={0.1}
                  name={labels.totalDetections}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Helmet Compliance Pie Chart */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">{helmetComplianceLabel}</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={helmetPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {helmetPieData.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-6 mt-4">
              {helmetPieData.map((item) => (
                <div key={item.name} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                  <span className="text-sm text-muted-foreground">{item.name}</span>
                  <span className="text-sm font-medium">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Compliance Rate Trend */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">{labels.complianceRate}</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" stroke="var(--muted-foreground)" fontSize={12} />
              <YAxis stroke="var(--muted-foreground)" fontSize={12} domain={[0, 100]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line
                type="monotone"
                dataKey="compliance"
                connectNulls
                stroke="var(--chart-1)"
                strokeWidth={3}
                dot={lineChartDot}
                name={labels.complianceRate}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </>
  )
})

export default DashboardCharts
