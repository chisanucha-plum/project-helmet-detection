export const hourlyData = [
    { hour: "06:00", violations: 1, total: 8, compliance: 88 },
    { hour: "07:00", violations: 12, total: 85, compliance: 86 },
    { hour: "08:00", violations: 28, total: 156, compliance: 82 },
    { hour: "09:00", violations: 15, total: 98, compliance: 85 },
    { hour: "10:00", violations: 8, total: 67, compliance: 88 },
    { hour: "11:00", violations: 22, total: 134, compliance: 84 },
    { hour: "12:00", violations: 35, total: 189, compliance: 81 },
    { hour: "13:00", violations: 41, total: 203, compliance: 80 },
    { hour: "14:00", violations: 18, total: 112, compliance: 84 },
    { hour: "15:00", violations: 25, total: 145, compliance: 83 },
    { hour: "16:00", violations: 32, total: 167, compliance: 81 },
    { hour: "17:00", violations: 19, total: 98, compliance: 81 },
    { hour: "18:00", violations: 7, total: 45, compliance: 84 },
]

export const weeklyData = [
    { day: "จันทร์", violations: 287, total: 1456, compliance: 80 },
    { day: "อังคาร", violations: 312, total: 1523, compliance: 79 },
    { day: "พุธ", violations: 298, total: 1489, compliance: 80 },
    { day: "พฤหัส", violations: 334, total: 1612, compliance: 79 },
    { day: "ศุกร์", violations: 356, total: 1678, compliance: 79 },
    { day: "เสาร์", violations: 89, total: 567, compliance: 84 },
    { day: "อาทิตย์", violations: 45, total: 298, compliance: 85 },
]

export const helmetComplianceData = [
    { name: "สวมหมวกกันน็อค", value: 79, color: "#10b981" },
    { name: "ไม่สวมหมวกกันน็อค", value: 21, color: "#ef4444" },
]

export const violationTypeData = [
    { type: "ไม่สวมหมวกกันน็อค", count: 1245, percentage: 68 },
    { type: "นั่งเกิน 2 คน", count: 387, percentage: 21 },
    { type: "ไม่สวมหมวก + เกิน 2 คน", count: 189, percentage: 10 },
    { type: "ไม่มีป้ายทะเบียน", count: 23, percentage: 1 },
]
