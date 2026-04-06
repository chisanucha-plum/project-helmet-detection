"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Switch } from "@/components/ui/switch"
import { getStoredUserRole } from "@/stores/auth-store"
import { BellRing, Camera, Save, Settings2, ShieldCheck, UserRound } from "lucide-react"
import { useRouter } from "next/navigation"
import { useEffect, useMemo, useState } from "react"

type UserRole = "admin" | "security" | "user"

type AppSettings = {
  fullName: string
  email: string
  timezone: string
  language: string
  notifyInApp: boolean
  notifyEmail: boolean
  notifySound: boolean
  notifyDigest: string
  realtimeRows: string
  refreshInterval: string
  showOnlyViolations: boolean
  cameraSource: string
  detectionThreshold: string
  snapshotEnabled: boolean
  retentionDays: string
}

const DEFAULT_SETTINGS: AppSettings = {
  fullName: "",
  email: "",
  timezone: "Asia/Bangkok",
  language: "th",
  notifyInApp: true,
  notifyEmail: false,
  notifySound: true,
  notifyDigest: "instant",
  realtimeRows: "20",
  refreshInterval: "5",
  showOnlyViolations: false,
  cameraSource: "rtsp://camera-main",
  detectionThreshold: "0.50",
  snapshotEnabled: true,
  retentionDays: "30",
}

const STORAGE_KEY = "helmet_app_settings"

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS)
  const [savedMessage, setSavedMessage] = useState("")
  const [role, setRole] = useState<UserRole>("security")

  useEffect(() => {
    const storedRole = getStoredUserRole()
    if (storedRole === "admin" || storedRole === "security" || storedRole === "user") {
      setRole(storedRole)
    }

    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (!raw) {
        const name = localStorage.getItem("userName") || ""
        setSettings((prev) => ({ ...prev, fullName: name }))
        return
      }

      const parsed = JSON.parse(raw) as Partial<AppSettings>
      const name = localStorage.getItem("userName") || ""
      setSettings({
        ...DEFAULT_SETTINGS,
        ...parsed,
        fullName: parsed.fullName || name,
      })
    } catch {
      setSettings(DEFAULT_SETTINGS)
    }
  }, [])

  const roleBadgeVariant = useMemo(() => {
    return role === "admin" ? "default" : "secondary"
  }, [role])

  const updateSettings = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }))
    setSavedMessage("")
  }

  const saveSettings = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
    setSavedMessage("บันทึกการตั้งค่าเรียบร้อยแล้ว")
  }

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold text-foreground">ตั้งค่าระบบ</h1>
          <p className="text-sm text-muted-foreground">ปรับการแสดงผล การแจ้งเตือน และการตั้งค่าการตรวจจับตามสิทธิ์ของคุณ</p>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant={roleBadgeVariant}>Role: {role === "admin" ? "Admin" : role === "security" ? "Security" : "User"}</Badge>
          <Button onClick={saveSettings} className="gap-2">
            <Save className="h-4 w-4" />
            บันทึกทั้งหมด
          </Button>
        </div>
      </div>

      {savedMessage ? (
        <div className="rounded-md border border-green-200 bg-green-50 px-4 py-2 text-sm text-green-700">
          {savedMessage}
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UserRound className="h-4 w-4" />
              โปรไฟล์ผู้ใช้
            </CardTitle>
            <CardDescription>ข้อมูลพื้นฐานและรูปแบบการใช้งานของบัญชีนี้</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="fullName">ชื่อที่แสดง</Label>
              <Input
                id="fullName"
                value={settings.fullName}
                onChange={(event) => updateSettings("fullName", event.target.value)}
                placeholder="ชื่อผู้ใช้งาน"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="email">อีเมล</Label>
              <Input
                id="email"
                value={settings.email}
                onChange={(event) => updateSettings("email", event.target.value)}
                placeholder="example@kmutt.ac.th"
              />
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="timezone">Time Zone</Label>
                <Select
                  value={settings.timezone}
                  onValueChange={(value) => updateSettings("timezone", value)}
                >
                  <SelectTrigger id="timezone" className="w-full">
                    <SelectValue placeholder="เลือก Time Zone" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Asia/Bangkok">Asia/Bangkok</SelectItem>
                    <SelectItem value="UTC">UTC</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="language">ภาษา</Label>
                <Select
                  value={settings.language}
                  onValueChange={(value) => updateSettings("language", value)}
                >
                  <SelectTrigger id="language" className="w-full">
                    <SelectValue placeholder="เลือกภาษา" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="th">ไทย</SelectItem>
                    <SelectItem value="en">English</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BellRing className="h-4 w-4" />
              การแจ้งเตือน
            </CardTitle>
            <CardDescription>ควบคุมรูปแบบและความถี่ในการรับแจ้งเตือน</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <SettingSwitch
              id="notifyInApp"
              label="แจ้งเตือนในระบบ"
              description="แสดงแจ้งเตือนในหน้าเว็บเมื่อพบเหตุผิดปกติ"
              checked={settings.notifyInApp}
              onCheckedChange={(checked) => updateSettings("notifyInApp", checked)}
            />
            <SettingSwitch
              id="notifyEmail"
              label="แจ้งเตือนทางอีเมล"
              description="ส่งเหตุการณ์สำคัญไปยังอีเมลที่ลงทะเบียน"
              checked={settings.notifyEmail}
              onCheckedChange={(checked) => updateSettings("notifyEmail", checked)}
            />
            <SettingSwitch
              id="notifySound"
              label="เปิดเสียงแจ้งเตือน"
              description="เล่นเสียงเมื่อพบการกระทำผิด"
              checked={settings.notifySound}
              onCheckedChange={(checked) => updateSettings("notifySound", checked)}
            />

            <Separator />

            <div className="space-y-2">
              <Label htmlFor="notifyDigest">รูปแบบแจ้งเตือน</Label>
              <Select
                value={settings.notifyDigest}
                onValueChange={(value) => updateSettings("notifyDigest", value)}
              >
                <SelectTrigger id="notifyDigest" className="w-full">
                  <SelectValue placeholder="เลือกรูปแบบ" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="instant">ทันที</SelectItem>
                  <SelectItem value="5min">สรุปทุก 5 นาที</SelectItem>
                  <SelectItem value="daily">สรุปรายวัน</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings2 className="h-4 w-4" />
              การแสดงผลหน้า Real-time
            </CardTitle>
            <CardDescription>ปรับการ refresh และข้อมูลที่ต้องการแสดง</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="realtimeRows">จำนวนรายการล่าสุด</Label>
                <Select
                  value={settings.realtimeRows}
                  onValueChange={(value) => updateSettings("realtimeRows", value)}
                >
                  <SelectTrigger id="realtimeRows" className="w-full">
                    <SelectValue placeholder="จำนวนรายการ" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="10">10</SelectItem>
                    <SelectItem value="20">20</SelectItem>
                    <SelectItem value="50">50</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="refreshInterval">ช่วงเวลารีเฟรช (วินาที)</Label>
                <Select
                  value={settings.refreshInterval}
                  onValueChange={(value) => updateSettings("refreshInterval", value)}
                >
                  <SelectTrigger id="refreshInterval" className="w-full">
                    <SelectValue placeholder="เลือกช่วงเวลา" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="2">2</SelectItem>
                    <SelectItem value="5">5</SelectItem>
                    <SelectItem value="10">10</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <SettingSwitch
              id="showOnlyViolations"
              label="แสดงเฉพาะรายการผิดกฎ"
              description="ซ่อนเหตุการณ์ที่ถูกต้องตามกฎจากตารางแสดงผล"
              checked={settings.showOnlyViolations}
              onCheckedChange={(checked) => updateSettings("showOnlyViolations", checked)}
            />
          </CardContent>
        </Card>

        {role === "admin" ? (
          <Card className="border-orange-200">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-orange-700">
                <ShieldCheck className="h-4 w-4" />
                Admin Settings
              </CardTitle>
              <CardDescription>ค่าระบบที่มีผลต่อการตรวจจับและการจัดเก็บข้อมูล</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="cameraSource" className="flex items-center gap-2">
                  <Camera className="h-4 w-4" />
                  Camera Source
                </Label>
                <Input
                  id="cameraSource"
                  value={settings.cameraSource}
                  onChange={(event) => updateSettings("cameraSource", event.target.value)}
                  placeholder="rtsp://camera-main"
                />
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="detectionThreshold">Detection Threshold</Label>
                  <Input
                    id="detectionThreshold"
                    value={settings.detectionThreshold}
                    onChange={(event) => updateSettings("detectionThreshold", event.target.value)}
                    placeholder="0.50"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="retentionDays">เก็บข้อมูล (วัน)</Label>
                  <Input
                    id="retentionDays"
                    value={settings.retentionDays}
                    onChange={(event) => updateSettings("retentionDays", event.target.value)}
                    placeholder="30"
                  />
                </div>
              </div>

              <SettingSwitch
                id="snapshotEnabled"
                label="เปิดการบันทึก Snapshot"
                description="บันทึกรูปเมื่อพบการกระทำผิดเพื่อใช้ตรวจสอบย้อนหลัง"
                checked={settings.snapshotEnabled}
                onCheckedChange={(checked) => updateSettings("snapshotEnabled", checked)}
              />
            </CardContent>
          </Card>
        ) : null}
      </div>
    </div>
  )
}

type SettingSwitchProps = {
  id: string
  label: string
  description: string
  checked: boolean
  onCheckedChange: (checked: boolean) => void
}

function SettingSwitch({
  id,
  label,
  description,
  checked,
  onCheckedChange,
}: SettingSwitchProps) {
  return (
    <div className="flex items-start justify-between gap-4 rounded-md border p-3">
      <div className="space-y-1">
        <Label htmlFor={id}>{label}</Label>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
      <Switch id={id} checked={checked} onCheckedChange={onCheckedChange} />
    </div>
  )
}
