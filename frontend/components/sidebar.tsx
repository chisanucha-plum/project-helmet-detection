"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { BarChart3, ChevronLeft, ChevronRight, HelpCircle, Home, Settings, Shield, X } from "lucide-react"
import { usePathname, useRouter } from "next/navigation"
import type React from "react"

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
  onNavigate?: () => void
  isMobile?: boolean
}

interface NavItem {
  icon: React.ComponentType<{ className?: string }>
  label: string
  path: string
  description?: string
}

const navItems: NavItem[] = [
  {
    icon: Home,
    label: "หน้าหลัก",
    path: "/",
    description: "การตรวจสอบแบบ Real-time",
  },
  {
    icon: BarChart3,
    label: "แดชบอร์ด",
    path: "/dashboard",
    description: "สถิติและรายงาน",
  },
]

const bottomNavItems: NavItem[] = [
  {
    icon: Settings,
    label: "ตั้งค่า",
    path: "/settings",
    description: "การตั้งค่าระบบ",
  },
  {
    icon: HelpCircle,
    label: "ช่วยเหลือ",
    path: "/help",
    description: "คู่มือการใช้งาน",
  },
]

export function Sidebar({ collapsed, onToggle, onNavigate, isMobile = false }: SidebarProps) {
  const pathname = usePathname()
  const router = useRouter()

  const handleNavigation = (path: string) => {
    router.push(path)
    onNavigate?.()
  }

  return (
    <div
      className={cn(
        "bg-sidebar border-r border-sidebar-border flex flex-col transition-all duration-300 ease-in-out h-full",
        isMobile ? "w-64" : collapsed ? "w-16" : "w-64",
      )}
    >
      {/* Header */}
      <div className="p-4 border-b border-sidebar-border">
        <div className="flex items-center justify-between">
          {(!collapsed || isMobile) && (
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-sidebar-accent rounded-lg flex items-center justify-center">
                <Shield className="w-5 h-5 text-sidebar-accent-foreground" />
              </div>
              <div className="min-w-0 flex-1">
                <h2 className="font-semibold text-sidebar-foreground text-sm truncate">ระบบตรวจจับหมวก</h2>
                <p className="text-xs text-sidebar-foreground/60 truncate">มหาวิทยาลัย</p>
              </div>
            </div>
          )}

          <Button
            variant="ghost"
            size="sm"
            onClick={onToggle}
            className="h-8 w-8 p-0 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground flex-shrink-0"
          >
            {isMobile ? (
              <X className="h-4 w-4" />
            ) : collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.path

            return (
              <Button
                key={item.path}
                variant="ghost"
                className={cn(
                  // base: make the button a flex container and vertically center contents
                  "w-full h-auto p-3 flex items-center transition-colors",
                  // when collapsed (desktop) center horizontally and remove extra horizontal padding
                  collapsed && !isMobile ? "justify-center text-center px-0" : "justify-start text-left px-3",
                  isActive
                    ? // Enhanced orange color scheme for better visibility
                    "bg-orange-500 text-white hover:bg-orange-600 shadow-sm"
                    : "text-sidebar-foreground hover:bg-sidebar-primary hover:text-sidebar-primary-foreground",
                )}
                onClick={() => handleNavigation(item.path)}
              >
                <Icon className={cn("h-5 w-5 flex-shrink-0", collapsed && !isMobile ? "" : "mr-3")} />
                {(!collapsed || isMobile) && (
                  <div className="flex flex-col items-start min-w-0 flex-1">
                    <span className="font-medium text-sm truncate w-full">{item.label}</span>
                    {item.description && <span className="text-xs opacity-60 truncate w-full">{item.description}</span>}
                  </div>
                )}
              </Button>
            )
          })}
        </div>
      </nav>

      {/* Status Indicator */}
      {(!collapsed || isMobile) && (
        <div className="p-4 border-t border-sidebar-border">
          <div className="bg-sidebar-primary rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse flex-shrink-0"></div>
              <span className="text-xs font-medium text-sidebar-primary-foreground truncate">สถานะระบบ</span>
            </div>
            <p className="text-xs text-sidebar-primary-foreground/80 truncate">เชื่อมต่อแล้ว</p>
            <p className="text-xs text-sidebar-primary-foreground/60 truncate">กล้อง: 2/2 ออนไลน์</p>
          </div>
        </div>
      )}

      {/* Bottom Navigation */}
      <div className="p-4 border-t border-sidebar-border">
        <div className="space-y-1">
          {bottomNavItems.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.path

            return (
              <Button
                key={item.path}
                variant="ghost"
                className={cn(
                  // base: flex with vertical centering
                  "w-full h-auto p-2 flex items-center transition-colors",
                  collapsed && !isMobile ? "justify-center text-center px-0" : "justify-start text-left px-2",
                  isActive
                    ? // Enhanced orange color scheme for bottom navigation
                    "bg-orange-500 text-white hover:bg-orange-600 shadow-sm"
                    : "text-sidebar-foreground/80 hover:bg-sidebar-primary hover:text-sidebar-primary-foreground",
                )}
                onClick={() => handleNavigation(item.path)}
              >
                <Icon className={cn("h-4 w-4 flex-shrink-0", collapsed && !isMobile ? "" : "mr-2")} />
                {(!collapsed || isMobile) && <span className="text-sm truncate">{item.label}</span>}
              </Button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
