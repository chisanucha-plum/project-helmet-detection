"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { getStoredUserRole, type UserRole } from "@/stores/auth-store"
import { BarChart3, ChevronLeft, ChevronRight, HelpCircle, Home, Menu, Settings, X } from "lucide-react"
import { usePathname, useRouter } from "next/navigation"
import type React from "react"
import { useEffect, useMemo, useState } from "react"

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
    path: "/real-time-monitoring",
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
    icon: HelpCircle,
    label: "ช่วยเหลือ",
    path: "/help",
    description: "คู่มือการใช้งาน",
  },
]

const adminOnlyBottomNavItems: NavItem[] = [
  {
    icon: Settings,
    label: "ตั้งค่า",
    path: "/settings",
    description: "การตั้งค่าระบบ",
  },
]

export function Sidebar({ collapsed, onToggle, onNavigate, isMobile = false }: SidebarProps) {
  const pathname = usePathname()
  const router = useRouter()
  const [role, setRole] = useState<UserRole>(null)

  useEffect(() => {
    setRole(getStoredUserRole())
  }, [])

  const visibleNavItems = useMemo(() => {
    if (role === "admin") {
      return navItems
    }

    return navItems.filter((item) => item.path !== "/dashboard")
  }, [role])

  const visibleBottomNavItems = useMemo(() => {
    if (role === "admin" || role === "security") {
      return [...adminOnlyBottomNavItems, ...bottomNavItems]
    }

    return bottomNavItems
  }, [role])

  const handleNavigation = (path: string) => {
    router.push(path)
    onNavigate?.()
  }

  return (
    <div
      className={cn(
        "bg-white border-r border-gray-200 shadow-md flex flex-col transition-all duration-300 ease-in-out h-full",
        isMobile 
          ? "w-48 sm:w-56" 
          : collapsed 
            ? "w-12 sm:w-14" 
            : "w-48 sm:w-56 lg:w-64",
      )}
    >
      {/* Header */}
      <div className="border-b border-gray-200 transition-all duration-300 p-3">
        {/* Expanded mode */}
        {(!collapsed || isMobile) && (
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 bg-white rounded-md flex items-center justify-center overflow-hidden shadow-sm border">
              <img 
                src="/icon.png" 
                alt="Logo" 
                className="w-7 h-7 object-contain"
              />
            </div>
            <div className="min-w-0 flex-1">
              <h2 className="font-semibold text-sidebar-foreground text-xs sm:text-sm truncate">ระบบตรวจจับการขับขี่</h2>
              <p className="text-xs text-sidebar-foreground/60 truncate hidden sm:block">เทคโนโลยีพระจอมเกล้าธนบุรี</p>
            </div>
            {isMobile && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onNavigate}
                className="h-8 w-8 sm:h-10 sm:w-10 p-0 hover:bg-gray-100 flex-shrink-0"
              >
                <X className="h-4 w-4 sm:h-5 sm:w-5" />
              </Button>
            )}
          </div>
        )}
        
        {/* Collapsed mode - show logo with same size */}
        {collapsed && !isMobile && (
          <div className="flex justify-center">
            <div className="w-9 h-9 bg-white rounded-md flex items-center justify-center overflow-hidden shadow-sm border">
              <img 
                src="/icon.png" 
                alt="Logo" 
                className="w-7 h-7 object-contain"
              />
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 sm:p-3 overflow-y-auto">
        <div className="space-y-1">
          {visibleNavItems.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.path

            return (
              <Button
                key={item.path}
                variant="ghost"
                className={cn(
                  // base: make the button a flex container and vertically center contents
                  "w-full h-auto p-1.5 sm:p-2 flex items-center transition-colors rounded-md",
                  // when collapsed (desktop) center horizontally and remove extra horizontal padding
                  collapsed && !isMobile ? "justify-center text-center px-0 py-1.5" : "justify-start text-left px-1.5 sm:px-2",
                  isActive
                    ? // Enhanced orange color scheme for better visibility
                    "bg-orange-500 text-white hover:bg-orange-600 shadow-sm"
                    : "text-sidebar-foreground hover:bg-gray-100 hover:text-gray-900",
                )}
                onClick={() => handleNavigation(item.path)}
              >
                <Icon className={cn(
                  "h-4 w-4 sm:h-5 sm:w-5 flex-shrink-0", 
                  collapsed && !isMobile ? "" : "mr-2 sm:mr-3"
                )} />
                {(!collapsed || isMobile) && (
                  <div className="flex flex-col items-start min-w-0 flex-1">
                    <span className="font-medium text-xs sm:text-sm truncate w-full">{item.label}</span>
                    {item.description && (
                      <span className="text-xs opacity-60 truncate w-full hidden sm:block">
                        {item.description}
                      </span>
                    )}
                  </div>
                )}
              </Button>
            )
          })}
        </div>
      </nav>

      {/* Status Indicator */}
      {(!collapsed || isMobile) && (
        <div className="p-3 sm:p-4 border-t border-gray-200">
          <div className="bg-gray-50 rounded-lg p-2 sm:p-3">
            <div className="flex items-center gap-2 mb-1 sm:mb-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse flex-shrink-0"></div>
              <span className="text-xs font-medium text-gray-700 truncate">สถานะระบบ</span>
            </div>
            <p className="text-xs text-gray-600 truncate">เชื่อมต่อแล้ว</p>
            <p className="text-xs text-gray-500 truncate">กล้อง: 2/2 ออนไลน์</p>
          </div>
        </div>
      )}

      {/* Bottom Navigation */}
      <div className="p-3 sm:p-4 border-t border-gray-200">
        <div className="space-y-1">
          {visibleBottomNavItems.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.path

            return (
              <Button
                key={item.path}
                variant="ghost"
                className={cn(
                  // base: flex with vertical centering
                  "w-full h-auto p-2 flex items-center transition-colors rounded-lg",
                  collapsed && !isMobile ? "justify-center text-center px-0" : "justify-start text-left px-2",
                  isActive
                    ? // Enhanced orange color scheme for bottom navigation
                    "bg-orange-500 text-white hover:bg-orange-600 shadow-sm"
                    : "text-gray-600 hover:bg-gray-100 hover:text-gray-900",
                )}
                onClick={() => handleNavigation(item.path)}
              >
                <Icon className={cn("h-4 w-4 flex-shrink-0", collapsed && !isMobile ? "" : "mr-2")} />
                {(!collapsed || isMobile) && <span className="text-xs sm:text-sm truncate">{item.label}</span>}
              </Button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
