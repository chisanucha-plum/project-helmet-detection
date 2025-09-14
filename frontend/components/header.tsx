"use client"

import { Button } from "@/components/ui/button"
import { Bell, Menu, Settings, Shield, User } from "lucide-react"

interface HeaderProps {
  onMenuClick: () => void
  sidebarCollapsed: boolean
  isMobile?: boolean
}

export function Header({ onMenuClick, sidebarCollapsed, isMobile = false }: HeaderProps) {
  return (
    <header className="h-14 sm:h-16 bg-card border-b border-border flex items-center justify-between px-4 sm:px-6 shadow-sm">
      <div className="flex items-center gap-3 sm:gap-4 min-w-0 flex-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={onMenuClick}
          className="hover:bg-accent hover:text-accent-foreground flex-shrink-0"
        >
          <Menu className="h-5 w-5" />
        </Button>

        <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
          <div className="w-6 h-6 sm:w-8 sm:h-8 bg-primary rounded-lg flex items-center justify-center flex-shrink-0 lg:hidden">
            <Shield className="w-4 h-4 sm:w-5 sm:h-5 text-primary-foreground" />
          </div>
          <div className="min-w-0 flex-1">
            <h1 className="text-sm sm:text-lg font-semibold text-foreground truncate">
              {isMobile ? "ตรวจจับหมวกกันน็อค" : "ระบบตรวจจับหมวกกันน็อค"}
            </h1>
            <p className="text-xs text-muted-foreground hidden sm:block truncate">Real-time Helmet Detection System</p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-1 sm:gap-2 flex-shrink-0">
        <Button
          variant="ghost"
          size="sm"
          className="relative hover:bg-accent hover:text-accent-foreground h-8 w-8 sm:h-9 sm:w-9"
        >
          <Bell className="h-4 w-4 sm:h-5 sm:w-5" />
          <span className="absolute -top-1 -right-1 w-2 h-2 sm:w-3 sm:h-3 bg-destructive rounded-full flex items-center justify-center">
            <span className="w-1 h-1 sm:w-1.5 sm:h-1.5 bg-destructive-foreground rounded-full"></span>
          </span>
        </Button>

        <Button
          variant="ghost"
          size="sm"
          className="hover:bg-accent hover:text-accent-foreground h-8 w-8 sm:h-9 sm:w-9 hidden sm:flex"
        >
          <Settings className="h-4 w-4 sm:h-5 sm:w-5" />
        </Button>

        <Button
          variant="ghost"
          size="sm"
          className="hover:bg-accent hover:text-accent-foreground h-8 w-8 sm:h-9 sm:w-9"
        >
          <User className="h-4 w-4 sm:h-5 sm:w-5" />
        </Button>
      </div>
    </header>
  )
}
