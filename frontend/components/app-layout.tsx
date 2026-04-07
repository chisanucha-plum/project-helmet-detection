"use client"

import { ErrorBoundary } from "@/components/error-boundary"
import { Header } from "@/components/header"
import { Sidebar } from "@/components/sidebar"
import { getStoredUserRole } from "@/stores/auth-store"
import { usePathname } from "next/navigation"
import { useRouter } from "next/navigation"
import type React from "react"
import { useEffect, useState } from "react"

interface AppLayoutProps {
  children: React.ReactNode
}

export function AppLayout({ children }: AppLayoutProps) {
  const pathname = usePathname()
  const router = useRouter()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 1024
      setIsMobile(mobile)
      if (mobile) {
        setSidebarCollapsed(true)
        setSidebarOpen(false)
      }
    }

    handleResize()
    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
    }
  }, [])

  useEffect(() => {
    if (pathname !== "/dashboard") return

    const role = getStoredUserRole()
    if (role !== "admin") {
      router.replace("/real-time-monitoring")
    }
  }, [pathname, router])

  const toggleSidebar = () => {
    if (isMobile) {
      setSidebarOpen(!sidebarOpen)
    } else {
      setSidebarCollapsed(!sidebarCollapsed)
    }
  }

  const closeMobileSidebar = () => {
    if (isMobile) {
      setSidebarOpen(false)
    }
  }

  if (pathname === "/") {
    return <>{children}</>
  }

  return (
    <ErrorBoundary>
      <div className="flex h-screen bg-background overflow-hidden">
        {/* Mobile Overlay */}
        {isMobile && sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-40 lg:hidden transition-opacity duration-300"
            onClick={closeMobileSidebar}
          />
        )}

        {/* Sidebar */}
        <div
          className={`
            ${isMobile ? "fixed left-0 top-0 h-full z-50" : "relative"}
            ${isMobile && !sidebarOpen ? "-translate-x-full" : "translate-x-0"}
            transition-transform duration-300 ease-in-out
          `}
        >
          <Sidebar
            collapsed={sidebarCollapsed}
            onToggle={toggleSidebar}
            onNavigate={closeMobileSidebar}
            isMobile={isMobile}
          />
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden min-w-0">
          <Header onMenuClick={toggleSidebar} sidebarCollapsed={sidebarCollapsed} isMobile={isMobile} />
          <main className="flex-1 overflow-auto p-4 sm:p-6">
            <ErrorBoundary>{children}</ErrorBoundary>
          </main>
        </div>
      </div>
    </ErrorBoundary>
  )
}
