"use client"

import { Shield } from "lucide-react"

export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-background">
      <div className="text-center">
        <div className="relative">
          <div className="w-16 h-16 bg-primary/20 rounded-full animate-pulse mb-4 mx-auto"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <Shield className="w-8 h-8 text-primary animate-bounce" />
          </div>
        </div>
        <h2 className="text-lg font-semibold text-foreground mb-2">กำลังโหลดระบบ</h2>
        <p className="text-sm text-muted-foreground">กรุณารอสักครู่...</p>
      </div>
    </div>
  )
}
