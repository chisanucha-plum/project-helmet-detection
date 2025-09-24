"use client"

// Logo will be loaded from /icon.png

export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-background">
      <div className="text-center">
        <div className="relative">
          <div className="w-20 h-20 bg-gray-100 rounded-full animate-pulse mb-4 mx-auto border shadow-sm"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <img 
              src="/icon.png" 
              alt="Logo" 
              className="w-16 h-16 object-contain animate-bounce"
            />
          </div>
        </div>
        <h2 className="text-lg font-semibold text-foreground mb-2">กำลังโหลดระบบ</h2>
        <p className="text-sm text-muted-foreground">กรุณารอสักครู่...</p>
      </div>
    </div>
  )
}
