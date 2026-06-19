"use client"
import { Button } from "@/components/ui/button"
import { Bell, Menu } from "lucide-react"
import { LogOut } from "lucide-react"
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover"
import { cn } from "@/lib/utils"
import { usePathname, useRouter } from 'next/navigation'

interface HeaderProps {
  onMenuClick: () => void
  sidebarCollapsed: boolean
  isMobile?: boolean
}

export function Header({ onMenuClick, sidebarCollapsed, isMobile = false }: HeaderProps) {
  const router = useRouter()

  const handleLogout = () => {
    try {
      localStorage.removeItem('token')
      localStorage.removeItem('userRole')
      localStorage.removeItem('userName')
      localStorage.removeItem('userEmail')
    } catch {
      // ignore storage errors and continue logout flow
    }
    router.replace('/')
  }

  return (
    <header className="h-10 sm:h-12 bg-white border border-gray-300 flex items-center justify-between px-3 sm:px-4 rounded-2xl mx-3 mt-3 mb-1">
      <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={onMenuClick}
          className="hover:bg-gray-100 hover:text-gray-900 flex-shrink-0 rounded-md"
        >
          <Menu className="h-5 w-5" />
        </Button>

        <div className="flex items-center gap-1 sm:gap-2 min-w-0 flex-1">
          <div className="w-4 h-4 sm:w-6 sm:h-6 bg-white rounded-md flex items-center justify-center flex-shrink-0 lg:hidden overflow-hidden shadow-sm border">
            <img 
              src="/icon.png" 
              alt="Logo" 
              className="w-3 h-3 sm:w-4 sm:h-4 object-contain"
            />
          </div>
        </div>
      </div>

      <div className="flex items-center gap-1 flex-shrink-0">
        <Button
          variant="ghost"
          size="sm"
          onClick={handleLogout}
          className="hover:bg-gray-100 hover:text-gray-900 h-6 sm:h-7 rounded-md px-2"
        >
          <LogOut className="h-3 w-3 sm:h-4 sm:w-4" />
          <span className="ml-1 text-xs hidden sm:inline">Logout</span>
        </Button>
        <Popover>
          <PopoverTrigger asChild>
            {/* <Button
              variant="ghost"
              size="sm"
              className="relative hover:bg-gray-100 hover:text-gray-900 h-6 w-6 sm:h-7 sm:w-7 rounded-md p-0"
            >
              <Bell className="h-3 w-3 sm:h-4 sm:w-4" />
              <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 sm:w-2 sm:h-2 bg-red-500 rounded-full flex items-center justify-center">
                <span className="w-0.5 h-0.5 sm:w-1 sm:h-1 bg-white rounded-full"></span>
              </span>
            </Button> */} 
          </PopoverTrigger>
          <PopoverContent sideOffset={8} className="w-72">
            <NotificationList />
          </PopoverContent>
        </Popover>
      </div>
    </header>
  )
}

function NotificationList() {
  const router = useRouter()
  const pathname = usePathname()

  const navigateToResults = () => {
    // If already on home page, just scroll to element; otherwise navigate with hash
    if (pathname === '/') {
      // small timeout to allow popover close animation
      setTimeout(() => {
        const el = document.getElementById('detection-results')
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 120)
    } else {
      router.push('/#detection-results')
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium">Notifications</h4>
        <button className={cn("text-xs text-muted-foreground hover:underline","ml-2")}>Mark all read</button>
      </div>

      <ul className="flex flex-col gap-2">
        <li>
          <button onClick={navigateToResults} className="w-full text-left p-2 rounded-md hover:bg-accent/5">
            <div className="text-sm font-medium">พบการกระทำผิด: ไม่สวมหมวกกันน็อค</div>
            <div className="text-xs text-muted-foreground">กล้องหลัก — 2 นาทีที่แล้ว</div>
          </button>
        </li>
        <li>
          <button onClick={navigateToResults} className="w-full text-left p-2 rounded-md hover:bg-accent/5">
            <div className="text-sm font-medium">พบการกระทำผิด: นั่งเกิน 2 คน</div>
            <div className="text-xs text-muted-foreground">กล้องรอง — 10 นาทีที่แล้ว</div>
          </button>
        </li>
      </ul>
    </div>
  )
}
