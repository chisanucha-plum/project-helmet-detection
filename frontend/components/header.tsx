"use client"
import { Button } from "@/components/ui/button"
import { LogOut, Menu } from "lucide-react"
import { useRouter } from 'next/navigation'
import { useLanguage } from "@/hooks/useLanguage"
import { LanguageSelector } from "@/components/LanguageSelector"
import { ThemeToggle } from "@/components/theme-toggle"
import { logoutApi } from "@/lib/api/auth"
import { clearStoredCurrentUser } from "@/stores/auth-store"

interface HeaderProps {
  onMenuClick: () => void
}

export function Header({ onMenuClick }: HeaderProps) {
  const router = useRouter()
  const { language, setLang, t } = useLanguage("en")

  const handleLogout = async () => {
    clearStoredCurrentUser()

    // Reset language preference (existing behavior on logout)
    try {
      localStorage.removeItem('language')
    } catch {
      // ignore storage errors and continue logout flow
    }

    await logoutApi() // best effort — clears refresh-token cookie
    router.replace('/')
  }

  return (
    <header className="h-10 sm:h-12 bg-card border border-border flex items-center justify-between px-3 sm:px-4 rounded-2xl mx-3 mt-3 mb-1">
      <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={onMenuClick}
          className="hover:bg-muted hover:text-foreground flex-shrink-0 rounded-md"
        >
          <Menu className="h-5 w-5" />
        </Button>

        <div className="flex items-center gap-1 sm:gap-2 min-w-0 flex-1">
          <div className="w-4 h-4 sm:w-6 sm:h-6 bg-white dark:bg-card rounded-md flex items-center justify-center flex-shrink-0 lg:hidden overflow-hidden shadow-sm border border-border">
            <img
              src="/icon.png"
              alt="Logo"
              className="w-3 h-3 sm:w-4 sm:h-4 object-contain"
            />
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 flex-shrink-0">
        <LanguageSelector currentLanguage={language as "en" | "th"} onLanguageChange={setLang} />
        <ThemeToggle />
        <Button
          variant="ghost"
          size="sm"
          onClick={handleLogout}
          className="hover:bg-muted hover:text-foreground h-6 sm:h-7 rounded-md px-2"
        >
          <LogOut className="h-3 w-3 sm:h-4 sm:w-4" />
          <span className="ml-1 text-xs hidden sm:inline">{t("buttons.logout")}</span>
        </Button>
      </div>
    </header>
  )
}
