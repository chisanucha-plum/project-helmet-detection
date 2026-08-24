"use client"

import { Moon, Sun } from "lucide-react"
import { useEffect, useState } from "react"

import { Button } from "@/components/ui/button"
import { useLanguage } from "@/hooks/useLanguage"
import { useTheme } from "@/hooks/useTheme"

export function ThemeToggle() {
  const { t } = useLanguage("en")
  const { theme, setTheme } = useTheme("light")
  // Avoid SSR/client markup mismatch: icon depends on DOM state only known after mount
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const isDark = mounted && theme === "dark"

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label={t("buttons.toggleTheme")}
      title={t("buttons.toggleTheme")}
      className="hover:bg-muted hover:text-foreground h-6 sm:h-7 rounded-md px-2"
    >
      {isDark ? <Sun className="h-3 w-3 sm:h-4 sm:w-4" /> : <Moon className="h-3 w-3 sm:h-4 sm:w-4" />}
    </Button>
  )
}
