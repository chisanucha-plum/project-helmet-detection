import { useCallback, useEffect, useState } from "react"

type Theme = "light" | "dark"

// Global event bus for theme changes (same pattern as useLanguage)
const themeChangeEvent = "theme-changed"

function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark")
}

export function useTheme(defaultTheme: Theme = "light") {
  const [theme, setThemeState] = useState<Theme>(defaultTheme)

  // Load initial theme from localStorage
  useEffect(() => {
    let saved: string | null = null
    try {
      saved = localStorage.getItem("theme")
    } catch {
      // storage unavailable (private mode etc.) — fall back to default
    }
    const initial: Theme = saved === "dark" || saved === "light" ? saved : defaultTheme
    setThemeState(initial)
    applyTheme(initial)
  }, [defaultTheme])

  // Listen for theme changes from other components
  useEffect(() => {
    const handleThemeChange = (e: Event) => {
      const customEvent = e as CustomEvent<Theme>
      setThemeState(customEvent.detail)
    }

    window.addEventListener(themeChangeEvent, handleThemeChange)
    return () => {
      window.removeEventListener(themeChangeEvent, handleThemeChange)
    }
  }, [])

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next)
    applyTheme(next)
    try {
      localStorage.setItem("theme", next)
    } catch {
      // keep the in-memory + DOM state even if persistence fails
    }
    // Dispatch event to notify all other components
    window.dispatchEvent(new CustomEvent<Theme>(themeChangeEvent, { detail: next }))
  }, [])

  const toggleTheme = useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark")
  }, [theme, setTheme])

  return { theme, setTheme, toggleTheme }
}
