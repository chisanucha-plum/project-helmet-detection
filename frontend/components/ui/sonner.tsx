"use client"

import { useEffect, useState } from "react"
import { Toaster as Sonner, type ToasterProps } from "sonner"

type Theme = "light" | "dark"

/**
 * Toaster styled with the project's popover tokens. Tracks the project's own
 * theme mechanism (localStorage "theme" + "theme-changed" event), not
 * next-themes, since the app manages its theme class manually.
 */
export function Toaster(props: ToasterProps) {
  const [theme, setTheme] = useState<Theme>("light")

  useEffect(() => {
    const sync = () => {
      try {
        setTheme(localStorage.getItem("theme") === "dark" ? "dark" : "light")
      } catch {
        // storage unavailable — keep current state
      }
    }

    sync()
    window.addEventListener("theme-changed", sync)
    return () => window.removeEventListener("theme-changed", sync)
  }, [])

  return (
    <Sonner
      theme={theme}
      position="bottom-right"
      richColors
      toastOptions={{
        classNames: {
          toast: "group toast group-[.toaster]:bg-popover group-[.toaster]:text-popover-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg",
          description: "group-[.toast]:text-muted-foreground",
        },
      }}
      {...props}
    />
  )
}
