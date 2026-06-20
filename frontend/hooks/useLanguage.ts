import { useState, useEffect, useCallback } from "react"
import enTranslations from "@/locales/en.json"
import thTranslations from "@/locales/th.json"

type Language = "en" | "th"

type Translations = {
  [key: string]: string | Translations
}

// Global event bus for language changes
const languageChangeEvent = "language-changed"

const translationsMap: Record<Language, Translations> = {
  en: enTranslations,
  th: thTranslations,
}

export function useLanguage(defaultLang: Language = "en") {
  const [language, setLanguageState] = useState<Language>(defaultLang)
  const [isLoading, setIsLoading] = useState(true)

  // Load initial language from localStorage
  useEffect(() => {
    setIsLoading(true)
    const saved = localStorage.getItem("language") as Language | null
    if (saved && ["en", "th"].includes(saved)) {
      setLanguageState(saved)
    } else {
      setLanguageState(defaultLang)
    }
    setIsLoading(false)
  }, [defaultLang])

  // Listen for language changes from other components
  useEffect(() => {
    const handleLanguageChange = (e: Event) => {
      const customEvent = e as CustomEvent<Language>
      setLanguageState(customEvent.detail)
    }

    window.addEventListener(languageChangeEvent, handleLanguageChange)
    return () => {
      window.removeEventListener(languageChangeEvent, handleLanguageChange)
    }
  }, [])

  const t = useCallback((key: string): string => {
    const translations = translationsMap[language]
    const keys = key.split(".")
    let value: any = translations

    for (const k of keys) {
      if (value && typeof value === "object" && k in value) {
        value = value[k]
      } else {
        return key
      }
    }

    return typeof value === "string" ? value : key
  }, [language])

  const setLang = useCallback((lang: Language) => {
    setLanguageState(lang)
    localStorage.setItem("language", lang)
    // Dispatch event to notify all other components
    const event = new CustomEvent<Language>(languageChangeEvent, { detail: lang })
    window.dispatchEvent(event)
  }, [])

  return { language, t, setLang, isLoading }
}
