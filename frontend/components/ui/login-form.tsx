"use client"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { getCurrentUser, loginWithApi } from "@/app/api/auth"
import { setStoredCurrentUser } from "@/stores/auth-store"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} 
from "@/components/ui/card"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "./field"
import { Input } from "@/components/ui/input"
import { useRouter } from "next/navigation"
import { FormEvent, useState } from "react"
import { useLanguage } from "@/hooks/useLanguage"
import { LanguageSelector } from "@/components/LanguageSelector"

export function LoginForm({
  className,
  ...props
}: React.ComponentProps<"div">) {
  const router = useRouter()
  const { language, setLang } = useLanguage("en")
  const { t } = useLanguage("en")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const response = await loginWithApi({ email, password })
      const currentUser = response.user ?? (await getCurrentUser(response.access_token))
      const appRole = currentUser.role
      const userName = currentUser.full_name || currentUser.username || currentUser.email || "User"

      localStorage.setItem("token", response.access_token)
      setStoredCurrentUser({
        role: appRole,
        email: currentUser.email || email,
        fullName: userName,
        username: currentUser.username,
      })

      router.push("/real-time-monitoring")
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message)
      } else {
        setError(t("login.errorMessage"))
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <div className="flex justify-end">
        <LanguageSelector currentLanguage={language as "en" | "th"} onLanguageChange={setLang} />
      </div>
      <Card>
        <CardHeader>
          <div className="flex justify-center">
            <img src="/icon.png" alt="KMUT logo" className="h-30 w-30 object-contain" />
          </div>
          <CardTitle></CardTitle>
          <CardDescription className="text-center text-sm font-bold leading-none  ">
            {t("login.title")}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="email">{t("login.email")}</FieldLabel>
                <Input
                  id="email"
                  type="email"
                  placeholder="kmutt@example.com"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  required
                />
              </Field>
              <Field>
                <div className="flex items-center">
                  <FieldLabel htmlFor="password">{t("login.password")}</FieldLabel>
                </div>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  required
                />
              </Field>
              {error && (
                <p className="text-sm text-red-600" role="alert">
                  {error}
                </p>
              )}
              <Field>
                <Button
                  type="submit"
                  className="bg-orange-500 text-white hover:bg-orange-600"
                  disabled={loading}
                >
                  {loading ? (
                    <span className="inline-flex items-center gap-1.5">
                      <span>{t("buttons.login")}</span>
                      <span className="inline-flex items-center gap-1" aria-hidden="true">
                        <span className="h-1.5 w-1.5 rounded-full bg-red-400 animate-pulse [animation-delay:0ms]" />
                        <span className="h-1.5 w-1.5 rounded-full bg-yellow-300 animate-pulse [animation-delay:180ms]" />
                        <span className="h-1.5 w-1.5 rounded-full bg-green-300 animate-pulse [animation-delay:360ms]" />
                      </span>
                    </span>
                  ) : (
                    t("buttons.login")
                  )}
                </Button>
                <Button
                  type="button"
                  className="bg-white text-black hover:bg-orange-600"
                  disabled={loading}
                >
                  {t("buttons.loginWithGoogle")}
                </Button>
                <FieldDescription className="text-center">
                  {/* Don&apos;t have an account? <a href="#">Sign up</a> */}
                </FieldDescription>
              </Field>
            </FieldGroup>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
