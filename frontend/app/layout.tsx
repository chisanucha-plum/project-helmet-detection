import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import { Suspense } from "react"
import { AppLayout } from "@/components/app-layout"
import "./globals.css"

export const metadata: Metadata = {
  title: "ระบบตรวจจับหมวกกันน็อค - มหาวิทยาลัย",
  description: "ระบบตรวจจับการสวมหมวกกันน็อคแบบ Real-time สำหรับมหาวิทยาลัย",
  generator: "v0.app",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="th">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable} antialiased`}>
        <AppLayout>
          <Suspense fallback={null}>{children}</Suspense>
        </AppLayout>
        <Analytics />
      </body>
    </html>
  )
}
