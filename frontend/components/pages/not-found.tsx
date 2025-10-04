"use client"

import { useRouter } from "next/navigation"
import { Home, ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface NotFoundProps {
  readonly message?: string
}

export function NotFound({ message = "ไม่พบหน้าที่คุณต้องการ" }: Readonly<NotFoundProps>) {
  const router = useRouter()

  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <Card className="w-full max-w-md text-center">
        <CardHeader>
          <div className="w-20 h-20 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-4xl font-bold text-muted-foreground">404</span>
          </div>
          <CardTitle className="text-xl">หน้าไม่พบ</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground">{message}</p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button onClick={() => router.back()} variant="outline" className="gap-2">
              <ArrowLeft className="w-4 h-4" />
              ย้อนกลับ
            </Button>
            <Button onClick={() => router.push("/")} className="gap-2">
              <Home className="w-4 h-4" />
              หน้าหลัก
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
