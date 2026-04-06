import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import Link from "next/link"
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

export function LoginForm({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <Card>
        <CardHeader>
          <div className="flex justify-center">
            <img src="/icon.png" alt="KMUT logo" className="h-30 w-30 object-contain" />
          </div>
          <CardTitle></CardTitle>
          <CardDescription className="text-center text-sm font-bold leading-none  ">
            ระบบตรวจจับการขับขี่ - เทคโนโลยีพระจอมเกล้าธนบุรี
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form>
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="email">Email</FieldLabel>
                <Input
                  id="email"
                  type="email"
                  placeholder="kmutt@example.com"
                />
              </Field>
              <Field>
                <div className="flex items-center">
                  <FieldLabel htmlFor="password">Password</FieldLabel>
                  <a
                    href="#"
                    className="ml-auto inline-block text-sm underline-offset-4 hover:underline"
                  >
                    Forgot your password?
                  </a>
                </div>
                <Input id="password" type="password" />
              </Field>
              <Field>
                <Button asChild className="bg-orange-500 text-white hover:bg-orange-600">
                  <Link href="/real-time-monitoring">Login</Link>
                </Button>
                <Button
                  type="button"
                  className="bg-white text-black hover:bg-orange-600"
                >
                  Login with Google
                </Button>
                <FieldDescription className="text-center">
                  Don&apos;t have an account? <a href="#">Sign up</a>
                </FieldDescription>
              </Field>
            </FieldGroup>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
