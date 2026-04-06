import { API_BASE_URL } from "./config"

export type UserRole = "admin" | "user"

export type LoginRequest = {
    email: string
    password: string
}

export type LoginResponse = {
    access_token: string
    token_type: "bearer"
    expires_at: string
    user: {
        id: number
        name: string
        email: string
        role: UserRole
    }
}

export async function loginWithMockApi(payload: LoginRequest): Promise<LoginResponse> {
    const res = await fetch(`${API_BASE_URL}/mock/login`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    })

    if (!res.ok) {
        const message = await safeReadErrorMessage(res)
        throw new Error(message || "Login failed")
    }

    return (await res.json()) as LoginResponse
}

async function safeReadErrorMessage(res: Response): Promise<string | null> {
    try {
        const data = (await res.json()) as { detail?: string }
        return data.detail ?? null
    } catch {
        return null
    }
}
