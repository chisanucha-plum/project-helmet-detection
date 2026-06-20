import { API_BASE_URL } from "./config"

export type UserRole = "admin" | "security" | "user"

export type LoginRequest = {
    email: string
    password: string
}

export type LoginResponse = {
    access_token: string
    refresh_token: string
    user?: CurrentUserResponse
}

export type CurrentUserResponse = {
    id: string
    username: string | null
    email: string | null
    full_name: string | null
    role: UserRole
    created_at: string | null
    last_login: string | null
}

export async function loginWithApi(payload: LoginRequest): Promise<LoginResponse> {
    const res = await fetch(`${API_BASE_URL}/user/login`, {
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

export async function getCurrentUser(accessToken: string): Promise<CurrentUserResponse> {
    const res = await fetch(`${API_BASE_URL}/user/me`, {
        method: "GET",
        headers: {
            "Authorization": `Bearer ${accessToken}`,
            "Content-Type": "application/json",
        },
    })

    if (!res.ok) {
        const message = await safeReadErrorMessage(res)
        throw new Error(message || "Failed to fetch current user")
    }

    return (await res.json()) as CurrentUserResponse
}

export async function refreshAccessToken(): Promise<{ access_token: string; refresh_token: string }> {
    const res = await fetch(`${API_BASE_URL}/user/refresh_token`, {
        method: "POST",
        credentials: "include", // ส่ง cookies (refresh_token)
        headers: {
            "Content-Type": "application/json",
        },
    })

    if (!res.ok) {
        const message = await safeReadErrorMessage(res)
        throw new Error(message || "Failed to refresh token")
    }

    return (await res.json()) as { access_token: string; refresh_token: string }
}

async function safeReadErrorMessage(res: Response): Promise<string | null> {
    try {
        const data = (await res.json()) as { detail?: string }
        return data.detail ?? null
    } catch {
        return null
    }
}
