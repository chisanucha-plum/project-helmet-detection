// In-memory storage for access token (safer than localStorage for XSS prevention)
let inMemoryAccessToken: string | null = null

export function setAccessToken(token: string | null): void {
    inMemoryAccessToken = token
}

export function getAccessToken(): string | null {
    return inMemoryAccessToken
}

// Backward compatibility alias
export function getStoredAccessToken(): string | null {
    return inMemoryAccessToken
}

export function createAuthHeadersFromStore(): Record<string, string> {
    try {
        const token = inMemoryAccessToken
        return token ? { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' }
    } catch {
        return { 'Content-Type': 'application/json' }
    }
}

export const AUTH_USER_UPDATED_EVENT = "auth-user-updated"

// In-memory storage for user role and email (safer than localStorage)
let inMemoryUserRole: UserRole = null
let inMemoryUserEmail: string | null = null

export type UserRole = 'admin' | 'security' | 'user' | null

export function getStoredUserRole(): UserRole {
    return inMemoryUserRole
}

export function isAdminRole(): boolean {
    return inMemoryUserRole === 'admin'
}

export function getStoredUserEmail(): string | null {
    return inMemoryUserEmail
}

export function setStoredCurrentUser(params: {
    role: string | null | undefined
    email: string | null | undefined
    fullName?: string | null
    username?: string | null
}): void {
    const normalizedRole =
        params.role === 'admin' || params.role === 'security' || params.role === 'user'
            ? params.role
            : null

    inMemoryUserRole = normalizedRole
    inMemoryUserEmail = params.email && params.email.trim().length > 0 ? params.email : null

    if (typeof window !== 'undefined') {
        window.dispatchEvent(new Event(AUTH_USER_UPDATED_EVENT))
    }
}
