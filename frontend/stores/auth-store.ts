export function createAuthHeadersFromStore(): Record<string, string> {
    // Placeholder: read token from localStorage or other store in a real app
    try {
        const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null
        return token ? { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' }
    } catch {
        return { 'Content-Type': 'application/json' }
    }
}

export const AUTH_USER_UPDATED_EVENT = "auth-user-updated"

export function getStoredAccessToken(): string | null {
    try {
        if (typeof window === 'undefined') return null
        return localStorage.getItem('token')
    } catch {
        return null
    }
}

export type UserRole = 'admin' | 'security' | 'user' | null

export function getStoredUserRole(): UserRole {
    try {
        if (typeof window === 'undefined') return null
        const role = localStorage.getItem('userRole')
        if (role === 'admin' || role === 'security' || role === 'user') return role
        return null
    } catch {
        return null
    }
}

export function isAdminRole(): boolean {
    return getStoredUserRole() === 'admin'
}

export function getStoredUserEmail(): string | null {
    try {
        if (typeof window === 'undefined') return null
        const email = localStorage.getItem('userEmail')
        return email && email.trim().length > 0 ? email : null
    } catch {
        return null
    }
}

export function setStoredCurrentUser(params: {
    role: string | null | undefined
    email: string | null | undefined
    fullName?: string | null
    username?: string | null
}): void {
    try {
        if (typeof window === 'undefined') return

        const normalizedRole =
            params.role === 'admin' || params.role === 'security' || params.role === 'user'
                ? params.role
                : null

        if (normalizedRole) {
            localStorage.setItem('userRole', normalizedRole)
        } else {
            localStorage.removeItem('userRole')
        }

        if (params.email && params.email.trim().length > 0) {
            localStorage.setItem('userEmail', params.email)
        }

        const userName = params.fullName || params.username || params.email || null
        if (userName) {
            localStorage.setItem('userName', userName)
        }
    } catch {
        // Ignore storage errors in helper
    }
}
