export function createAuthHeadersFromStore(): Record<string, string> {
    // Placeholder: read token from localStorage or other store in a real app
    try {
        const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null
        return token ? { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' }
    } catch {
        return { 'Content-Type': 'application/json' }
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
