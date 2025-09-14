export function createAuthHeadersFromStore(): Record<string, string> {
    // Placeholder: read token from localStorage or other store in a real app
    try {
        const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null
        return token ? { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' }
    } catch {
        return { 'Content-Type': 'application/json' }
    }
}
