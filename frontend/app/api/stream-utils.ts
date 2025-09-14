/**
 * Helper: convert a ReadableStream (from fetch response.body) into an async iterator of lines.
 * Each yielded value is a string (one line) trimmed.
 */
export async function* streamToLines(body: ReadableStream<Uint8Array>) {
    const reader = body.getReader()
    const decoder = new TextDecoder()
    let buf = ""
    try {
        while (true) {
            const { done, value } = await reader.read()
            if (done) break
            buf += decoder.decode(value, { stream: true })
            let idx: number
            while ((idx = buf.indexOf("\n")) >= 0) {
                const line = buf.slice(0, idx).trim()
                buf = buf.slice(idx + 1)
                if (line) yield line
            }
        }
        if (buf.trim()) yield buf.trim()
    } finally {
        try {
            await reader.cancel()
        } catch { }
    }
}
