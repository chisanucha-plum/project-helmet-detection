import { createAuthHeadersFromStore } from "@/stores/auth-store"
import { API_BASE_URL } from "./config"
import { streamToLines } from "./stream-utils"

export type DetectionEvent = {
    id: string
    timestamp: string
    camera?: string
    license_plate?: string
    helmet: boolean
    person_count: number
    confidence?: number
    image_url?: string
}

/**
 * Fetch most recent detections (non-streaming)
 * @param limit number of records to fetch
 * @param abortSignal optional AbortSignal
 */
export async function fetchLatestDetections(limit = 50, abortSignal?: AbortSignal): Promise<DetectionEvent[]> {
    const params = new URLSearchParams({ limit: String(limit) }).toString()
    const res = await fetch(`${API_BASE_URL}/realtime/latest?${params}`, {
        method: "GET",
        headers: createAuthHeadersFromStore(),
        signal: abortSignal,
    })

    if (!res.ok) {
        throw new Error(`Failed to fetch latest detections: ${res.status}`)
    }

    return (await res.json()) as DetectionEvent[]
}

/**
 * Stream realtime detections from backend.
 * Tries Fetch streaming first (using streamToLines), falls back to SSE (EventSource) when available.
 * @param onEvent called for each parsed DetectionEvent
 * @param onError called on network/parse errors
 * @param abortSignal optional AbortSignal to cancel fetch streaming
 */
export async function streamRealtimeDetections(
    onEvent: (ev: DetectionEvent) => void,
    onError: (err: Error) => void,
    abortSignal?: AbortSignal
) {
    // Preferred streaming endpoint; adjust to your backend implementation
    const streamUrl = `${API_BASE_URL}/realtime/stream`

    // If running in a browser and EventSource is available, prefer SSE because it's simpler
    if (typeof window !== "undefined" && typeof (window as any).EventSource !== "undefined") {
        try {
            const url = streamUrl
            const es = new (window as any).EventSource(url, { withCredentials: false })

            es.onmessage = (ev: MessageEvent) => {
                try {
                    const data = JSON.parse(ev.data) as DetectionEvent
                    onEvent(data)
                } catch (err) {
                    console.warn("Failed to parse SSE message", err)
                }
            }

            es.onerror = (err: any) => {
                // forward an Error instance
                onError(new Error("SSE connection error"))
                try {
                    es.close()
                } catch { }
            }

            return () => {
                try {
                    es.close()
                } catch { }
            }
        } catch (err) {
            // fallthrough to fetch-stream approach
            console.warn("SSE unavailable, falling back to fetch streaming", err)
        }
    }

    // Fetch streaming approach (server-sent JSON lines)
    try {
        const res = await fetch(streamUrl, {
            method: "GET",
            headers: createAuthHeadersFromStore(),
            signal: abortSignal,
        })

        if (!res.ok) {
            throw new Error(`Streaming endpoint returned ${res.status}`)
        }

        if (!res.body) {
            throw new Error("Streaming response body is empty")
        }

        // streamToLines yields each JSON string (line) from the stream
        for await (const line of streamToLines(res.body)) {
            try {
                const parsed = JSON.parse(line) as DetectionEvent
                onEvent(parsed)
            } catch (err) {
                console.warn("Failed to parse streamed line", err, line)
            }
        }
    } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") {
            // aborted by caller - just return
            return
        }
        onError(err instanceof Error ? err : new Error("Unknown streaming error"))
    }
}

/**
 * Fetch current detection from /detect/helmet endpoint.
 * The frontend should call the server proxy at /api/proxy/detect/helmet
 */
// NOTE: The backend currently serves `/detect/helmet` as a streaming endpoint (newline-delimited JSON
// or SSE-style `data:` messages). Historically `fetchDetectHelmet` returned a single JSON payload,
// but at the moment the server streams events continuously. Keep this in mind – if the server
// changes to return a single JSON object in the future we can re-introduce a non-streaming helper.

/**
 * Async generator that connects to `/detect/helmet` and yields parsed DetectionEvent objects
 * as they arrive. Accepts an optional AbortSignal to cancel the stream.
 */
export async function* streamDetectHelmet(abortSignal?: AbortSignal): AsyncGenerator<DetectionEvent> {
    const url = `${API_BASE_URL.replace(/\/$/, '')}/detect/helmet`
    const res = await fetch(url, {
        method: 'GET',
        headers: createAuthHeadersFromStore(),
        signal: abortSignal,
    })

    if (!res.ok) {
        throw new Error(`Stream endpoint returned ${res.status}`)
    }

    if (!res.body) {
        throw new Error('Stream response has no body')
    }

    // Use existing streamToLines helper to turn the ReadableStream into lines of text
    for await (const line of streamToLines(res.body)) {
        let raw = line.trim()
        if (!raw) continue
        // handle SSE-style lines that start with `data:`
        if (raw.startsWith('data:')) raw = raw.replace(/^data:\s?/, '')

        try {
            const parsed = JSON.parse(raw) as DetectionEvent
            yield parsed
        } catch (err) {
            // ignore parse errors but log for debugging
            console.warn('Failed to parse /detect/helmet stream line', err, raw)
        }
    }
}

/**
 * Deprecated: fetching `/detect/helmet` as a single JSON payload is no longer supported
 * by the backend (it streams events). Call `streamDetectHelmet()` instead.
 */
export async function fetchDetectHelmet(_abortSignal?: AbortSignal): Promise<DetectionEvent | null> {
    throw new Error('fetchDetectHelmet is deprecated: /detect/helmet is a streaming endpoint. Use streamDetectHelmet()')
}
