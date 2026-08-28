/**
 * Short two-tone violation beep via WebAudio — no audio asset needed.
 * The AudioContext is created lazily; browsers allow it after any user
 * gesture (e.g. pressing "start recording"), which always precedes SSE events.
 */

let audioContext: AudioContext | null = null

export function playViolationBeep(): void {
  try {
    const ctx = audioContext ?? new AudioContext()
    audioContext = ctx
    if (ctx.state === "suspended") void ctx.resume()

    const start = ctx.currentTime
    ;[0, 0.2].forEach((offset, index) => {
      const oscillator = ctx.createOscillator()
      const gain = ctx.createGain()
      oscillator.type = "sine"
      oscillator.frequency.value = index === 0 ? 880 : 660
      gain.gain.setValueAtTime(0.0001, start + offset)
      gain.gain.exponentialRampToValueAtTime(0.25, start + offset + 0.02)
      gain.gain.exponentialRampToValueAtTime(0.0001, start + offset + 0.16)
      oscillator.connect(gain).connect(ctx.destination)
      oscillator.start(start + offset)
      oscillator.stop(start + offset + 0.18)
    })
  } catch {
    // audio unavailable (unsupported browser, autoplay policy) — stay silent
  }
}
