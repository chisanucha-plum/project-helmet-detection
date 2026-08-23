"use client"

import { Button } from "@/components/ui/button"
import { Camera, Maximize, Minimize } from "lucide-react"
import { useCallback, useEffect, useState } from "react"

interface VideoStreamProps {
  mjpegUrl?: string
}

export function VideoStream({ mjpegUrl }: VideoStreamProps) {
  const [isFullscreen, setIsFullscreen] = useState(false)

  const toggleFullscreen = useCallback(() => {
    const container = document.getElementById("video-container") as HTMLDivElement | null
    if (!container) return
    if (!document.fullscreenElement) container.requestFullscreen?.()
    else document.exitFullscreen?.()
  }, [])

  useEffect(() => {
    const handleChange = () => setIsFullscreen(!!document.fullscreenElement)
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === "Escape" && document.fullscreenElement) document.exitFullscreen?.()
      else if (e.key.toLowerCase() === "f" && !document.fullscreenElement) toggleFullscreen()
    }
    document.addEventListener("fullscreenchange", handleChange)
    document.addEventListener("keydown", handleKeyPress)
    return () => {
      document.removeEventListener("fullscreenchange", handleChange)
      document.removeEventListener("keydown", handleKeyPress)
    }
  }, [toggleFullscreen])

  return (
    <div id="video-container" className={`aspect-video bg-muted rounded-lg flex items-center justify-center relative overflow-hidden ${isFullscreen ? "fixed inset-0 z-50 bg-black rounded-none aspect-auto" : ""}`}>
      {mjpegUrl ? (
        <img id="mjpeg-stream" src={mjpegUrl} alt="Live MJPEG" className={`absolute inset-0 w-full h-full ${isFullscreen ? "object-contain" : "object-cover"}`} />
      ) : (
        <div className="z-10 text-center">
          <Camera className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
          <span className="text-muted-foreground">Live Video Feed</span>
        </div>
      )}
      <div className="absolute top-3 left-3 flex items-center gap-1 bg-red-500 text-white px-2 py-1 rounded text-xs z-20">
        <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
        REC
      </div>
      {mjpegUrl && (
        <Button size="sm" variant="secondary" onClick={toggleFullscreen} className="absolute top-3 right-3 z-20">
          {isFullscreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
        </Button>
      )}
      {isFullscreen && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20">
          <div className="flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded-lg px-4 py-2 text-white">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-sm">Camera 1 - Rider Detection</span>
            <Button size="sm" variant="ghost" onClick={toggleFullscreen} className="text-white hover:bg-white/20">
              <Minimize className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
