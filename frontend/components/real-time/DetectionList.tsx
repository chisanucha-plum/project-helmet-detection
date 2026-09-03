"use client"

import { Badge } from "@/components/ui/badge"
import { AlertTriangle, Camera, CheckCircle, Users } from "lucide-react"
import { memo, useRef, useState } from "react"
import { useVirtualizer } from "@tanstack/react-virtual"
import { DetectionModal } from "./DetectionModal"
import { API_BASE_URL } from "@/lib/api/config"
import type { DetectionResult } from "@/types/detection.types"

interface DetectionListProps {
  detections: DetectionResult[]
  t: (key: string) => string
}

/** memo: page-level state (filters, loading) re-renders without item props changing —
 *  detection/t identities are stable, so 500 items must not re-render with them */
const DetectionItem = memo(function DetectionItem({ detection, t }: { detection: DetectionResult; t: (key: string) => string }) {
  const [showModal, setShowModal] = useState(false)

  return (
    <>
      <div className="flex flex-row gap-3 p-3 bg-muted rounded-lg border border-border/50">
        {detection.framePath ? (
          <div 
            className="flex-shrink-0 w-20 h-20 rounded-md overflow-hidden bg-background border border-border cursor-pointer hover:ring-2 hover:ring-primary transition-all"
            onClick={() => setShowModal(true)}
          >
            <img src={`${API_BASE_URL}/helmet/frame/${detection.framePath}`} alt={`Detection ${detection.id}`} className="w-full h-full object-cover" loading="lazy" />
          </div>
        ) : (
          <div className="flex-shrink-0 w-20 h-20 rounded-md bg-muted border border-border flex items-center justify-center">
            <Camera className="h-6 w-6 text-muted-foreground" />
          </div>
        )}
        <div className="flex-1 flex flex-col justify-between gap-2 min-w-0">
          <div>
            <div className="text-xs text-muted-foreground">{detection.timestamp}</div>
            <div className="flex items-center gap-2 mt-1">
              {detection.helmetStatus === "wearing" ? <CheckCircle className="h-4 w-4 text-success-foreground flex-shrink-0" /> : <AlertTriangle className="h-4 w-4 text-critical-foreground flex-shrink-0" />}
              <span className={`text-sm font-semibold truncate ${detection.helmetStatus === "wearing" ? "text-success-foreground" : "text-critical-foreground"}`}>
                {detection.helmetStatus === "wearing" ? t("detection.wearingHelmet") : t("detection.notWearingHelmet")}
              </span>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <div className="flex items-center gap-1">
              <Users className="h-3 w-3 text-muted-foreground" />
              <span>{detection.passengerCount} {t("detection.passengers")}</span>
              {detection.passengerCount > 2 && <Badge variant="destructive" className="ml-1 text-xs px-1 py-0">{t("detection.overCapacityBadge")}</Badge>}
            </div>
            {detection.violation && <Badge variant="destructive" className="text-xs px-1.5 py-0">{t("detection.violation")}</Badge>}
            <span className="text-xs text-muted-foreground">{detection.camera}</span>
          </div>
        </div>
      </div>

      <DetectionModal
        isOpen={showModal}
        imageUrl={`${API_BASE_URL}/helmet/frame/${detection.framePath}`}
        onClose={() => setShowModal(false)}
      />
    </>
  )
})

/** Virtualizer: history loads 500 rows — mounting all costs ~10k DOM nodes
 *  and heavy layout/image decode. Only visible rows (+overscan) exist in the DOM.
 *  Own scroll container because the app scrolls inside <main>, not the window. */
export function DetectionList({ detections, t }: DetectionListProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  const virtualizer = useVirtualizer({
    count: detections.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 120,
    overscan: 6,
  })

  if (detections.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <Camera className="h-12 w-12 mx-auto mb-3 opacity-50" />
        <p>{t("detection.noDetections")}</p>
      </div>
    )
  }

  return (
    <div ref={scrollRef} className="max-h-[calc(100vh-280px)] overflow-auto rounded-lg">
      <div className="relative" style={{ height: virtualizer.getTotalSize() }}>
        {virtualizer.getVirtualItems().map((virtualRow) => (
          <div
            key={detections[virtualRow.index].id}
            ref={virtualizer.measureElement}
            data-index={virtualRow.index}
            className="absolute left-0 w-full pb-4"
            style={{ transform: `translateY(${virtualRow.start}px)` }}
          >
            <DetectionItem detection={detections[virtualRow.index]} t={t} />
          </div>
        ))}
      </div>
    </div>
  )
}
