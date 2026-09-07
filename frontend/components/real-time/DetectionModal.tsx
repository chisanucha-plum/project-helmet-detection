import { X } from "lucide-react"
import { useEffect } from "react"

interface DetectionModalProps {
  isOpen: boolean
  imageUrl: string
  onClose: () => void
}

export function DetectionModal({ isOpen, imageUrl, onClose }: DetectionModalProps) {
  // Prevent body scroll when modal is open
  useEffect(() => {
    if (!isOpen) return
    const originalOverflow = document.body.style.overflow
    document.body.style.overflow = "hidden"
    return () => {
      document.body.style.overflow = originalOverflow
    }
  }, [isOpen])

  // ESC key to close
  useEffect(() => {
    if (!isOpen) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    document.addEventListener("keydown", onKeyDown)
    return () => document.removeEventListener("keydown", onKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div 
      className="fixed inset-0 bg-black/80 flex items-center justify-center p-4" 
      style={{ zIndex: 9999 }}
      onClick={onClose}
    >
      <div className="relative max-w-4xl max-h-[90vh] w-full" onClick={(e) => e.stopPropagation()}>
        <button 
          onClick={onClose} 
          className="absolute -top-10 right-0 bg-white/10 text-white rounded-full p-2 hover:bg-white/20 backdrop-blur-sm transition-colors"
          aria-label="Close"
        >
          <X className="h-6 w-6" />
        </button>
        <img 
          src={imageUrl} 
          alt="Detection Frame" 
          className="w-full h-full object-contain rounded-lg bg-black"
          loading="eager"
        />
      </div>
    </div>
  )
}
