"use client"

interface DetectionModalProps {
  isOpen: boolean
  imageUrl: string
  onClose: () => void
}

export function DetectionModal({ isOpen, imageUrl, onClose }: DetectionModalProps) {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="relative max-w-4xl max-h-[90vh] w-full">
        <button onClick={onClose} className="absolute top-2 right-2 bg-black/50 text-white rounded-full p-2 hover:bg-black/70 z-10">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        <img src={imageUrl} alt="Detection Frame" className="w-full h-full object-contain rounded-lg" onClick={(e) => e.stopPropagation()} />
      </div>
    </div>
  )
}
