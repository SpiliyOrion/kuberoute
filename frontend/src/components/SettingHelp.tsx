import { useEffect, useId, useRef, useState } from 'react'
import type { ReactNode } from 'react'

interface SettingHelpProps {
  label: string
  children: ReactNode
}

export function SettingHelp({ label, children }: SettingHelpProps) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const container = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const closeOutside = (event: PointerEvent) => {
      if (event.target instanceof Node && !container.current?.contains(event.target)) setOpen(false)
    }
    document.addEventListener('pointerdown', closeOutside)
    return () => document.removeEventListener('pointerdown', closeOutside)
  }, [open])

  return <div className="setting-help" ref={container} onKeyDown={(event) => {
    if (event.key === 'Escape') { setOpen(false); event.stopPropagation() }
  }}>
    <button type="button" className="help-trigger" aria-label={`${label} help`} aria-expanded={open} aria-controls={open ? id : undefined} onClick={() => setOpen(!open)}>i</button>
    {open && <div className="help-popover" id={id} role="note">{children}</div>}
  </div>
}
