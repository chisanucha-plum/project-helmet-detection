/**
 * CSV export helper shared by dashboard and history views.
 * Prepends a UTF-8 BOM so Excel opens Thai text correctly.
 */

export function downloadCsv(filename: string, rows: Record<string, string | number>[]): void {
  if (rows.length === 0) return

  const headers = Object.keys(rows[0])
  const escapeCell = (value: string | number): string => {
    const cell = String(value)
    return /[",\n]/.test(cell) ? `"${cell.replace(/"/g, '""')}"` : cell
  }

  const lines = [
    headers.join(","),
    ...rows.map((row) => headers.map((h) => escapeCell(row[h] ?? "")).join(",")),
  ]

  const blob = new Blob(["\uFEFF" + lines.join("\n")], { type: "text/csv;charset=utf-8;" })
  const url = URL.createObjectURL(blob)

  const link = document.createElement("a")
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
