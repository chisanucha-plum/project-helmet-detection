# Frontend changes in this branch

This document summarizes the frontend changes made on branch `feat/fe/home-page` and provides quick verification steps for reviewers.

## What changed

- UI: Adjusted the sidebar active state to center the icon when the sidebar is collapsed.
  - File: `frontend/components/sidebar.tsx`
  - Behavior: When collapsed the nav button uses centered layout; when expanded the label + icon layout remains.

- TypeScript config: Removed references to `.next` build types and a Next plugin from `tsconfig.json`.
  - File: `frontend/tsconfig.json`
  - Reason: `.next` is a build artifact and was removed; keeping `.next` types in `include` caused TS errors. The plugin entry caused a path-normalization Debug Failure on Windows, so it was removed.

- Cleanup: Deleted `frontend/vite.config.ts` (untracked) and cleared `frontend/.next/` (build artifacts).
  - Note: `vite.config.ts` was untracked (no git history). If this was needed, restore from local backup or recreate.

- Documentation: Added `frontend/README.md` with install/run/troubleshooting notes (Thai).

## How to verify

1. Install dependencies and run development server

```powershell
cd frontend
npm install
npm run dev
```

2. Or build + start (production-like) to recreate `.next`

```powershell
cd frontend
npm install
npm run build
npm run start
```

3. Check these acceptance criteria
- Sidebar active icon is centered when collapsed (test desktop + mobile/responsive)
- No TypeScript/Next errors during dev or build
- No `.next` build artifacts are committed to git

## Troubleshooting
- If `npm install` fails with peer dependency errors, try:

```powershell
npm install --legacy-peer-deps
```

- If you need the deleted `vite.config.ts`, it was untracked; recover from a local backup or re-create it.

- If any IDE shows path-related Debug Failure on Windows, ensure `tsconfig.json` does not include `.next` plugin entries that rely on a specific path format.

## Notes
- This change intentionally removes `.next` references from `tsconfig.json` to avoid build/runtime errors when `.next` is not present. If you rely on Next-specific TS plugins, consider adding them conditionally or fixing path-normalization issues on CI/Windows hosts.

---
Generated: 2025-09-14
