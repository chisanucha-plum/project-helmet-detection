---
name: frontend
description: "Use this skill whenever working with frontend code, React, Next.js, TypeScript, or when the user mentions: frontend review, refactor UI, check React components, improve frontend architecture, optimize performance, accessibility issues, type safety, state management, or any frontend-related task. Trigger even when users say 'review this component', 'optimize this page', 'check types', or 'improve accessibility'."
---

# Frontend Architecture & Review

Senior Frontend Engineer skill for reviewing, refactoring, and improving React/Next.js codebases with focus on maintainability, scalability, performance, type safety, accessibility, clean architecture, and developer experience.

## Critical rules

- Never suggest `as any` - always use proper TypeScript types
- Always analyze for accessibility (WCAG 2.2) compliance
- Prefer composition over inheritance
- Prefer explicit code over clever code
- Do not recommend abstractions that increase complexity unnecessarily
- Think like a senior engineer reviewing production code
- Provide actionable recommendations with code examples

## Workflow

### 1. Primary Objectives

1. Improve code readability
2. Reduce technical debt
3. Enforce consistent architecture
4. Detect performance bottlenecks
5. Identify dead code and unnecessary complexity
6. Improve accessibility (A11y)
7. Increase type safety
8. Ensure code is scalable for large teams

### 2. Tech Stack Knowledge

Expert knowledge areas:
- **Frameworks**: React, Next.js (App Router & Pages Router), TypeScript, JavaScript ES2023
- **State Management**: React Context, Zustand, Redux Toolkit, TanStack Query, SWR
- **Styling**: TailwindCSS, CSS Modules, Styled Components, SCSS
- **APIs**: REST API, GraphQL, SSE, WebSocket
- **Performance**: React Memoization, Bundle Optimization, Code Splitting, Lazy Loading, Suspense, Server Components
- **Accessibility**: WCAG 2.2, ARIA, Keyboard Navigation, Screen Readers

### 3. Review Categories

When reviewing code, ALWAYS analyze these 12 categories:

**3.1. Architecture**

Detect: Components that do too much, business logic inside UI, tight coupling, poor folder structure, duplicate logic, missing abstraction layers

Recommend: Custom Hooks, Shared Components, Service Layer, Repository Pattern, Feature-based architecture

**3.2. Component Design**

Detect: Large components (>300 lines), deep prop drilling, duplicate JSX, multiple responsibilities

Recommend: Extract Components, Compound Components, Custom Hooks, Context Provider

**3.3. State Management**

Detect: Derived state stored in useState, state duplication, unnecessary re-renders, state synchronization issues

**3.4. API Layer**

Detect: Raw fetch calls inside components, duplicate API URLs, missing error handling, missing loading states, missing retry logic

Recommend: Separate services/, repositories/, api/ layers

**3.5. TypeScript**

Detect: `any`, unknown misuse, duplicated interfaces, type assertions, inconsistent types

Never allow: `as any`

Prefer: interface, type, generics, discriminated unions

**3.6. Dead Code**

Detect: Unused files, unused functions, unused imports, commented code, deprecated exports, empty files

Classify: Safe to delete vs. Verify before deleting

**3.7. Performance**

Analyze: Re-render issues (missing useMemo/useCallback), expensive computations (filter/map/reduce), rendering issues (large lists, missing virtualization), network issues (duplicate requests, refetch loops), bundle size (heavy dependencies, large client components)

**3.8. Accessibility (A11y)**

Detect: Missing aria-label, missing alt text, missing keyboard support, missing focus states, color-only indicators, missing semantic HTML

Follow: WCAG 2.2, Screen reader compatibility

**3.9. React Best Practices**

Check: useEffect misuse, infinite loops, incorrect dependencies, side effects in render, state mutation, inline object creation

**3.10. Next.js Best Practices**

Check: Server Components vs Client Components, data fetching strategy, Suspense, dynamic imports, route organization, metadata usage, image optimization

**3.11. Security**

Detect: dangerouslySetInnerHTML, XSS risks, exposed environment variables, sensitive data in client bundles, insecure localStorage usage

**3.12. Folder Structure Review**

Recommend: Feature-based or layered architecture (app/, components/, hooks/, services/, repositories/, schemas/, types/, utils/, providers/, features/, lib/)

### 4. Severity Levels

Classify every issue:
- 🔴 **High**: Bugs, Security, Architecture problems, Memory leaks
- 🟠 **Medium**: Performance, Type safety, Maintainability
- 🟡 **Low**: Style, Naming, Minor optimizations

### 5. Required Output Format

Always produce:

**Executive Summary** - Short overview

**Findings** - For every issue:
- Category
- Severity (🔴/🟠/🟡)
- Location (file:line)
- Problem
- Why it matters
- Recommendation
- Example Fix

**Refactoring Plan** - Step-by-step implementation order

**Summary Table** - Issue count by category and severity

### 6. Review Principles

1. Prefer simplicity
2. Prefer composition over inheritance
3. Prefer explicit code over clever code
4. Prefer type safety
5. Avoid premature optimization
6. Reduce cognitive load
7. Design for long-term maintainability
8. Optimize for team scalability

## Bundled resources

None - This skill contains all necessary guidelines inline.
