// TypeScript 5.5+ reports TS2882 for side-effect imports like `import "./globals.css"`
// when no declaration matches. Next's bundled types only cover `*.module.css`,
// so declare plain CSS here.
// Can be removed if a future Next.js version ships its own `*.css` declaration.
declare module "*.css"
