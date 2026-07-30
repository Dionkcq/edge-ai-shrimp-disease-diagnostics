import { readdir, readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const dist = path.resolve(here, '..', 'dist')
const textExtensions = new Set(['.html', '.css', '.js', '.json', '.svg', '.webmanifest'])
const files = []

async function walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name)
    if (entry.isDirectory()) await walk(full)
    else files.push(full)
  }
}

await walk(dist)
if (!files.some((file) => path.basename(file) === 'index.html')) {
  throw new Error('frontend/dist/index.html is missing; run npm run build first')
}

const allowedEmbeddedUrls = ['https://react.dev/errors/', 'http://www.w3.org/']
const violations = []
for (const file of files) {
  if (!textExtensions.has(path.extname(file))) continue
  const text = await readFile(file, 'utf8')
  const urls = text.match(/https?:\/\/[^"'`\s)]+/gi) ?? []
  if (urls.some((url) => !allowedEmbeddedUrls.some((prefix) => url.startsWith(prefix)))) {
    violations.push(path.relative(dist, file))
  }
  if (/(?:src|href)=["']\/\//i.test(text) || /@import\s+(?:url\()?\s*["']?\/\//i.test(text)) {
    violations.push(path.relative(dist, file))
  }
}
if (violations.length) {
  throw new Error(`Remote runtime references found in: ${[...new Set(violations)].join(', ')}`)
}
console.log(`Offline asset check passed: ${files.length} built files, no remote runtime references`)
