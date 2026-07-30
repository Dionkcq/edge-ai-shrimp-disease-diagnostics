import { readFile, readdir, stat } from 'node:fs/promises'
import { join, relative, resolve, sep } from 'node:path'

const root = resolve('dist')
const expectedBase = '/edge-ai-shrimp-disease-diagnostics/'
const errors = []

async function filesUnder(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const nested = await Promise.all(entries.map(async entry => {
    const path = join(directory, entry.name)
    return entry.isDirectory() ? filesUnder(path) : [path]
  }))
  return nested.flat()
}

try {
  if (!(await stat(root)).isDirectory()) {
    throw new Error('dist is not a directory')
  }
} catch {
  console.error('architecture/dist is missing; run npm run build first')
  process.exit(1)
}

const files = await filesUnder(root)
const relativeFiles = new Set(files.map(path => relative(root, path).split(sep).join('/')))
const textFiles = files.filter(path => /\.(?:html|css|js|json|map|svg|txt)$/u.test(path))
const prohibited = [
  '/tmp/shrimp_opus5_architecture_memo.md',
  'shrimp_opus5_architecture_memo',
  'datasets/raw/',
  'datasets/processed/',
  'mapping_acceptance.json',
  'artifacts/audit/',
  'ShrimpDiseaseImageBD_v3.zip',
  'TigerShrimpBD_v1.zip',
  'PRIVATE KEY',
]

for (const path of textFiles) {
  const body = await readFile(path, 'utf8')
  const display = relative(root, path)

  for (const token of prohibited) {
    if (body.includes(token)) {
      errors.push(`${display}: contains prohibited publication token ${JSON.stringify(token)}`)
    }
  }

  if (/\.onnx(?:\b|[?#"'])/iu.test(body)) {
    errors.push(`${display}: contains a model artifact filename`)
  }

  if (!path.endsWith('.js') && /\.(?:jpe?g|parquet|npz|zip)(?:\b|[?#"'])/iu.test(body)) {
    errors.push(`${display}: contains a possible data/archive filename`)
  }

  if (/\.(?:html|css)$/u.test(path)) {
    for (const match of body.matchAll(/(?:src|href)=["']([^"']+)["']/giu)) {
      const target = match[1]
      if (/^(?:https?:)?\/\//iu.test(target)) {
        errors.push(`${display}: external runtime reference ${target}`)
        continue
      }
      if (target.startsWith('#') || target.startsWith('data:') || target.startsWith('blob:')) {
        continue
      }
      const withoutQuery = target.split(/[?#]/u, 1)[0]
      if (withoutQuery.startsWith('/')) {
        if (!withoutQuery.startsWith(expectedBase)) {
          errors.push(`${display}: absolute reference is outside expected Pages base: ${target}`)
          continue
        }
        const artifactPath = withoutQuery.slice(expectedBase.length)
        if (artifactPath && !relativeFiles.has(artifactPath)) {
          errors.push(`${display}: referenced artifact is missing: ${target}`)
        }
      }
    }

    for (const match of body.matchAll(/url\(["']?([^)'"\s]+)["']?\)/giu)) {
      const target = match[1]
      if (/^(?:https?:)?\/\//iu.test(target)) {
        errors.push(`${display}: external CSS runtime reference ${target}`)
      }
    }
  }
}

const index = await readFile(join(root, 'index.html'), 'utf8')
if (!index.includes(expectedBase)) {
  errors.push(`index.html: expected base path ${expectedBase} was not found`)
}

if (errors.length > 0) {
  console.error(`Architecture site check failed with ${errors.length} issue(s):`)
  for (const error of errors) console.error(`- ${error}`)
  process.exit(1)
}

console.log(`Architecture site check passed: ${files.length} files, base ${expectedBase}, no broken local HTML/CSS asset links, external runtime references, or prohibited publication tokens.`)
