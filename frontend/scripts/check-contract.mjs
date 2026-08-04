import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(here, '..', '..')
const typesPath = path.join(root, 'frontend', 'src', 'api', 'types.ts')

const types = await readFile(typesPath, 'utf8')

// Every generated schema the client binds to, and the title it must still carry.
const bound = [
  { file: 'screening_result.schema.json', title: 'ScreeningResult' },
  { file: 'advice_document.schema.json', title: 'AdviceDocument' },
]

for (const { file, title } of bound) {
  const schema = JSON.parse(await readFile(path.join(root, 'contracts', file), 'utf8'))
  if (schema.title !== title) throw new Error(`Unexpected title in ${file}: ${schema.title}`)

  for (const [name, definition] of Object.entries(schema.$defs ?? {})) {
    if (!Array.isArray(definition.enum)) continue
    for (const value of definition.enum) {
      if (!types.includes(`'${String(value)}'`)) {
        throw new Error(`Frontend types are missing ${name} value ${String(value)} (${file})`)
      }
    }
  }
  // Fixed literals such as review_status and provider are the disclosure itself: the
  // frontend must pin them, not accept any string.
  for (const [field, definition] of Object.entries(schema.properties ?? {})) {
    if (typeof definition.const !== 'string') continue
    if (!types.includes(`'${definition.const}'`)) {
      throw new Error(`Frontend types are missing the fixed ${field} literal ${definition.const}`)
    }
  }
  for (const field of schema.required ?? []) {
    const fieldPattern = new RegExp(`\\b${field.replaceAll('_', '[_]')}\\??:`)
    if (!fieldPattern.test(types)) throw new Error(`Frontend ${title} is missing ${field}`)
  }
}

if (!types.includes("schema_version: '1.0.0'")) {
  throw new Error('Frontend schema version does not match contract 1.0.0')
}
console.log(`Frontend contract check passed against ${bound.length} generated schemas`)
