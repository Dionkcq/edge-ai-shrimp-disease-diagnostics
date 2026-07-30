import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(here, '..', '..')
const schemaPath = path.join(root, 'contracts', 'screening_result.schema.json')
const typesPath = path.join(root, 'frontend', 'src', 'api', 'types.ts')

const schema = JSON.parse(await readFile(schemaPath, 'utf8'))
const types = await readFile(typesPath, 'utf8')

if (schema.title !== 'ScreeningResult') throw new Error('Unexpected screening schema title')
for (const [name, definition] of Object.entries(schema.$defs ?? {})) {
  if (!Array.isArray(definition.enum)) continue
  for (const value of definition.enum) {
    const quoted = `'${String(value)}'`
    if (!types.includes(quoted)) {
      throw new Error(`Frontend types are missing ${name} value ${String(value)}`)
    }
  }
}
for (const field of schema.required ?? []) {
  const fieldPattern = new RegExp(`\\b${field.replaceAll('_', '[_]')}\\??:`)
  if (!fieldPattern.test(types)) throw new Error(`Frontend ScreeningResult is missing ${field}`)
}
if (!types.includes("schema_version: '1.0.0'")) {
  throw new Error('Frontend schema version does not match contract 1.0.0')
}
console.log('Frontend contract check passed against contracts/screening_result.schema.json')
