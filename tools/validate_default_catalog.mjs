import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const [catalogPath, frontendRoot] = process.argv.slice(2)
assert(catalogPath && frontendRoot, 'Usage: validate_default_catalog.mjs CATALOG_JSON FRONTEND_ROOT')
const { parseDinksterNodes } = await import(pathToFileURL(resolve(
  frontendRoot, 'packages/core/src/schema/dinkster-wire.ts',
)).href)
const payload = JSON.parse(await readFile(catalogPath, 'utf8'))
const decoded = parseDinksterNodes(payload)
const diagnostics = decoded.diagnostics
for (const diagnostic of diagnostics) console.log(`${diagnostic.severity}: ${diagnostic.message}`)
assert.equal(diagnostics.filter((item) => item.severity === 'error').length, 0,
  diagnostics.filter((item) => item.severity === 'error').map((item) => item.message).join('\n'))
const declared = Object.keys(payload.nodes)
assert(declared.length > 0, 'default catalog must exercise real published schemas')
for (const type of declared) {
  assert(decoded.schemas.has(type), `${type}: published schema was lost`)
}
assert.equal(decoded.schemas.size, declared.length, 'decoded schema count differs from published catalog')
console.log(`Decoded ${decoded.schemas.size} schemas`)
