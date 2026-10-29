import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const [catalogPath, frontendRoot] = process.argv.slice(2)
assert(catalogPath && frontendRoot, 'Usage: validate_default_catalog.mjs CATALOG_JSON FRONTEND_ROOT')
const schemaModule = (name) => import(pathToFileURL(resolve(
  frontendRoot, 'packages/core/src/schema', `${name}.ts`,
)).href)
const { parseDinksterNodes } = await schemaModule('dinkster-wire')
const { comfyAliasCatalogFromDinksterWire } = await schemaModule('comfy-alias')
const { comfyGroupCatalogFromDinksterWire } = await schemaModule('comfy-group')
const payload = JSON.parse(await readFile(catalogPath, 'utf8'))
const decoded = parseDinksterNodes(payload)
const aliases = comfyAliasCatalogFromDinksterWire(payload, decoded.schemas)
const groups = comfyGroupCatalogFromDinksterWire(payload, decoded.schemas)
const diagnostics = [...decoded.diagnostics, ...aliases.diagnostics, ...groups.diagnostics]
for (const diagnostic of diagnostics) console.log(`${diagnostic.severity}: ${diagnostic.message}`)
assert.equal(diagnostics.filter((item) => item.severity === 'error').length, 0,
  diagnostics.filter((item) => item.severity === 'error').map((item) => item.message).join('\n'))
for (const [key, result] of [['comfyAliases', aliases], ['comfyGroups', groups]]) {
  const declared = Object.values(payload.packs).flatMap((pack) => pack[key]?.records ?? [])
  assert.equal(result.catalog.records.length, declared.length, `${key}: published records were lost`)
  for (const pack of ['dinkster-nodes-generation', 'dinkster-nodes-image']) {
    const records = payload.packs[pack][key].records
    assert(records.length > 0, `${pack} ${key} must exercise real maintained records`)
    assert.equal(result.catalog.records.filter((record) => record.ownerPack === pack).length,
      records.length, `${pack} ${key}: published records were lost`)
  }
}
console.log(`Decoded ${decoded.schemas.size} schemas, ${aliases.catalog.records.length} aliases, ${groups.catalog.records.length} groups`)
