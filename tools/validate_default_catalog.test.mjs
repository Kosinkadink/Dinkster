import assert from 'node:assert/strict'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'

test('catalog validation needs only the native decoder and rejects errors or lost schemas', async () => {
  const root = await mkdtemp(join(tmpdir(), 'native-catalog-'))
  try {
    const schemaRoot = join(root, 'packages/core/src/schema')
    await mkdir(schemaRoot, { recursive: true })
    await writeFile(join(root, 'package.json'), '{"type":"module"}')
    await writeFile(join(schemaRoot, 'dinkster-wire.ts'), `
export function parseDinksterNodes(payload) {
  return {
    schemas: new Map(Object.keys(payload.nodes).filter((key) => key !== payload.drop).map((key) => [key === payload.rename ? 'native.unpublished' : key, {}])),
    diagnostics: payload.diagnostics ?? [],
  }
}
`)
    const catalog = join(root, 'catalog.json')
    const run = async (payload) => {
      await writeFile(catalog, JSON.stringify(payload))
      return spawnSync(process.execPath, [fileURLToPath(new URL('./validate_default_catalog.mjs', import.meta.url)), catalog, root], { encoding: 'utf8' })
    }
    const payload = { nodes: { 'native.first': {}, 'native.second': {} } }
    const valid = await run(payload)
    assert.equal(valid.status, 0, valid.stdout + valid.stderr)
    assert.match(valid.stdout, /Decoded 2 schemas/)
    const dropped = await run({ ...payload, drop: 'native.second' })
    assert.notEqual(dropped.status, 0)
    assert.match(dropped.stderr, /native.second.*published schema was lost/)
    const renamed = await run({ ...payload, rename: 'native.second' })
    assert.notEqual(renamed.status, 0)
    assert.match(renamed.stderr, /native.second.*published schema was lost/)
    const empty = await run({ nodes: {} })
    assert.notEqual(empty.status, 0)
    assert.match(empty.stderr, /default catalog must exercise real published schemas/)
    const invalid = await run({ ...payload, diagnostics: [{ severity: 'error', message: 'invalid native schema' }] })
    assert.notEqual(invalid.status, 0)
    assert.match(invalid.stderr, /invalid native schema/)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
