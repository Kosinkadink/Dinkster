import { readFileSync } from 'node:fs'
import { basename } from 'node:path'
import {
  asConnectionId,
  compile,
  importLitegraph,
  parseDinksterNodes,
  type InterfaceItem,
  type JsonObject,
  type NodeSchema,
} from '../../Dinkster-Frontend/packages/core/src/index.ts'

const templates = process.argv.slice(3)
const catalog = parseDinksterNodes(JSON.parse(readFileSync(process.argv[2]!, 'utf8')))

function stubSchema(node: JsonObject): NodeSchema {
  const items: InterfaceItem[] = []
  for (const input of node.inputs ?? []) {
    if (typeof input !== 'object' || input === null) continue
    const item = input as JsonObject
    if (typeof item.name !== 'string') continue
    items.push({ kind: 'input', id: item.name, type: { kind: 'wildcard' }, optional: true })
  }
  for (const output of node.outputs ?? []) {
    if (typeof output !== 'object' || output === null) continue
    const item = output as JsonObject
    if (typeof item.name !== 'string') continue
    items.push({ kind: 'output', id: item.name, type: { kind: 'wildcard' } })
  }
  return {
    type: String(node.type),
    displayName: String(node.type),
    category: 'template smoke stub',
    source: 'v1',
    isOutputNode: false,
    items,
  }
}

for (const path of templates) {
  const workflow = JSON.parse(readFileSync(path, 'utf8')) as JsonObject
  const stubs = new Map<string, NodeSchema>()
  for (const graph of [workflow, ...(workflow.definitions as JsonObject | undefined)?.subgraphs ?? []]) {
    for (const node of (graph as JsonObject).nodes ?? []) {
      if (typeof node !== 'object' || node === null) continue
      const typed = node as JsonObject
      if (typeof typed.type !== 'string' || catalog.schemas.has(typed.type)) continue
      stubs.set(typed.type, stubSchema(typed))
    }
  }
  const resolveSchema = (type: string) => catalog.schemas.get(type) ?? stubs.get(type)
  const imported = importLitegraph(workflow, resolveSchema)
  if (imported.document === undefined) {
    throw new Error(`${basename(path)} import failed: ${JSON.stringify(imported.diagnostics)}`)
  }
  const compiled = compile({
    document: imported.document,
    revision: 1,
    resolve: resolveSchema,
    scope: { kind: 'full' },
    connection: asConnectionId('template-smoke'),
    schemaHash: 'template-smoke',
    graphFeatures: ['regions', 'typedLiteral'],
  })
  if (!compiled.ok) throw new Error(`${basename(path)} compile failed: ${JSON.stringify(compiled.diagnostics)}`)
  console.log(`${basename(path)} stubs=${stubs.size} nodes=${Object.keys(compiled.artifact.dinksterGraph!.nodes).length}`)
}
