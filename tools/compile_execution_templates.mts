import { readFileSync, mkdirSync, writeFileSync } from 'node:fs'
import { basename, join } from 'node:path'
import {
  asConnectionId,
  compile,
  importLitegraph,
  parseDinksterNodes,
  type JsonObject,
} from '../../Dinkster-Frontend/packages/core/src/index.ts'
import { targetsForArtifact } from '../../Dinkster-Frontend/packages/client/src/dinkster-connection.ts'

const [catalogPath, outputDirectory, ...templates] = process.argv.slice(2)
if (!catalogPath || !outputDirectory || templates.length === 0) {
  throw new Error('Usage: compile_execution_templates.mts CATALOG OUTPUT_DIRECTORY TEMPLATE...')
}
const catalog = parseDinksterNodes(JSON.parse(readFileSync(catalogPath, 'utf8')))
const errors = catalog.diagnostics.filter((diagnostic) => diagnostic.severity === 'error')
if (errors.length > 0) throw new Error(`Catalog errors: ${JSON.stringify(errors)}`)
mkdirSync(outputDirectory, { recursive: true })

for (const path of templates) {
  const workflow = JSON.parse(readFileSync(path, 'utf8')) as JsonObject
  const resolve = (type: string) => catalog.schemas.get(type)
  const imported = importLitegraph(workflow, resolve)
  if (imported.document === undefined || imported.diagnostics.some((diagnostic) => diagnostic.severity === 'error')) {
    throw new Error(`${basename(path)} import failed: ${JSON.stringify(imported.diagnostics)}`)
  }
  const compiled = compile({
    document: imported.document,
    revision: 1,
    resolve,
    scope: { kind: 'full' },
    connection: asConnectionId('template-smoke'),
    schemaHash: 'template-smoke',
    graphFeatures: ['regions', 'typedLiteral'],
  })
  if (!compiled.ok) throw new Error(`${basename(path)} compile failed: ${JSON.stringify(compiled.diagnostics)}`)
  const targets = targetsForArtifact(compiled.artifact, resolve)
  if (!compiled.artifact.dinksterGraph || targets.length === 0) {
    throw new Error(`${basename(path)} did not produce an executable native graph and targets`)
  }
  const result = {
    graph: compiled.artifact.dinksterGraph,
    targets,
    diagnostics: [...imported.diagnostics, ...compiled.artifact.diagnostics],
  }
  writeFileSync(join(outputDirectory, basename(path)), JSON.stringify(result, null, 2) + '\n')
  console.log(`${basename(path)} nativeNodes=${Object.keys(result.graph.nodes).length} targets=${result.targets.length}`)
}
