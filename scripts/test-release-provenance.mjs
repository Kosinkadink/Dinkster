import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { test } from 'node:test'

const require = createRequire(import.meta.url)
const workflow = readFileSync(
  new URL('../.github/workflows/release.yml', import.meta.url),
  'utf8',
)
const scripts = [
  ...workflow.matchAll(/^ {10}script: \|\n((?: {12}[^\n]*\n|\n)+)/gm),
]
  .map((match) => match[1].replace(/^ {12}/gm, ''))
  .filter((script) => script.includes('createRef'))
assert.ok(scripts.length > 0, 'provenance scripts must be exercised')
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor
const commit = '1234567890abcdef1234567890abcdef12345678'

for (const [index, script] of scripts.entries()) {
  for (const scenario of [
    'created',
    'conflict',
    'existing',
    'missing',
    'wrong-target',
    'wrong-provenance',
    'forbidden',
    'create-forbidden',
  ]) {
    test(`provenance ${index}: ${scenario}`, async () => {
      const frontend = script.includes('frontend-manifest.json')
      const supervisor = script.includes('supervisor-main-')
      const version = '0.1.1.dev42'
      const tag = frontend
        ? `main-${version}`
        : supervisor
          ? `supervisor-main-${version}`
          : 'main-engine'
      const manifest = {
        version,
        tag,
        commit,
        sourceCommit: commit,
        sha256: 'wheel-digest',
      }
      const bytes = Buffer.from(JSON.stringify(manifest))
      const digest = createHash('sha256').update(bytes).digest('hex')
      const provenance = { ...manifest, manifest: digest }
      if (scenario === 'wrong-provenance') provenance.version = '0.1.1.dev41'
      let reads = 0
      let delays = 0
      const warnings = []
      const httpError = (status) =>
        Object.assign(new Error(`HTTP ${status}`), {
          status,
          response: { data: { message: 'Reference already exists' } },
        })
      const github = {
        rest: {
          repos: {
            getReleaseByTag: async () => {
              throw httpError(404)
            },
          },
          git: {
            getRef: async () => {
              reads++
              if (scenario === 'forbidden') throw httpError(403)
              if (
                scenario !== 'existing' &&
                (scenario === 'missing' || reads < 4)
              )
                throw httpError(404)
              return { data: { object: { type: 'tag', sha: 'annotation' } } }
            },
            createTag: async () => ({ data: { sha: 'annotation' } }),
            createRef: async () => {
              if (scenario === 'conflict') throw httpError(422)
              if (scenario === 'create-forbidden') throw httpError(403)
            },
            getTag: async () => ({
              data: {
                tag,
                message: JSON.stringify(provenance),
                object: {
                  type: 'commit',
                  sha:
                    scenario === 'wrong-target' ? 'different-commit' : commit,
                },
              },
            }),
          },
        },
      }
      const execute = new AsyncFunction(
        'require',
        'github',
        'context',
        'core',
        'setTimeout',
        'process',
        script,
      )
      const result = execute(
        (name) =>
          name === 'fs' ? { readFileSync: () => bytes } : require(name),
        github,
        {
          repo: { owner: 'test', repo: 'test' },
          sha: commit,
          ref: 'refs/heads/main',
        },
        { setOutput() {}, warning: (message) => warnings.push(message) },
        (resolve) => {
          delays++
          resolve()
        },
        { env: { DINKSTER_RELEASE_COMMIT: commit } },
      )
      if (['created', 'conflict', 'existing'].includes(scenario)) {
        await result
      } else if (
        ['missing', 'forbidden', 'create-forbidden'].includes(scenario)
      ) {
        await assert.rejects(result, {
          status: scenario === 'missing' ? 404 : 403,
        })
      } else {
        await assert.rejects(result, /differs|different/)
      }
      if (scenario === 'conflict')
        assert.ok(
          warnings.some(
            (message) =>
              message.includes('422') &&
              message.includes('Reference already exists'),
          ),
        )
      if (scenario === 'missing') {
        assert.equal(
          reads,
          5,
          'initial lookup plus four bounded readback attempts',
        )
        assert.equal(delays, 3)
      } else if (['created', 'conflict'].includes(scenario)) {
        assert.equal(reads, 4)
        assert.equal(delays, 2)
      } else if (
        ['existing', 'forbidden', 'create-forbidden'].includes(scenario)
      ) {
        assert.equal(delays, 0)
      }
    })
  }
}
