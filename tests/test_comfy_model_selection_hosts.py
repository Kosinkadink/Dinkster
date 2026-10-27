"""Native selection loaders cross ordinary hosts with controlled inference seams."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from dinkster_assets import MountDef, MountTable
from dinkster_workers import detect_bubblewrap

PROBE = """
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from dinkster_api.v1 import Node, NodeSchema, InputSpec, OutputSpec, TypeExpr
from dinkster_nodes_comfy import model_selections

class VAE:
    def __init__(self, **kwargs):
        self.arguments = kwargs
        self.patcher = SimpleNamespace()
    def throw_exception_if_invalid(self):
        self.validated = True

def load(path, *, safe_load, return_metadata=False):
    assert safe_load
    data = {'weight': Path(path).read_text()}
    return (data, {'format': 'test'}) if return_metadata else data

def diffusers(path, *, output_vae, output_clip, embedding_directory):
    assert output_vae and output_clip
    vae = VAE(sd={'tree': (Path(path) / 'weights.bin').read_text()}, metadata=None)
    vae.throw_exception_if_invalid()
    return SimpleNamespace(), SimpleNamespace(), vae

def register_types(registry):
    from dinkster_api.v1 import register_inference_types
    register_inference_types(registry)
    model_selections.initialize_inference_process = lambda: None
    sys.modules['torch'] = SimpleNamespace(tensor=lambda value: value)
    sys.modules['dinkster_inference.utils'] = SimpleNamespace(load_torch_file=load)
    sys.modules['dinkster_inference.sd'] = SimpleNamespace(VAE=VAE, load_vae_patcher=object())
    sys.modules['dinkster_inference.diffusers_load'] = SimpleNamespace(load_diffusers=diffusers)

class Inspect(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(node_type='selectionprobe.inspect', inputs=(
            InputSpec('vae', TypeExpr.concrete('VAE')),
        ), outputs=(OutputSpec('summary', TypeExpr.concrete('STRING')),))
    @classmethod
    def execute(cls, *, vae):
        assert vae.validated
        return cls.outputs(summary=json.dumps({**vae.arguments,
            'reload': hasattr(vae.patcher, 'cached_patcher_init')}))

NODES = (Inspect,)
"""

SCENARIO = """
import asyncio
import json
import sys
from pathlib import Path
from dinkster.compose import PackSpec, compose_serving
from dinkster_engine import ExecutionError
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, validate
from dinkster_server.preflight import bind_model_names
from dinkster_workers import SandboxPolicy

async def main():
    mode, root, native, probe = sys.argv[1:]
    root, native, probe = Path(root), Path(native), Path(probe)
    isolated = mode != 'in-process'
    manifests = (native, probe)
    specs = [PackSpec(path, in_process=not isolated, trust_reserved=True,
        **({'worker_group':'selection-host', 'group_manifests':manifests} if isolated else {}))
        for path in manifests]
    policy = (SandboxPolicy(ro_binds=(str(native.parents[1]), str(root)))
        if mode == 'sandbox' else None)
    composition = await compose_serving(specs, include_default_packs=False,
        worker_env={'DINKSTER_MOUNTS_SNAPSHOT':str(root / 'snapshot.json')}, sandbox_policy=policy)
    try:
        engine = composition.make_engine(lambda event: None)
        async def run(node_type, input_id, value):
            graph = Graph({'load':GraphNode(node_type, {input_id:value}),
                'inspect':GraphNode('selectionprobe.inspect', {'vae':Link('load','VAE')})})
            bound, missing = bind_model_names(graph, composition.schemas, None)
            assert not missing, missing
            result = await engine.run(bound, ['inspect'])
            assert not result.diagnostics, result.diagnostics
            return json.loads(result.outputs['inspect']['summary'].resolve())
        assert await run('VAELoader','vae_name','pixel_space') == {
            'sd':{'pixel_space_vae':1.0},'metadata':None,'reload':False}
        assert await run('VAELoader','vae_name','real.safetensors') == {
            'sd':{'weight':'file'},'metadata':{'format':'test'},'reload':True}
        assert await run('VAELoader','vae_name','taesd') == {
            'sd':{'taesd_encoder.weight':'enc','taesd_decoder.weight':'dec',
                'vae_scale':0.18215,'vae_shift':0.0},'metadata':None,'reload':False}
        assert await run('DiffusersLoader','model_path','model') == {
            'sd':{'tree':'tree'},'metadata':None,'reload':False}
        assert await run('VAELoader','vae_name',TypedLiteral('COMBO','pixel_space')) == {
            'sd':{'pixel_space_vae':1.0},'metadata':None,'reload':False}
        invalid = Graph({'load':GraphNode('DiffusersLoader', {'model_path':{'digest':'bad'}})})
        assert [d.code for d in validate(invalid,composition.schemas,['load'])] == [
            'unstamped-asset-widget-literal']
        (root / 'model-vae-approx/taesd_encoder.safetensors').write_text('changed')
        changed = await run('VAELoader','vae_name','taesd')
        assert changed['sd']['taesd_encoder.weight'] == 'changed'
        (root / 'model-diffusers/model/weights.bin').write_text('new tree')
        assert (await run('DiffusersLoader','model_path','model'))['sd']['tree'] == 'new tree'
        bound, missing = bind_model_names(Graph({
            'load':GraphNode('VAELoader', {'vae_name':'taesd'})}),composition.schemas,None)
        assert not missing
        snapshot = root / 'snapshot.json'
        grants = json.loads(snapshot.read_text())
        grants['mounts'] = [row for row in grants['mounts'] if row['kind'] != 'model/vae-approx']
        snapshot.write_text(json.dumps(grants))
        graph = Graph({'load':GraphNode('VAELoader', {'vae_name':'taesd'})})
        _, missing = bind_model_names(graph, composition.schemas, None)
        assert len(missing) == 1
        try:
            await engine.run(bound,['load'])
        except ExecutionError as error:
            assert 'not granted' in str(error)
        else:
            raise AssertionError('bound source name survived grant revocation')
        print(json.dumps({'host':mode,'literal':True,'file':True,'pair':True,
            'directory':True,'revocation':True}))
    finally:
        await composition.close()

asyncio.run(main())
"""


@pytest.mark.parametrize("mode", ["in-process", "isolated", "sandbox"])
def test_native_selection_loaders_in_public_hosts(tmp_path: Path, mode: str) -> None:
    if mode == "sandbox":
        capability = detect_bubblewrap()
        if not capability.available:
            pytest.skip(capability.detail)
    root = Path(__file__).resolve().parents[1]
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    probe = fixture / "dinkster-pack.toml"
    probe.write_text(
        '[pack]\nname="selection-probe"\npublisher="dinkster"\n'
        'namespaces=["selectionprobe"]\n[pack.entry]\n'
        'nodes="selection_probe:NODES"\ntypes="selection_probe:register_types"\n'
    )
    (fixture / "selection_probe.py").write_text(PROBE)
    table = MountTable(tmp_path / "snapshot.json")
    for kind, files in (
        ("model/vae", {"real.safetensors": "file"}),
        (
            "model/vae-approx",
            {"taesd_encoder.safetensors": "enc", "taesd_decoder.safetensors": "dec"},
        ),
        ("model/diffusers", {"model/model_index.json": "{}", "model/weights.bin": "tree"}),
    ):
        path = tmp_path / kind.replace("/", "-")
        for relative, text in files.items():
            member = path / relative
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_text(text)
        table.add(MountDef(kind.replace("/", "-"), path, kind=kind))
        table.scan(kind.replace("/", "-"))
    native = root / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
    environment = {
        **os.environ,
        "DINKSTER_MOUNTS_SNAPSHOT": str(tmp_path / "snapshot.json"),
        "PYTHONPATH": os.pathsep.join((str(fixture), os.environ.get("PYTHONPATH", ""))),
    }
    result = subprocess.run(
        [sys.executable, "-c", SCENARIO, mode, str(tmp_path), str(native), str(probe)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"revocation": true' in result.stdout
