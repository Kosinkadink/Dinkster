"""Build folded wheels from checkouts and self-contained source archives."""

import json
import tempfile
import tomllib
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        if self.target_name == "sdist":
            text = Path(self.root, "pyproject.toml").read_text(encoding="utf-8")
            config = tomllib.loads(text)
            sources = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
            included = {str(Path(__file__)): "hatch_build.py"}
            for source, destination in sources.items():
                bundled = f"_sources/{destination.replace('/', '-')}"
                included[source] = bundled
                text = text.replace(f"{json.dumps(source)} =", f"{json.dumps(bundled)} =")
            text = text.replace(json.dumps(self.config["path"]), '"hatch_build.py"')
            # Only the archive gets local build paths; checkout builds keep their live peers.
            self._staged_config = tempfile.TemporaryDirectory(prefix="dinkster-sdist-")
            project = Path(self._staged_config.name, "pyproject.toml")
            project.write_text(text, encoding="utf-8")
            included[str(project)] = "pyproject.toml"
            build_data["force_include"] = included
            return
        if version != "editable":
            return
        directories = self.build_config.dev_mode_dirs
        directories.extend(str(Path(self.root, path).parent) for path in self.build_config.packages)
        for source, destination in list(self.build_config.force_include.items()):
            if len(Path(destination).parts) == 1:
                directories.append(str(Path(source).parent))
                del self.build_config.force_include[source]

    def finalize(self, version: str, build_data: dict[str, object], artifact_path: str) -> None:
        if self.target_name == "sdist":
            self._staged_config.cleanup()
