"""Keep folded wheel modules live in editable workspace installations."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        if version != "editable":
            return
        directories = self.build_config.dev_mode_dirs
        directories.extend(str(Path(self.root, path).parent) for path in self.build_config.packages)
        for source, destination in list(self.build_config.force_include.items()):
            if len(Path(destination).parts) == 1:
                directories.append(str(Path(source).parent))
                del self.build_config.force_include[source]
