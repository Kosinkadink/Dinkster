Refs #

## Extension contract

- [ ] Behavior keyed on a node id, family id, widget type, or pack id goes through the same registry door available to packs.
- [ ] Scanner allowlist additions link the issue that removes the entry.
- [ ] If a scanned file was renamed or split, retarget or re-add its allowlist entries with their owning issues before `--write`; ownership does not carry across path changes.
- [ ] After merging main, run `uv run --locked python scripts/check_extension_factories.py --write`, review the diff, confirm no ceiling rose, and run `bash scripts/ci-fast.sh`.
- [ ] A new extension kind or capability has a runtime consumer, doctor check, core dogfooding test, and synthetic-pack coverage in this PR.
