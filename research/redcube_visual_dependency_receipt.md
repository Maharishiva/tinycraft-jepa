# Posthoc flow-kernel dependency verification

**Pass.** The current local file, current VM file, and initial source snapshot all identify the same 23,477-byte `studies/stoch/flow_recipe/kernels.py`:

```text
7358061ebaae220616196e8905a912270247a517c5315034e40d538781a511ef
```

- Local: `/private/tmp/tinycraft-jepa-iclr-20260919/studies/stoch/flow_recipe/kernels.py`.
- VM: `/home/rldoeswork/personal/tinycraft-jepa-iclr-20260919/studies/stoch/flow_recipe/kernels.py` on `multigpu-vm`, read back at 2026-09-19T23:35:09.959793+00:00.
- Initial source snapshot: `research/source_snapshot.json`, timestamp 2026-09-19T19:43:21.964962+00:00, matching path, byte count, and SHA256 entry.

The frozen export calls `F.load(args.flow_code, args.flow, 4)`. Each original case manifest records `ftrain.__file__` in the VM recipe directory above. The manifest-hashed loader prepends that directory to `sys.path` and removes any cached `kernels` module before importing the recipe's `train.py`; that file imports `kernels`. A read-only Python import-spec lookup resolves the dependency to the same VM path. No module import or GPU initialization was used for this check.

This is a **separate posthoc verification of current dependency bytes**, supported by the earlier source snapshot. It is not export-launch pinning and does not retroactively supply an export-time runtime module hash. The original manifests still omit this dependency. No inference was rerun.

All original manifests, selections, packets, arrays, images, models, and source snapshots were preserved. The JSON companion records the four original manifest hashes and exact checks. Newly added `gate_s1` is outside the original four-packet audit.
