# Target-gradient campaign execution record

Launch: 2026-09-19 21:15:51.735753 UTC on `multigpu-vm`, GPU0, queue PID264170. Frozen remote checkout: `/home/rldoeswork/personal/tinycraft-jepa-iclr-20260919`. Interpreter: `/home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python`.

Authorization prerequisite: campaign monitor explicitly cleared GPU0 after all45 MLP and27 native-grid policies completed150 updates and their queues exited. Launch preflight independently checked all72 status entries had returncode0 and completed budgets; GPU0 had zero memory usage and no compute process. All9 source SHA256 values,4 input archive SHA256 values, and protocol SHA256 matched `research/target_gradient_launch_manifest.json` again immediately before launch. No training/probe/runner source was edited.

Exact launched command:

```sh
nohup /home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python -m studies.revision.run_target_gradient_queue --data-root /dev/shm/redcube --out runs/revision/target_gradient --gpu 0 --deadline-utc 2026-09-20T07:41:00+00:00
```

Queue log: `runs/revision/target_gradient_queue.log`; per-job logs under `runs/revision/target_gradient/logs/`. The runner applies the prospective timing-only admission decision; smoke outputs are not scientific endpoints. All7 checkpoints per admitted fit, full readout arrays and manifests will be retained.

Independent analysis source identities verified locally and remotely after launch:

- `analyze_target_gradient.py`: `828b62c806b0828d34d9ffa356a3dee332f8086e647e19d395abacfa65cef923`.
- `verify_target_gradient_execution.py`: `ddcd06f94074c3207ce6f18da256c3003277fb2150ad55eddeeceac7fd0808f2`.

Status at21:18:55 UTC: joint500-update timing fixture completed; stop-target500-update fixture running. Admission pending.

Admission at21:21 UTC: **admitted**, with all12 final and all12 midpoint readouts. Runtime-only projections: joint587.2s, stop-target588.4s per8500-update fit; readout54.459s percheckpoint. Conservative campaign13631.724s including50% margin, versus37188.162s remaining. No outcome-dependent change.
