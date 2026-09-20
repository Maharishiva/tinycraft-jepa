# Independent last-only queue review

Verdict: **GREEN for the queue mechanics and registered six-fit scheduling**, after the process-group cleanup and prospective-pin checks were repaired. This verdict does not assert GPU runtime, successful model training, matched baseline outcomes, or completion of the eighteen-model audit.

Reviewed source: `studies/revision/run_last_only_queue.py`, SHA-256 `19ea6f5e5a22d9321443cb53dd11a6f51db7f1a818f8ac969deb0d893ed77913`.

The CPU validation command was `python3 research/last_only_validation/test_queue_cpu.py`. All **12 tests passed** in 2.529 seconds. The test harness mocks GPU discovery and all scientific children; it never calls a remote host, nvidia-smi, or a fitting implementation. One process-lifecycle test launches ordinary local Python children in an isolated process group and verifies that an orphaned, SIGTERM-ignoring descendant actually stops. The original failure was reproduced and retained in `queue_cleanup_initial.json`; the revised implementation passes that same case. Python emitted resource warnings from repeatedly invoking the CLI's `main` function in one test process without explicitly closing its advisory-lock handle; the production CLI runs once and the OS releases that handle on exit.

Verified behavior:

- Exactly six registered production commands: `gray_frame_still` and `gray_frame_wind`, model seeds 0, 1 and 2, mode `last_only`. Production commands expose no update-count, latent-size or objective-weight overrides. All final readouts use `model.npz` and require 4,000 MLP updates. The two runtime fixtures are separate from those six fits.
- Admission depends on runtime and the fixed deadline `2026-09-20T07:41:00+00:00`. The mocked successful campaign provides no MAE/result values at all, demonstrating that none is required by the scheduling path. A short budget executes the two timing fixtures and admits zero registered fits.
- Existing output paths, even empty directories, are rejected. Child failure stops immediately, preserves artifacts and the failed job record, and does not retry or replace a seed. A child deadline timeout records `deadline_incomplete`; an expired initial deadline launches no child.
- The launch must contain the exact four archive identities, exact training/runtime source set and the fixed protocol. Before any smoke, the common-window pin must be complete, have the two worlds, 8,192 windows, PCG64 seed 20260919, the exact sampler string and the exact three producer-source identities. Pin artifacts and protocol sources remain frozen with the campaign.
- Source changes are rejected after a child. Archive changes are rejected before launch and after each child, including an in-place modification that preserves inode, size and mtime. The queue rechecks full archive hashes at completion.
- Each child starts a new process group. Cleanup addresses the group even after its leader has exited, waits a bounded two seconds for SIGTERM, then sends SIGKILL and reaps the leader. The independent heartbeat test verifies the previously missed orphan-descendant case.

The parent campaign changed deployment from hardcoded e2e GPU 0 to an explicitly selected GPU 0 or 1 after transfer approval problems. This is a resource-routing change only. The queue checks that the selected GPU is idle before and after each child and takes an advisory lock shared by cooperating queue instances. Actual host assignment and GPU availability remain launcher responsibilities.

Baseline initialization/window/key/sample pairing is explicitly pending external verification; the queue no longer claims it has compared unavailable baseline fit manifests. Its completion scope is six last-only fits and their readouts. The common audit must enforce the eighteen-model provenance and pairing contract before reporting a matched comparison.
