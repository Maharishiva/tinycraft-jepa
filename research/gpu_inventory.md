# Live GPU and artifact inventory

Read-only VM audit on 2026-09-19. GPU availability snapshot: **19:43:16 UTC**. Later experiments may change availability. This audit launched no training, stopped no jobs, and changed no remote files. Existing result files were copied into separate local source directories.

## Availability

| Host | GPU | Device | Allocated memory | GPU utilization | Root-disk free | Shared-memory free |
|---|---:|---|---:|---:|---:|---:|
| `multigpu-vm` | 0 | NVIDIA A100 PCIe 40 GB | 0 MiB | 0% | 295 GiB | 5 GiB |
| `multigpu-vm` | 1 | NVIDIA A100 PCIe 40 GB | 0 MiB | 0% | 295 GiB | 5 GiB |
| `e2e` | 0 | NVIDIA A100 PCIe 40 GB | 4 MiB | 0% | 70 GiB | 51 GiB |

`nvidia-smi --query-compute-apps` returned no compute processes on either host. MG had idle background dispatch workers and a CPU red-cube web server; e2e had other users' development/assessor containers. No processes or containers were modified. MG `/dev/shm` is almost full; new outputs should use the disk-based revision directory.

SSH aliases resolve to `multigpu-vm.sas.yp-c.yandex.net` and `e2e.sas.yp-c.yandex.net`, user `rldoeswork`, port 22. Sandboxed SSH reported DNS failures; the explicitly authorized read-only checks succeeded with `require_escalated`.

## Execution environments

| Host | Python executable | Python version |
|---|---|---|
| MG | `/home/rldoeswork/personal/tinywmcraft-slim/.venv/bin/python` | 3.11.0rc1 |
| e2e | `/home/rldoeswork/personal/tinywmcraft-e2e/.venv/bin/python` | 3.11.15 |

Both environments: JAX/JAXLIB 0.10.1, Flax 0.12.7, Optax 0.2.8, NumPy 2.4.6, Craftax 1.5.0. Versions were read using `importlib.metadata`; the audit did not initialize a GPU JAX process.

Original remote source directories are `/home/rldoeswork/personal/tinycraft-jepa` on both hosts. They are file snapshots **without `.git`**, so no remote Git revision can be asserted. `train.py` SHA256 on both is `72cd5c463b4ac2fe7e289f7032e91d88ffac2e59ebb15534524fafe4804e9d0b`; `studies/stoch/env.py` is `bcceb9d5c660664ed2b0aab9ef30b9368b8a3db7a4fc9196bf5a5d6dc1d2bb69`. All audited MG `studies/stoch/*.py` match the initial local originals. e2e differs in `rl.py`, `heads_audit.py`, `latent_check.py`, `figures.py`, and `summarize.py`; copies are preserved in `gpu_inventory/original_e2e_source/`. The e2e `rl.py` predates later real-latent/decoded observation arms and amplification controls. New experiments should use the independently reviewed revision snapshot.

The parent assigned separate remote working roots `/home/rldoeswork/personal/tinycraft-jepa-iclr-20260919`: `experiment_design` owns MG staging, `craftax_audit` owns initial e2e staging and hands GPU0 to `pve_objective`. This audit did not stage those roots.

## Exact gate artifacts

MG data: `/home/rldoeswork/personal/stoch/gate/{train,eval}.npz`.
e2e data: `/dev/shm/stoch/gate/{train,eval}.npz`.

The training archive is 5,964,695,842 bytes, with 100 streams, 5,000 transitions per stream, seed 0, uniform-random policy, and 128-step episode cap. The evaluation archive is 299,370,654 bytes. Dataset metadata names `gate_off_still`, four actions, no image noise, no wind, and reward -1 for touching border or wall, otherwise zero.

**The two archive hashes differ, but all ten decompressed `.npy` members have identical SHA256 hashes.** This verifies exact equality of frames, actions, rewards, dones, position, wind, episode IDs, wall positions, gaps, and metadata. It resolves the apparent mismatch between seed 0 and seeds1/2 training provenance. The difference lies in archive container bytes; the exact differing ZIP field was not identified.

| Archive | SHA256 |
|---|---|
| MG gate train | `c347aa2db1726bceadbc6a16fc2b4a7559343f988c4c2074e90c914b05354b55` |
| e2e gate train | `ea1315a8c612afc7f5a6a80938d9ef982e3bd3af9ad04f6684ec9377df817cce` |
| Both `frames.npy` members | `e9704734dd833de512ae039d317399c606bd8b7e8833e08639f0716d7df407b3` |

Full per-array SHA256 manifests are `gpu_inventory/tinycraft-jepa-{mg,e2e}-gate-arrayhash.json`.

| Model | MG checkpoint, relative to `/home/rldoeswork/personal/stoch/gate` | SHA256 | Encoder/predictor train time |
|---|---|---|---:|
| Grid JEPA seed 0 | `tile_factored/model_probe.npz` | `f81ef20ab6d6f032a85310fe9233e32749924839eb8bb55e4c0e6b1db2b880d2` | 586.5 s |
| Grid JEPA seed 1 | `tile_factored_s1/model_probe.npz` | `7190a8c231596043b62b6750f807e1c9fa542287dc161b7ca72182f2e68e625a` | 576.3 s |
| Grid JEPA seed 2 | `tile_factored_s2/model_probe.npz` | `8ab666aa5838fb81b11af99c5f700f8dd33373f757ef6109a07116df70f9f716` | 580.9 s |
| Flow seed 0 | `flow/model.npz` | `72b829ea8cfb5f4bf21f3b8c0ce8ffe53f34dd5aa67bad001bb8d4cd0bab27e9` | 563.4 s |

The e2e flow alias `/dev/shm/stoch/gate/flow_model.npz` has the **same exact checkpoint SHA256** as MG. It is one model fit, not an independent replicate. e2e seed 0 JEPA lives at `/dev/shm/stoch/gate/tile_factored/model_probe.npz`, with matching metadata to MG seed 0. e2e also stores the 2x/4x training-budget variants and `lewm6`; these were inventoried, not selected as new study arms.

Grid JEPA uses 16×32=512 latent coordinates, 11,121,426 parameters, 8,500 training steps, batch 128, history 8, learning rate 0.001, SIGReg weight 0.09. Its decoder and reward/done/value probes are additional frozen-representation training; their costs are excluded from the model's `train_s` and must be reported separately. Seed0 decoder 95.8 s; heads 37.1 s. The flow model's 563.4 s includes tokenizer 167.3 s. These are existing runtimes, not guarantees for new jobs.

## Craftax

e2e data are `/home/rldoeswork/personal/tinywmcraft-e2e/starter/data/tinywmcraft_500k/train.npz` and `eval.npz` (approximately 1.1 GiB and 106 MiB). The original tinycraft-jepa `data/tinywmcraft_500k` folders are empty on both VMs; do not use those apparent default paths.

e2e checkpoint: `/dev/shm/stoch/craftax/tile_factored/model_probe.npz`.
SHA256: `59cc282c9bd7a49ab1c272342220bc8e0d8ca0f1aa77bbcb4c23ae31dfd6835d`.
Recorded training dataset SHA256: `09f2cb41a38b0f34937a89c4122f6b25bbb3094e34a077b3a572504269d0a1b2` (metadata identity; this audit did not independently rehash the Craftax dataset).

The grid model has 11,121,556 parameters, seed 0, 8,500 steps, 17 actions, and train 604.5 s; decoder 113.4 s plus heads 62.0 s. Existing measured result `runs/stoch/craftax/tile_factored_plan.json`: 128 environments, horizon 6, goal-reaching 94.53125% versus random 18.75%; 100-step reward MPC return 0.47344 versus random 1.44531. Held-out reward-head correlation 0.3527 and value-head correlation 0.2665. These establish a narrow goal-reaching result and reward-control failure for one fit, not general Craftax mastery. Further evaluation must use unchanged task rewards.

## Existing result provenance and local copies

Read-only `.json`/`.jsonl` copies:

- `/tmp/tinycraft-jepa-remote-mg/runs/`
- `/tmp/tinycraft-jepa-remote-e2e/runs/`

MG had 433 identical metadata files, 84 absent locally, and one differing `final.json`; e2e had 65 identical and 9 absent locally. The evidence-audit agent checked that remote JSONL ledgers match the local versions byte-for-byte. Most absent `final.json` files record the best evaluation, not the last fixed-budget endpoint, so these additions do not change endpoint inclusion. The differing MG `ppo_dream_latent_mean8_p1/final.json` is stale relative to its canonical 150-update ledger endpoint. `gray_episode_wind/baselines_edge.json` is a useful source-separated addition. No original local result was overwritten.

Metadata inventories and source/result hashes are preserved under `research/gpu_inventory/`. Some existing artifacts contain amplification or wall-specific controls prohibited by the current user instruction. They were inventoried only for provenance and are not authorized new experimental arms.

## Follow-up: gray-frame target-gradient experiment data

At 20:16 UTC, the two original red-cube datasets were reverified for the approved target-gradient comparison. This follow-up remained read-only and launched no fits.

| Dataset on MG | Full archive SHA256 |
|---|---|
| `/dev/shm/redcube/gray_frame_still/train.npz` | `fceb10cbcdd8710c29d687b34475f119889709a6f3eb94a2cf1d6079f27aa882` |
| `/dev/shm/redcube/gray_frame_still/eval.npz` | `397c96e130f662e52f8be537eac09b0a32ce014e10ba9454432ba34bedb25e3d` |
| `/dev/shm/redcube/gray_frame_wind/train.npz` | `9aa4b0736a83f27f3da77a4c5c79715fadf57253f41df5d27bf240e6ab3bc8ab` |
| `/dev/shm/redcube/gray_frame_wind/eval.npz` | `0e2670a70285e85d5b32ade26d8a4ee78cf7d740c36d38f3198c55f0694575dc` |

Each training dataset has 500,000 transitions (100 streams×5,000 steps), collection seed 0. Each evaluation dataset has 25,000 transitions (100×250), independent collection seed 1000. All have `meta.env="redcube"`, four actions, uniform-random collection, and 128-step episode caps. Both use gray image noise renewed per frame (`per_episode=false`); wind is false/true as named. Reward metadata is -1 on border contact, otherwise zero.

All eight members of each archive were individually SHA256-hashed, preserved in `gpu_inventory/tinycraft-jepa-mg-graydata-hashes.json`; metadata/shapes are in `tinycraft-jepa-mg-graydata.json`. No matching train/eval archives were found on e2e in `/dev/shm/redcube`, `/dev/shm/stoch`, `/dev/shm/stoch/redcube`, or `/home/rldoeswork/personal/stoch`. e2e's existing model probes are not copies of the datasets.

`studies/redcube/env.py` has SHA256 `468212e57824968bbabb5363c897ea30492ce7dd53099add309af40ffe5e7b4e` in both original remote trees, the original local checkout, and the isolated local revision. This verifies source/spec consistency for the original environment. Available capacity refreshed: MG disk 294.01 GiB, shared memory 4.91 GiB; e2e disk approximately 70 GiB, shared memory approximately 51 GiB.

## Read-only commands used

```sh
ssh -G multigpu-vm
ssh -G e2e
ssh HOST 'nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv'
ssh HOST 'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv'
ssh HOST 'df -h /home /tmp /dev/shm'
ssh HOST 'ps -u rldoeswork -o pid,ppid,etime,args'
ssh HOST 'docker ps --format "{{.ID}} {{.Image}} {{.Names}}"'
ssh HOST 'sha256sum CHECKPOINT_PATH'
```

Python scripts read checkpoint `meta` arrays with NumPy, package versions with `importlib.metadata`, archive members with `zipfile`, and stream SHA256 hashes with `hashlib`. Existing `.json` and `.jsonl` files were streamed through tar into isolated `/tmp` source directories.
