"""Small CPU-only provenance and reset-bank helpers for the inventory study."""
import argparse
import hashlib
import json
from pathlib import Path

DATA_SHA256 = "09f2cb41a38b0f34937a89c4122f6b25bbb3094e34a077b3a572504269d0a1b2"
FLOW_SHA256 = "b4d7b82ba698f1b7aeaab7383c324523bf27bde1aba2fe817745fce38efa0aef"
JEPA_SOURCE_SHA256 = "59cc282c9bd7a49ab1c272342220bc8e0d8ca0f1aa77bbcb4c23ae31dfd6835d"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    temp.replace(path)


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    runtime_names = ('inventory_common.py', 'inventory_flow.py', 'inventory_jepa.py',
                     'inventory_jepa_probe.py', 'inventory_ppo.py', 'inventory_eval.py',
                     'inventory_numerical_review.py')
    paths = [*(Path(__file__).with_name(name) for name in runtime_names),
             Path(__file__).with_name("run_inventory_queue.py"),
             *Path(__file__).with_name("inventory_source").glob("*.py"),
             Path(__file__).with_name("inventory_source") / "SOURCE_MANIFEST.json",
             root / "research/craftax_inventory_protocol.md",
             root / "research/craftax_native_inventory_protocol.md",
             root / "train.py", root / "probe.py", root / "model.py"]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths) if p.exists()}


def validate_jepa_bundle(path):
    """CPU-only verification of the fixed readout recipe and unchanged base arrays."""
    import numpy as np
    path = Path(path)
    manifest_path = path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    expected = {'seed': 0, 'steps': 8000, 'batch': 256, 'lr': .001, 'hidden': 256,
                'depth': 6, 'head_steps': 6000, 'head_batch': 1024, 'head_hidden': 512, 'head_lr': .001}
    if manifest['status'] != 'complete' or manifest['format'] != 'frozen_jepa_inventory_v1':
        raise ValueError('JEPA readout fitting is incomplete or has the wrong format')
    if not (manifest.get('frozen_array_verification') is True
            and manifest.get('saved_bundle_roundtrip_verified') is True):
        raise ValueError('JEPA fitting has not completed its frozen-array and reload verification')
    if manifest['source_sha256'] != {'model': JEPA_SOURCE_SHA256, 'data': DATA_SHA256}:
        raise ValueError('JEPA source/data identity differs from the registered recipe')
    if any(manifest['arguments'].get(k) != v for k, v in expected.items()):
        raise ValueError('JEPA readout configuration differs from the predeclared fixed recipe')
    if not (manifest['seed'] == 0 and manifest['data_count'] == 500000
            and manifest['latent_transform'] == 'none' and manifest['model_gradients'] is False
            and manifest['value_head'] is False):
        raise ValueError('JEPA fitting contract differs from the fixed protocol')
    if manifest['artifacts_sha256'][path.name] != sha256(path):
        raise ValueError('JEPA fitted artifact does not match its fitting manifest')
    current_sources = source_hashes()
    if any(current_sources.get(k) != v for k, v in manifest['runtime_source_sha256'].items()):
        raise ValueError('JEPA fitting runtime sources differ from the current reviewed sources')
    original = Path(manifest['arguments']['model'])
    if sha256(original) != JEPA_SOURCE_SHA256:
        raise ValueError('Original JEPA source bundle is not the registered model')
    with np.load(original, allow_pickle=False) as source, np.load(path, allow_pickle=False) as fitted:
        original_meta, fitted_meta = json.loads(str(source['meta'])), json.loads(str(fitted['meta']))
        if original_meta['cfg']['seed'] != 0 or fitted_meta['cfg'] != original_meta['cfg']:
            raise ValueError('Frozen JEPA model configuration changed')
        decoder, heads = fitted_meta['decoder'], fitted_meta['inventory_heads']
        if any(decoder.get(k) != v for k, v in {'hidden': 256, 'depth': 6, 'tokens': 16,
                'steps': 8000, 'batch': 256, 'lr': .001, 'frames': 500000}.items()):
            raise ValueError('Stored decoder metadata differs from the registered fitting recipe')
        if any(heads.get(k) != v for k, v in {'hidden': 512, 'steps': 6000, 'batch': 1024,
                'lr': .001, 'transitions': 500000}.items()):
            raise ValueError('Stored reward/done metadata differs from the registered fitting recipe')
        names = sorted(k for k in source.files if k.startswith(('model.p.', 'model.b.')))
        if not names or names != sorted(k for k in fitted.files if k.startswith(('model.p.', 'model.b.'))):
            raise ValueError('Frozen JEPA model arrays differ')
        for key in names:
            left, right = source[key], fitted[key]
            if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
                raise ValueError(f'Frozen JEPA model array changed: {key}')
    return {'manifest_sha256': sha256(manifest_path), 'original_sha256': JEPA_SOURCE_SHA256,
            'frozen_model_arrays_compared': len(names), 'frozen_model_arrays_unchanged': True,
            'config': expected}


def prepare_bank(data_path, output):
    """Every eligible genuine eight-frame prefix, ordered by native episode start.

The seven transitions must not cross a reset. Eligibility uses only archive
length and done flags; never reward, inventory, achievement or pixel content.
"""
    import numpy as np
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data_hash = sha256(data_path)
    if data_hash != DATA_SHA256:
        raise ValueError(f"Training archive mismatch: {data_hash}")
    with np.load(data_path, allow_pickle=False) as archive:
        done = np.asarray(archive["dones"])
        t, e = np.nonzero(np.concatenate([np.ones((1, done.shape[1]), bool), done[:-1] > .5], 0))
        candidate_count = len(t)
        within_archive = t + 7 <= done.shape[0]
        t, e = t[within_archive], e[within_archive]
        no_reset = ~np.any(done[t[:, None] + np.arange(7), e[:, None]] > .5, axis=1)
        t, e = t[no_reset], e[no_reset]
        contexts = np.asarray(archive["frames"])[t[:, None] + np.arange(8), e[:, None]]
        actions = np.asarray(archive['actions'])[t[:, None] + np.arange(7), e[:, None]].astype(np.int32)
        # Privileged metadata, when present, is descriptive only; never consumed by the policy or WM.
        descriptive = {key: np.asarray(archive[key])[t + 7, e] for key in ('inventory', 'achievements')
                       if key in archive.files}
        prefix_rewards = np.asarray(archive['rewards'])[t[:, None] + np.arange(7), e[:, None]].sum(axis=1)
    if contexts.dtype != np.uint8 or contexts.shape[1:] != (8, 63, 63, 3):
        raise ValueError(f"Wrong reset contexts {contexts.dtype} {contexts.shape}")
    if not len(t) or actions.shape != (len(t), 7) or not np.all((actions >= 0) & (actions < 17)):
        raise ValueError('Invalid genuine-prefix bank')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez(handle, contexts=contexts, actions=actions, episode_start_time_index=t,
                 actor_start_time_index=t + 7, env_index=e,
                 descriptive_prefix_native_return=prefix_rewards,
                 bank_format=np.asarray('genuine_episode_prefix8_v1'),
                 training_data_sha256=np.asarray(data_hash),
                 **{'descriptive_initial_' + key: value for key, value in descriptive.items()})
    write_json(output.with_suffix(".json"), {"archive": str(data_path), "archive_sha256": data_hash,
               "bank_sha256": sha256(output), "count": len(t), "shape": list(contexts.shape),
               'bank_format': 'genuine_episode_prefix8_v1', 'candidate_episode_starts': candidate_count,
               'excluded_incomplete': int((~within_archive).sum()), 'excluded_reset_crossing': int((~no_reset).sum()),
               "order": "np.nonzero([ones(1,E); dones[:-1] > .5]), stable eligibility filtering",
               "eligibility": "Eight recorded frames exist and seven intervening done flags are false",
               "context": "Genuine frames t..t+7 and outgoing actions t..t+6; actor sees frame t+7",
               'descriptive_metadata_fields': list(descriptive), 'outcome_filtering': False,
               'initial_inventory_distribution': 'Recorded directly' if 'inventory' in descriptive else 'Not present in archive',
               'descriptive_prefix_return_mean': float(prefix_rewards.mean()),
               'actor_reset_previous_action': 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare_bank(args.data, args.out)
