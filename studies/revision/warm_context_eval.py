"""Paired native-environment evaluation from seven-action real prefixes.

One saved packet is shared by every model family and every training seed. The
first 256 candidates that survive seven uniformly random actions are admitted;
no policy or post-prefix outcome participates in admission. All simulator and
action-sampling keys are stored in the packet, including its fresh texture keys.
The original 128-step episode cap leaves 121 controlled transitions.

Materialize on CPU, without loading a model::

    JAX_PLATFORMS=cpu python -m studies.revision.warm_context_eval \
        --config gray_frame_wind --out /path/shared/gray_frame_wind.npz

Adapters provide ``posterior(B,T,H,W,C)`` and
``posterior_step(previous_real, current_real)`` returning decoded uint8 RGB.
Policies are the original ``studies.stoch.rl.AC(obs="pixels")``. This module
never supplies privileged simulator state to the policy or the adapter.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from studies.stoch import env as S


SEED = 20260919
EPISODES = 256
CONTEXT = 8
PREFIX_STEPS = CONTEXT - 1
CONTROL_STEPS = S.MAX_STEPS - PREFIX_STEPS
WORLDS = ("gray_frame_still", "gray_frame_wind", "gray_episode_wind", "gate")
FORMAT = "native_warm_context_v1"


def _keys(base, domain, ids):
    root = jax.random.fold_in(base, domain)
    return jax.vmap(lambda i: jax.random.fold_in(root, i))(jnp.asarray(ids, jnp.uint32))


def _time_keys(base, domain, ids, steps):
    roots = _keys(base, domain, ids)
    return jax.vmap(lambda k: jax.vmap(lambda t: jax.random.fold_in(k, t))(
        jnp.arange(steps, dtype=jnp.uint32)))(roots)


def _choose(active, new, old):
    return jnp.where(active.reshape((active.shape[0],) + (1,) * (new.ndim - 1)), new, old)


@dataclass
class WarmStartPacket:
    raw_context: np.ndarray
    incoming_actions: np.ndarray
    state: S.State
    prefix_ids: np.ndarray
    reset_keys: np.ndarray
    prefix_action_keys: np.ndarray
    prefix_transition_keys: np.ndarray
    control_action_keys: np.ndarray
    control_transition_keys: np.ndarray
    rejected_prefix_ids: np.ndarray
    metadata: dict[str, Any]

    @property
    def episodes(self):
        return len(self.prefix_ids)

    def arrays(self):
        names = ("raw_context", "incoming_actions", "prefix_ids", "reset_keys",
                 "prefix_action_keys", "prefix_transition_keys", "control_action_keys",
                 "control_transition_keys", "rejected_prefix_ids")
        arrays = {name: np.asarray(getattr(self, name)) for name in names}
        arrays.update({f"state_{name}": np.asarray(value)
                       for name, value in zip(S.State._fields, self.state)})
        return arrays

    def fingerprint(self):
        """Array-level digest, independent of ZIP timestamps and host paths."""
        digest = hashlib.sha256()
        digest.update(json.dumps(self.metadata, sort_keys=True).encode())
        for name, value in sorted(self.arrays().items()):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def validate(self):
        n = self.episodes
        if self.metadata.get("format") != FORMAT:
            raise ValueError("unrecognized warm-context packet format")
        for name, expected in (("episodes", n), ("prefix_steps", PREFIX_STEPS),
                               ("native_max_steps", S.MAX_STEPS), ("control_steps", CONTROL_STEPS)):
            if self.metadata.get(name) != expected:
                raise ValueError(f"packet metadata {name} differs from native protocol")
        if (self.metadata.get("config") not in WORLDS
                or self.metadata.get("spec") != asdict(S.CONFIGS[self.metadata["config"]])):
            raise ValueError("packet config/spec mismatch")
        if self.raw_context.shape != (n, CONTEXT, S.H, S.W, 3) or self.raw_context.dtype != np.uint8:
            raise ValueError("context must contain eight native uint8 RGB frames")
        if self.incoming_actions.shape != (n, CONTEXT) or np.any(self.incoming_actions[:, 0] != 0):
            raise ValueError("incoming actions must be [0, a0, ..., a6]")
        if np.any((self.incoming_actions < 0) | (self.incoming_actions >= S.N_ACTIONS)):
            raise ValueError("invalid prefix action")
        if np.asarray(self.state.step).shape != (n,) or np.any(self.state.step != PREFIX_STEPS):
            raise ValueError("packet state must be at native step seven")
        if n < 1 or np.any(np.diff(self.prefix_ids) <= 0):
            raise ValueError("prefix IDs must be nonempty and in admission order")
        attempts = int(self.metadata["total_attempts"])
        rejected = np.asarray(self.rejected_prefix_ids)
        if (attempts != int(self.prefix_ids[-1]) + 1 or attempts != n + len(rejected)
                or len(rejected) != self.metadata["excluded_prefix_deaths"]
                or not np.array_equal(np.sort(np.concatenate([self.prefix_ids, rejected])),
                                      np.arange(attempts))):
            raise ValueError("candidate accounting must end at the final admitted prefix")
        for name, shape in (("reset_keys", (n, 2)),
                            ("prefix_action_keys", (n, PREFIX_STEPS, 2)),
                            ("prefix_transition_keys", (n, PREFIX_STEPS, 2)),
                            ("control_action_keys", (n, CONTROL_STEPS, 2)),
                            ("control_transition_keys", (n, CONTROL_STEPS, 2))):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype != np.uint32:
                raise ValueError(f"invalid {name} shape/dtype")
        return self


def create_packet(config, episodes=EPISODES, seed=SEED, candidate_batch=64):
    """Host rejection loop; admission inspects only the seven prefix dones.

    Batch size cannot affect keys, admitted IDs, or the candidate accounting.
    ``episodes`` is configurable for correctness tests; scientific evaluation
    enforces 256 in ``make_evaluator`` by default.
    """
    if config not in WORLDS:
        raise ValueError(f"config must be one of {WORLDS}")
    if episodes < 1 or candidate_batch < 1:
        raise ValueError("episodes and candidate_batch must be positive")
    spec = S.CONFIGS[config]
    reset, step = S.make(spec)
    base = jax.random.PRNGKey(seed)

    @jax.jit
    def prefixes(reset_keys, action_keys, transition_keys):
        frame, state = jax.vmap(reset)(reset_keys)

        def advance(carry, keys):
            old_state, old_frame, alive = carry
            ka, ks = keys
            action = jax.vmap(lambda k: jax.random.randint(k, (), 0, S.N_ACTIONS))(ka)
            new_frame, new_state, _, done = jax.vmap(step)(old_state, action, ks)
            new_state = jax.tree.map(lambda new, old: _choose(alive, new, old), new_state, old_state)
            new_frame = _choose(alive, new_frame, old_frame)
            return (new_state, new_frame, alive & ~done), (new_frame, action)

        (state, _, alive), (tail, actions) = jax.lax.scan(
            advance, (state, frame, jnp.ones((len(reset_keys),), bool)),
            (jnp.swapaxes(action_keys, 0, 1), jnp.swapaxes(transition_keys, 0, 1)))
        frames = jnp.concatenate([frame[:, None], jnp.swapaxes(tail, 0, 1)], axis=1)
        incoming = jnp.concatenate([jnp.zeros((len(reset_keys), 1), jnp.int32),
                                    jnp.swapaxes(actions, 0, 1)], axis=1)
        return frames, incoming, state, alive

    pieces, rejected, cursor, admitted = [], [], 0, 0
    while admitted < episodes:
        ids = np.arange(cursor, cursor + candidate_batch, dtype=np.int32)
        rk = _keys(base, 11, ids)
        ak = _time_keys(base, 21, ids, PREFIX_STEPS)
        sk = _time_keys(base, 41, ids, PREFIX_STEPS)
        frames, incoming, state, alive = jax.device_get(prefixes(rk, ak, sk))
        keep = np.flatnonzero(alive)[:episodes - admitted]
        examined = int(keep[-1]) + 1 if len(keep) == episodes - admitted else len(ids)
        rejected.extend(ids[:examined][~alive[:examined]].tolist())
        if len(keep):
            pieces.append((ids[keep], frames[keep], incoming[keep],
                           jax.tree.map(lambda a: a[keep], state)))
            admitted += len(keep)
        cursor += candidate_batch
    ids = np.concatenate([p[0] for p in pieces])
    state = jax.tree.map(lambda *a: np.concatenate(a), *[p[3] for p in pieces])
    metadata = {
        "format": FORMAT, "config": config, "spec": asdict(spec), "seed": int(seed),
        "episodes": int(episodes), "total_attempts": int(ids[-1]) + 1,
        "excluded_prefix_deaths": len(rejected), "prefix_steps": PREFIX_STEPS,
        "native_max_steps": S.MAX_STEPS, "control_steps": CONTROL_STEPS,
        "admission": "first candidates surviving seven uniform random actions; no future filtering",
        "reset_distribution": "original S.make(spec), uniform; fresh simulator texture key retained",
        "evaluation_rng": "all stored keys shared across families and training policy seeds",
        "key_domains": {"reset": 11, "prefix_action": 21, "prefix_transition": 41,
                        "control_action": 61, "control_transition": 81},
    }
    return WarmStartPacket(
        np.concatenate([p[1] for p in pieces]), np.concatenate([p[2] for p in pieces]), state, ids,
        np.asarray(_keys(base, 11, ids)), np.asarray(_time_keys(base, 21, ids, PREFIX_STEPS)),
        np.asarray(_time_keys(base, 41, ids, PREFIX_STEPS)),
        np.asarray(_time_keys(base, 61, ids, CONTROL_STEPS)),
        np.asarray(_time_keys(base, 81, ids, CONTROL_STEPS)),
        np.asarray(rejected, np.int32), metadata).validate()


def save_packet(packet, path):
    packet.validate()
    path = Path(path)
    if path.exists():
        existing = load_packet(path)
        if existing.fingerprint() != packet.fingerprint():
            raise FileExistsError("refusing to overwrite a different paired evaluation packet")
        return existing.fingerprint()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=json.dumps(packet.metadata, sort_keys=True),
                            fingerprint=packet.fingerprint(), **packet.arrays())
    temporary.replace(path)
    return packet.fingerprint()


def load_packet(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        names = ("raw_context", "incoming_actions", "prefix_ids", "reset_keys",
                 "prefix_action_keys", "prefix_transition_keys", "control_action_keys",
                 "control_transition_keys", "rejected_prefix_ids")
        packet = WarmStartPacket(**{name: np.array(data[name]) for name in names},
                                 state=S.State(*(np.array(data[f"state_{f}"]) for f in S.State._fields)),
                                 metadata=metadata).validate()
        if packet.fingerprint() != str(data["fingerprint"]):
            raise ValueError("warm-context packet fingerprint mismatch")
    return packet


def load_or_create_packet(path, config, episodes=EPISODES, seed=SEED):
    if Path(path).exists():
        packet = load_packet(path)
        for name, expected in (("config", config), ("episodes", episodes), ("seed", seed)):
            if packet.metadata[name] != expected:
                raise ValueError(f"existing packet {name} differs from requested protocol")
        return packet
    packet = create_packet(config, episodes, seed)
    save_packet(packet, path)
    return packet


def burn_context(ac, adapter, raw_context, incoming_actions, hidden_size=128):
    """Consume decoded frames 0..6 once; frame seven is the next policy input."""
    decoded = adapter.posterior(raw_context)
    if decoded.shape != raw_context.shape or decoded.dtype != jnp.uint8:
        raise ValueError("adapter.posterior must return native-shaped uint8 decoded RGB")
    g = jnp.zeros((raw_context.shape[0], hidden_size), jnp.float32)

    def burn(g, inputs):
        frame, previous_action = inputs
        return ac.step(frame, previous_action, g)[2], None

    g, _ = jax.lax.scan(burn, g, (jnp.swapaxes(decoded[:, :-1], 0, 1),
                                 jnp.swapaxes(incoming_actions[:, :-1], 0, 1)))
    return g, decoded[:, -1], incoming_actions[:, -1]


def make_evaluator(spec, adapter, packet, *, require_episodes=EPISODES, hidden_size=128):
    """Return compiled ``eval_ac(ac) -> arrays`` using the packet's fixed keys.

    ``adapter`` is an NNX module or a stateless object whose decoder methods can
    be traced. The returned function accepts the stock pixel actor-critic.
    There is deliberately no evaluation seed argument: training seeds do not
    change the shared held-out episode cohort or the policy-sampling keys.
    """
    import flax.nnx as nnx

    packet.validate()
    if require_episodes is not None and packet.episodes != require_episodes:
        raise ValueError(f"scientific evaluation requires {require_episodes} episodes")
    if asdict(spec) != packet.metadata["spec"]:
        raise ValueError("evaluation world differs from packet world")
    n, nt = packet.episodes, min(3, packet.episodes)
    raw = jnp.asarray(packet.raw_context)
    incoming = jnp.asarray(packet.incoming_actions)
    states = jax.tree.map(jnp.asarray, packet.state)
    action_keys = jnp.swapaxes(jnp.asarray(packet.control_action_keys), 0, 1)
    transition_keys = jnp.swapaxes(jnp.asarray(packet.control_transition_keys), 0, 1)
    vstep = jax.vmap(S.make(spec)[1])

    def run(ac, current_adapter):
        g, decoded, pa = burn_context(ac, current_adapter, raw, incoming, hidden_size)
        trace_rgb = jnp.zeros((nt, CONTROL_STEPS + 1, S.H, S.W, 3), jnp.uint8)
        trace_rgb = trace_rgb.at[:, 0].set(raw[:nt, -1])
        trace_actions = jnp.full((nt, CONTROL_STEPS), -1, jnp.int32)
        trace_rewards = jnp.zeros((nt, CONTROL_STEPS), jnp.float32)
        trace_active = jnp.zeros((nt, CONTROL_STEPS), bool)
        trace_dones = jnp.zeros((nt, CONTROL_STEPS), bool)

        def advance(carry, inputs):
            state, real, decoded, g, pa, done, ret, length, walls, rgb, acts, rewards, active_log, dones = carry
            t, ka, ks = inputs
            logits, _, ng = ac.step(decoded, pa, g)
            action = jax.vmap(lambda k, p: jax.random.categorical(k, p))(ka, logits).astype(jnp.int32)
            frame, nxt, reward, terminal = vstep(state, action, ks)
            active = ~done
            passed = active & (state.wall_x != S.NO_WALL) & (nxt.wall_x == S.NO_WALL)
            nxt = jax.tree.map(lambda new, old: _choose(active, new, old), nxt, state)
            frame = _choose(active, frame, real)
            # Flow must see the actual preceding RGB observation. Passing the
            # decoded frame here, or duplicating the current frame, is wrong.
            next_decoded = current_adapter.posterior_step(real, frame)
            next_decoded = _choose(active, next_decoded, decoded)
            ng, next_pa = _choose(active, ng, g), jnp.where(active, action, pa)
            reward = jnp.where(active, reward, 0.0)
            done = done | terminal
            rgb = rgb.at[:, t + 1].set(frame[:nt])
            acts = acts.at[:, t].set(jnp.where(active[:nt], action[:nt], -1))
            rewards = rewards.at[:, t].set(reward[:nt])
            active_log = active_log.at[:, t].set(active[:nt])
            dones = dones.at[:, t].set(done[:nt])
            return (nxt, frame, next_decoded, ng, next_pa, done, ret + reward,
                    length + active.astype(jnp.int32), walls + passed.astype(jnp.int32),
                    rgb, acts, rewards, active_log, dones), None

        initial = (states, raw[:, -1], decoded, g, pa, jnp.zeros(n, bool),
                   jnp.zeros(n, jnp.float32), jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32),
                   trace_rgb, trace_actions, trace_rewards, trace_active, trace_dones)
        final, _ = jax.lax.scan(advance, initial, (jnp.arange(CONTROL_STEPS), action_keys, transition_keys))
        state, _, _, g, pa, done, ret, length, walls, rgb, acts, rewards, active, dones = final
        return {
            "return": ret, "survival": ret > -0.5, "control_length": length,
            "total_length": length + PREFIX_STEPS, "walls_passed": walls, "done": done,
            "final_state": state, "final_gru": g, "final_previous_action": pa,
            "trace_control_rgb": rgb, "trace_control_actions": acts,
            "trace_control_rewards": rewards, "trace_control_active": active,
            "trace_control_dones": dones,
        }

    # NNX adapters carry their frozen modules explicitly through the graph.
    # Stateless test adapters may be safely closed over, without model arrays.
    if isinstance(adapter, nnx.Module):
        compiled = nnx.jit(run)
        return lambda ac: compiled(ac, adapter)
    return nnx.jit(lambda ac: run(ac, adapter))


def summarize(result):
    ret = np.asarray(result["return"])
    total = np.asarray(result["total_length"])
    control = np.asarray(result["control_length"])
    alive = np.asarray(result["survival"], bool)
    return {
        "return": float(ret.mean()), "survival": float(alive.mean()),
        "survival_sem": float(alive.std() / np.sqrt(len(alive))),
        "length": float(total.mean()), "total_length": float(total.mean()),
        "control_length": float(control.mean()),
        "deaths_per_100": float(100. * (~alive).sum() / total.sum()),
        "control_deaths_per_100": float(100. * (~alive).sum() / control.sum()),
        "walls_passed": float(np.asarray(result["walls_passed"]).mean()), "n_ep": len(ret),
    }


def save_evaluation(result, packet, out, *, policy_seed=None, provenance=None):
    """Write every outcome and fixed traces 0,1,2; never select successful runs."""
    result = jax.device_get(result)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ntrace = min(3, packet.episodes)
    trace_names = [name for name in result if name.startswith("trace_")]
    traces = {name.removeprefix("trace_"): result[name] for name in trace_names}
    traces.update({
        "episode_indices": np.arange(ntrace), "prefix_ids": packet.prefix_ids[:ntrace],
        "prefix_rgb": packet.raw_context[:ntrace],
        "prefix_actions": packet.incoming_actions[:ntrace, 1:],
        "native_rgb": np.concatenate([packet.raw_context[:ntrace, :-1],
                                       traces["control_rgb"]], axis=1),
        "native_actions": np.concatenate([packet.incoming_actions[:ntrace, 1:],
                                           traces["control_actions"]], axis=1),
        "native_active": np.concatenate([np.ones((ntrace, PREFIX_STEPS), bool),
                                          traces["control_active"]], axis=1),
    })
    np.savez_compressed(out / "traces.npz", **traces)
    outcomes = {name: value for name, value in result.items()
                if name not in trace_names and name != "final_state"}
    outcomes.update({f"final_state_{name}": value
                     for name, value in zip(S.State._fields, result["final_state"])})
    outcomes.update({"episode_indices": np.arange(packet.episodes), "prefix_ids": packet.prefix_ids,
                     "reset_keys": packet.reset_keys, "prefix_action_keys": packet.prefix_action_keys,
                     "prefix_transition_keys": packet.prefix_transition_keys,
                     "control_action_keys": packet.control_action_keys,
                     "control_transition_keys": packet.control_transition_keys})
    np.savez_compressed(out / "outcomes.npz", **outcomes)
    summary = summarize(result)
    summary.update({"packet_fingerprint": packet.fingerprint(), "protocol": packet.metadata,
                    "training_policy_seed": policy_seed, "provenance": provenance or {},
                    "trace_episode_indices": list(range(ntrace)),
                    "trace_selection": "first three admitted episodes, irrespective of outcome",
                    "score_scope": "conditional on survival of the seven random prefix actions",
                    "length_definition": "total_length includes prefix; control_length excludes it"})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def evaluate(ac, adapter, packet, out=None, *, policy_seed=None, provenance=None):
    """Convenience wrapper; reuse ``make_evaluator`` across repeated checkpoints."""
    result = make_evaluator(S.CONFIGS[packet.metadata["config"]], adapter, packet)(ac)
    if out is None:
        return result
    return save_evaluation(result, packet, out, policy_seed=policy_seed, provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=WORLDS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    packet = load_or_create_packet(args.out, args.config)
    print(json.dumps({"path": str(args.out), "fingerprint": packet.fingerprint(),
                      **packet.metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
