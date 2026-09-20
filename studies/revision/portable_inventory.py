"""Verify an anonymous bundle and plan a fresh inventory rerun, without execution.

Use --check for JSON on stdout only, or --write-plan NEW_DIRECTORY for an
exclusive attestation/plan copy. --run-root must name a separate nonexistent
directory. All commands run sequentially with cwd set to the verified bundle.
Deferred argv elements are typed SHA-256 dependencies, not shell placeholders:
after the named producer completes, hash the specified file and substitute its
digest as one argument. This program never imports or runs scientific code.

The plan describes new measurements, not replay of historical manifests or any
reported result. Original source-to-bundle hashes remain in the attestation;
fresh fits record the bundled sources through the unchanged frozen validators.
The plan contains supplied local paths and is a private execution record.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


REGISTERED_INPUTS = {
    "data": "09f2cb41a38b0f34937a89c4122f6b25bbb3094e34a077b3a572504269d0a1b2",
    "flow_model": "b4d7b82ba698f1b7aeaab7383c324523bf27bde1aba2fe817745fce38efa0aef",
    "jepa_model": "59cc282c9bd7a49ab1c272342220bc8e0d8ca0f1aa77bbcb4c23ae31dfd6835d",
}
CONSTANT_NAMES = {"data": "DATA_SHA256", "flow_model": "FLOW_SHA256",
                  "jepa_model": "JEPA_SOURCE_SHA256"}
REQUIRED_SOURCES = (
    "train.py", "probe.py", "pyproject.toml",
    "studies/revision/portable_inventory.py", "studies/revision/inventory_common.py",
    "studies/revision/inventory_jepa.py", "studies/revision/inventory_jepa_probe.py",
    "studies/revision/inventory_flow.py", "studies/revision/inventory_ppo.py",
    "studies/revision/inventory_eval.py",
    "studies/revision/inventory_source/agent.py",
    "studies/revision/inventory_source/kernels.py",
    "studies/revision/inventory_source/tinywm_module.py",
    "studies/revision/inventory_source/wm_env2.py",
    "studies/revision/inventory_source/wm_generic.py",
)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PlanError(ValueError):
    pass


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PlanError("Duplicate JSON object key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def relative_file(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise PlanError("Manifest paths must be nonempty POSIX-relative strings")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != value:
        raise PlanError("Noncanonical or escaping manifest path: " + value)
    if any(part in ("", ".") for part in relative.parts):
        raise PlanError("Invalid manifest path: " + value)
    path = root.joinpath(*relative.parts)
    for parent in [path, *path.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise PlanError("Symlinks are not accepted in a verified bundle: " + value)
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise PlanError("Missing or escaped manifest file: " + value)
    return path


def literal_constants(path, names):
    """Read literal constants without importing or executing the source."""
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    if target.id in values or not isinstance(node.value, ast.Constant):
                        raise PlanError("Expected a unique literal constant: " + target.id)
                    values[target.id] = node.value.value
    if set(values) != set(names):
        raise PlanError("Missing required literal source constants")
    return values


def verify_bundle(root):
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise PlanError("Bundle root is not a directory")
    manifest_path = relative_file(root, "manifest.json")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or not isinstance(manifest.get("records"), list):
        raise PlanError("Expected anonymous bundle manifest schema 1 with records")
    if not manifest["records"]:
        raise PlanError("Empty bundle manifest")
    verified = {}
    for record in manifest["records"]:
        if not isinstance(record, dict):
            raise PlanError("Invalid manifest record")
        name, expected = record.get("path"), record.get("bundle_sha256")
        path = relative_file(root, name)
        if name in verified:
            raise PlanError("Duplicate manifest path: " + name)
        if not isinstance(expected, str) or not HEX_SHA256.fullmatch(expected):
            raise PlanError("Invalid bundled SHA-256: " + name)
        actual = sha256(path)
        if actual != expected:
            raise PlanError("Bundled file hash differs: " + name)
        if "bundle_bytes" in record and record["bundle_bytes"] != path.stat().st_size:
            raise PlanError("Bundled file size differs: " + name)
        source_hash = record.get("source_sha256")
        if not isinstance(source_hash, str) or not HEX_SHA256.fullmatch(source_hash):
            raise PlanError("Missing original source SHA-256: " + name)
        verified[name] = {"source_sha256": source_hash, "bundle_sha256": actual,
                          "bundle_bytes": path.stat().st_size}
    absent = sorted(set(REQUIRED_SOURCES) - verified.keys())
    if absent:
        raise PlanError("Required runtime sources are unmanifested: " + ", ".join(absent))
    # Dynamic imports and source_hashes() glob this directory. An unlisted Python
    # file could change both execution and provenance despite valid listed hashes.
    actual_python = {str(p.relative_to(root)) for p in root.rglob("*.py")}
    extras = sorted(actual_python - verified.keys())
    if extras:
        raise PlanError("Unmanifested Python sources: " + ", ".join(extras))
    for name in ("research/craftax_inventory_protocol.md",
                 "research/craftax_native_inventory_protocol.md",
                 "studies/revision/inventory_source/SOURCE_MANIFEST.json"):
        if (root / name).exists() and name not in verified:
            raise PlanError("Unmanifested provenance input: " + name)
    generated = manifest.get("generated_files", {})
    if not isinstance(generated, dict):
        raise PlanError("generated_files must map paths to SHA-256 digests")
    readme_hash = manifest.get("generated_readme_sha256")
    if readme_hash is not None:
        if "README.md" in generated and generated["README.md"] != readme_hash:
            raise PlanError("Conflicting generated README hashes")
        generated = {**generated, "README.md": readme_hash}
    for name, expected in generated.items():
        path = relative_file(root, name)
        if not isinstance(expected, str) or not HEX_SHA256.fullmatch(expected) or sha256(path) != expected:
            raise PlanError("Generated artifact hash differs: " + name)
        if name in verified and verified[name]["bundle_sha256"] != expected:
            raise PlanError("Conflicting generated artifact identity: " + name)
    constants = literal_constants(root / "studies/revision/inventory_common.py",
                                  set(CONSTANT_NAMES.values()))
    if any(constants[CONSTANT_NAMES[key]] != value for key, value in REGISTERED_INPUTS.items()):
        raise PlanError("Bundled input identity constants differ from the registered original inputs")
    actor = literal_constants(root / "studies/revision/inventory_eval.py", {"AGENT_SHA256"})
    if actor["AGENT_SHA256"] != verified["studies/revision/inventory_source/agent.py"]["bundle_sha256"]:
        raise PlanError("Bundled actor bytes cannot pass the unchanged evaluator identity check")
    return root, {"manifest_sha256": sha256(manifest_path), "record_count": len(verified),
                  "all_recorded_file_hashes_verified": True, "files": verified,
                  "generated_files_verified": generated,
                  "authenticity_scope": "Integrity against the supplied manifest, not an external authenticity certificate"}


def new_destination(path, label):
    path = Path(path).expanduser().absolute()
    if ".." in path.parts:
        raise PlanError(label + " must not contain parent traversal")
    if os.path.lexists(path):
        raise PlanError(label + " must not exist")
    # Existing parent symlinks would silently redirect eventual command outputs.
    for parent in path.parents:
        if parent.is_symlink():
            raise PlanError(label + " must not have symlink parents")
    resolved = path.resolve(strict=False)
    if os.path.lexists(resolved):
        raise PlanError(label + " must not resolve to an existing path")
    return resolved


def make_plan(bundle_root, data, flow_model, jepa_model, run_root, python_executable=None):
    root, bundle = verify_bundle(bundle_root)
    run_root = new_destination(run_root, "Run root")
    if run_root.is_relative_to(root) or root.is_relative_to(run_root):
        raise PlanError("Run root must be separate from the verified bundle")
    inputs = {}
    for key, value in (("data", data), ("flow_model", flow_model), ("jepa_model", jepa_model)):
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise PlanError("Input is not a regular file: " + key)
        actual = sha256(path)
        if actual != REGISTERED_INPUTS[key]:
            raise PlanError("Input SHA-256 differs from the registered original: " + key)
        inputs[key] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    interpreter = str(Path(python_executable or sys.executable).expanduser().resolve(strict=True))
    if not Path(interpreter).is_file() or not os.access(interpreter, os.X_OK):
        raise PlanError("Python executable must be an existing executable file")
    bank, fitted = run_root / "reset_bank.npz", run_root / "jepa_readouts.npz"
    command = lambda module, args: [interpreter, "-m", "studies.revision." + module, *args]
    def stage(name, argv, depends_on=(), outputs=()):
        return {"id": name, "cwd": str(root), "depends_on": list(depends_on), "argv": argv,
                "outputs": [str(p) for p in outputs], "execution": "not_executed"}
    stages = [stage("reset_bank", command("inventory_common", ["--data", inputs["data"]["path"],
                    "--out", str(bank)]), outputs=(bank, bank.with_suffix(".json"))),
              stage("jepa_readouts", command("inventory_jepa_probe", ["--model", inputs["jepa_model"]["path"],
                    "--data", inputs["data"]["path"], "--out", str(fitted),
                    "--cache", str(run_root / "readout_cache"), "--seed", "0", "--encode-batch", "512",
                    "--steps", "8000", "--batch", "256", "--lr", "0.001", "--hidden", "256", "--depth", "6",
                    "--head-steps", "6000", "--head-batch", "1024", "--head-hidden", "512", "--head-lr", "0.001"]),
                    depends_on=("reset_bank",), outputs=(fitted, fitted.with_suffix(".manifest.json")))]
    dependencies = {
        "reset_bank_sha256": {"producer": "reset_bank", "algorithm": "sha256", "file": str(bank)},
        "jepa_readouts_sha256": {"producer": "jepa_readouts", "algorithm": "sha256", "file": str(fitted)},
    }
    previous = "jepa_readouts"
    for seed in (0, 1, 2):
        for family in ("flow", "jepa"):
            name = family + "_p" + str(seed)
            out = run_root / "policies" / name
            train_id, eval_id = "train_" + name, "eval_" + name
            model = inputs["flow_model"]["path"] if family == "flow" else str(fitted)
            model_hash = inputs["flow_model"]["sha256"] if family == "flow" else {"hash_dependency": "jepa_readouts_sha256"}
            stages.append(stage(train_id, command("inventory_ppo", ["--family", family, "--wm", model,
                "--wm-sha256", model_hash, "--reset-bank", str(bank),
                "--reset-bank-sha256", {"hash_dependency": "reset_bank_sha256"},
                "--seed", str(seed), "--out", str(out)]), depends_on=(previous,),
                outputs=(out / "model.pkl", out / "final.json")))
            stages.append(stage(eval_id, command("inventory_eval", ["--checkpoint", str(out / "model.pkl"),
                "--out", str(out / "native_eval"), "--eval-seed", "123", "--episodes", "256", "--horizon", "10000"]),
                depends_on=(train_id,), outputs=(out / "native_eval" / "summary.json",)))
            previous = eval_id
    attestation = {"schema_version": 1, "kind": "portable_inventory_fresh_rerun_attestation",
        "created_utc": datetime.now(timezone.utc).isoformat(), "bundle_root": str(root), "bundle": bundle,
        "inputs": inputs, "run_root": str(run_root), "python_executable": interpreter,
        "scientific_sources_modified": False, "experiments_executed": False, "results": None,
        "historical_manifest_replay": False,
        "scope": "Fresh rerun of the original fixed data/model recipes; no claimed result or historical replay"}
    plan = {"schema_version": 1, "kind": "portable_inventory_fresh_rerun_plan", "execution": "not_executed",
        "cwd": str(root), "run_root": str(run_root), "hash_dependencies": dependencies, "stages": stages,
        "semantics": [
            "Run strictly in listed order; stop on any nonzero exit or incomplete-budget final artifact.",
            "Use argv arrays directly with the declared cwd; these are not shell command strings.",
            "For a hash_dependency argument, require its producer to finish successfully, hash its file, and pass that digest as one argv element.",
            "All producers and consumers use fresh outputs. Do not reuse historical fitting or policy manifests.",
            "Frozen validators remain enabled. Fresh readouts and policies record the verified bundled source bytes.",
            "Keep bundle and supplied input bytes unchanged throughout; reverify this attestation before executing the plan.",
            "Allocate suitable hardware and dependencies separately. This planner checks neither GPU availability nor installed scientific packages.",
            "The six policies use fixed original PPO defaults and exactly 4998400 completed imagined transitions each; partial budgets are failures.",
            "All six native evaluations use the same seed 123, 256 resets and native horizon 10000; retain every outcome.",
        ]}
    return attestation, plan


def write_plan(destination, attestation, plan):
    destination = new_destination(destination, "Plan directory")
    run_root, bundle_root = Path(plan["run_root"]), Path(attestation["bundle_root"])
    if any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in (run_root, bundle_root)):
        raise PlanError("Plan directory must be separate from the bundle and fresh run root")
    # mkdir is exclusive and parent creation is deliberately not implicit.
    destination.mkdir(parents=False, exist_ok=False)
    for name, value in (("attestation.json", attestation), ("plan.json", plan)):
        with (destination / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return {"status": "plan_written_not_executed", "directory": str(destination),
            "attestation_sha256": sha256(destination / "attestation.json"),
            "plan_sha256": sha256(destination / "plan.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--flow-model", type=Path, required=True)
    parser.add_argument("--jepa-model", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, dest="python_executable")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Print verified attestation and plan; write nothing")
    mode.add_argument("--write-plan", type=Path, help="Create a new private plan directory; execute nothing")
    args = parser.parse_args(argv)
    try:
        attestation, plan = make_plan(args.bundle_root, args.data, args.flow_model, args.jepa_model,
                                     args.run_root, args.python_executable)
        result = {"status": "verified_plan_not_executed", "attestation": attestation, "plan": plan} if args.check else write_plan(args.write_plan, attestation, plan)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, SyntaxError) as error:
        print(json.dumps({"status": "rejected", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
