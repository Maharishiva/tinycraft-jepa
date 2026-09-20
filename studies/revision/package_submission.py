"""Build a local anonymous review draft only after both manuscript versions exist.

No publishing, downloading, training or manuscript editing is performed. Run
``python -m studies.revision.package_submission --check`` for a read-only audit.
After V2 is ready, add ``--v2-ready --out PATH`` to create a new bundle directory.
See research/packaging_protocol.md for the inclusion and anonymization contract.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import tokenize
from urllib.parse import unquote, urlsplit
import zlib


ROOT = Path(__file__).resolve().parents[2]
PUBLICATION_DIRS = ("v1", "v2", "shared", "shared-v2", "html", "html-v2", "html-redcube")
REQUIRED = tuple(f"submission/{version}/{name}" for version in ("v1", "v2")
                 for name in ("main.tex", "main.pdf")) + (
    "submission/html/index.html", "submission/html-v2/index.html", "submission/html-redcube/index.html",
    "research/evidence-ledger.json", "research/craftax_audit.md", "research/redcube_audit.md",
)
CORE_SOURCE = (
    "train.py", "probe.py", "plan.py", "viz.py", "eval.py", "pyproject.toml",
    "studies/redcube/env.py", "studies/redcube/bench.py", "studies/redcube/flow.py",
    "studies/stoch/env.py", "studies/stoch/rl.py", "studies/stoch/sampler.py",
    "studies/stoch/control.py", "studies/stoch/latent_check.py", "studies/stoch/wallmass.py",
    # flow.load_modules loads these by file path, not by an ordinary Python import.
    "studies/stoch/flow_recipe/train.py", "studies/stoch/flow_recipe/eval.py",
    "studies/stoch/flow_recipe/kernels.py",
)
# Explicit scientific entry points. The historical evidence ledger remains
# complete independently of this code list. Machine allocation/queue drivers and
# unlaunched objective lanes are not a public reproduction recipe.
REVISION_SOURCE = (
    "analyze_craftax_replication.py", "analyze_forecast.py", "analyze_horizon48.py",
    "analyze_matched.py", "analyze_target_gradient.py", "analyze_termination.py",
    "audit_redcube_goal.py", "audit_redcube_goal_distance.py", "check_matched.py",
    "craftax_eval.py", "craftax_frames.py", "decoded_dream.py", "decoded_models.py",
    "export_redcube_story.py", "figures.py", "forecast_readout.py",
    "inventory_common.py", "inventory_eval.py", "inventory_flow.py",
    "inventory_flow_parity.py", "inventory_jepa.py", "inventory_jepa_probe.py",
    "inventory_ppo.py", "make_redcube_concepts.py", "matched_ppo.py",
    "matched_ppo_termination.py", "oracle_variance.py", "policy.py", "portable_inventory.py",
    "pve.py", "pve_history.py", "readout_baseline_audit.py",
    "redcube_goal_distance.py", "redcube_goal_eval.py", "refit_full_label_heads.py",
    "render_redcube_goals.py", "shared_decoded_ppo.py", "target_gradient.py",
    "target_gradient_probe.py", "target_gradient_sensitivity.py",
    "verify_target_gradient_execution.py", "warm_context_eval.py",
)
INVENTORY_SOURCE = ("agent.py", "kernels.py", "tinywm_module.py", "train_ppo_real.py",
                    "wm_env2.py", "wm_generic.py", "wm_gym.py")
REPORTS = (
    "research/evidence-ledger.json", "research/evidence-audit.md",
    "research/craftax_audit.md", "research/craftax_evidence.json", "research/craftax_table_v2.json",
    "research/redcube_audit.md", "research/redcube_evidence.json", "research/redcube_table.md",
    "research/redcube_diagnostics.md", "research/theory_review.md",
    "research/experiment_plan.md", "research/experiment_source_audit.md",
    "research/matched_analysis_protocol.md", "research/matched_ppo_theory_review.md",
    "research/forecast_readout_protocol.md", "research/pve_protocol.md",
    "research/target_gradient_protocol.md", "artifacts/revision/craftax_goal_v2.json",
    "research/craftax_inventory_protocol.md", "research/craftax_native_inventory_protocol.md",
    "research/redcube_story_visual_protocol.md", "research/redcube_goal_protocol.md",
    "research/redcube_goal_distance_protocol.md", "research/oracle_variance_design.md",
)
SKIP_DIRS = {"qa", "output", "__pycache__", ".playwright-cli", ".git", ".venv", "cache"}
PUBLICATION_SUFFIXES = {".tex", ".bib", ".sty", ".bst", ".pdf", ".png", ".svg",
                        ".html", ".css", ".js", ".json"}
TEXT_SUFFIXES = PUBLICATION_SUFFIXES - {".pdf", ".png"} | {".py", ".md", ".toml", ".jsonl", ".csv", ".tsv"}
REPORT_SUFFIXES = {".md", ".json", ".jsonl", ".csv", ".tsv"}

# Exact, reviewed substitutions only. Unknown private roots/hosts are errors.
# These internal originals are never written into the public bundle manifest.
REPLACEMENTS = (
    ("revision_checkout_local", "/private/tmp/tinycraft-jepa-iclr-20260919", "revision_checkout"),
    ("revision_checkout_tmp", "/tmp/tinycraft-jepa-iclr-20260919", "revision_checkout"),
    ("revision_checkout_remote", "/home/rldoeswork/personal/tinycraft-jepa-iclr-20260919", "revision_checkout"),
    ("original_checkout", "/Users/rldoeswork/Documents/rl/tinycraft-jepa", "source_checkout"),
    ("original_checkout_remote", "/home/rldoeswork/personal/tinycraft-jepa", "source_checkout"),
    ("figure_python", "/private/tmp/tinycraft-figenv/bin/python", "python"),
    ("old_figure_python", "/tmp/figenv/bin/python", "python"),
    ("cube_summary", "/tmp/tinycraft-redcube-summary.json", "research/redcube_evidence.json"),
    ("cube_table", "/tmp/tinycraft-redcube-table.md", "research/redcube_table.md"),
    ("cube_diagnostics", "/tmp/tinycraft-redcube-diag.md", "research/redcube_diagnostics.md"),
    ("archive_hash_a", "/tmp/tinycraft-jepa-mg-gate-arrayhash.json", "unbundled_hash_records/gate_archive_a.json"),
    ("archive_hash_b", "/tmp/tinycraft-jepa-e2e-gate-arrayhash.json", "unbundled_hash_records/gate_archive_b.json"),
    ("native_craftax_source_snapshot", "/private/tmp/craftax-native-source-20260920", "unbundled_native_craftax_source"),
    ("internal_scope_note", "/Users/rldoeswork/.codex/memories/MEMORY.md:75–76", "internal_scope_note"),
    ("internal_scope_record", "01a0b6fb-e660-7053-a7c9-49280c7e2f93", "internal_scope_record"),
    ("remote_artifact_store", "/home/rldoeswork/personal", "artifact_store"),
    ("compute_host_a_full", "multigpu-vm.sas.yp-c.yandex.net", "compute_host_a"),
    ("compute_host_a", "multigpu-vm", "compute_host_a"),
    ("compute_host_b", "e2e.sas.yp-c.yandex.net", "compute_host_b"),
    ("runtime_environment_b", "tinywmcraft-e2e", "runtime_environment_b"),
    ("anonymous_user", "rldoeswork", "anonymous_user"),
)
ALIAS_RULES = (
    ("compute_host_b_short", re.compile(r"(?<![A-Za-z0-9])e2e(?![A-Za-z0-9])", re.I), "compute_host_b"),
    ("compute_host_a_short", re.compile(r"(?<![A-Za-z0-9])mg(?![A-Za-z0-9])", re.I), "compute_host_a"),
)
IDENTITY_PATTERNS = (
    ("known_username", re.compile(r"rldoeswork", re.I)),
    ("corporate_identity", re.compile(r"yandex|multigpu-vm|\.yp-c\.|\.openclaw|\.codex", re.I)),
    ("private_absolute_path", re.compile(r"(?<![\w])/(?:Users|home|private|tmp|mnt|scratch|root|workspace|srv|media|var/folders)(?:/|\b)")),
    ("named_home", re.compile(r"~[A-Za-z0-9_.-]+/")),
    ("private_uri", re.compile(r"(?:file|ssh)://", re.I)),
    ("private_ipv4", re.compile(r"\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b")),
    ("secret_material", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}")),
    ("known_compute_host_alias", re.compile(r"(?<![A-Za-z0-9])(?:e2e|mg)(?![A-Za-z0-9])", re.I)),
)
DIGEST = re.compile(r"(?<![a-fA-F0-9])(?:[a-fA-F0-9]{64}|[a-fA-F0-9]{40})(?![a-fA-F0-9])")
STORY_GLOBALS = {"fresh-story.js": "FRESH_STORY", "historical-story.js": "HISTORICAL_STORY",
                 "craftax-data.js": "CRAFTAX_EVIDENCE"}


class PackageError(RuntimeError):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(path):
    p = Path(path)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise PackageError(f"expected a source-relative path: {path}")
    if any(part.startswith(".") or part in SKIP_DIRS for part in p.parts):
        raise PackageError(f"hidden/cache path is not eligible: {path}")
    return p


def local_dependencies(root, relative):
    """Resolve existing local imports, including function-local imports."""
    tree = ast.parse((root / relative).read_text())
    found = set()
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            modules = [prefix] + [f"{prefix}.{alias.name}" for alias in node.names]
        else:
            continue
        for module in modules:
            if not module:
                continue
            name = Path(*module.split("."))
            bases = [relative.parent, Path()] if getattr(node, "level", 0) else [Path(), relative.parent]
            for base in bases:
                for candidate in (base / name.with_suffix(".py"), base / name / "__init__.py"):
                    if (root / candidate).is_file():
                        found.add(candidate)
                        break
                else:
                    continue
                break
    return found


def select_files(root, reports=()):
    selected = {Path(p) for p in CORE_SOURCE}
    for directory in PUBLICATION_DIRS:
        for path in (root / "submission" / directory).rglob("*"):
            relative = path.relative_to(root)
            if any(p in SKIP_DIRS or p.startswith(".") for p in relative.parts):
                continue
            if path.is_file() and path.suffix.lower() in PUBLICATION_SUFFIXES:
                selected.add(relative)
    selected.update(Path("studies/revision") / name for name in REVISION_SOURCE)
    # The inventory adapter imports these frozen dependencies through sys.path.
    # Its internal source-location manifest is intentionally not public input.
    selected.update(Path("studies/revision/inventory_source") / name for name in INVENTORY_SOURCE)
    selected.update(Path(p) for p in REPORTS if (root / p).is_file())
    for item in reports:
        relative = safe_relative(item)
        if relative.suffix not in REPORT_SUFFIXES or not (
                relative.parts[0] == "research" or relative.parts[:2] == ("artifacts", "revision")):
            raise PackageError(f"extra report must be text under research/ or artifacts/revision/: {item}")
        if any(x in relative.parts for x in ("reading", "gpu_inventory", "reviews")):
            raise PackageError(f"internal/third-party report path is excluded: {item}")
        if any(word in relative.name.lower() for word in ("partial", "smoke", "snapshot", "gpu_inventory")):
            raise PackageError(f"incomplete/internal artifact is excluded: {item}")
        selected.add(relative)
    pending = [p for p in selected if p.suffix == ".py"]
    while pending:
        path = pending.pop()
        if not (root / path).is_file():
            raise PackageError(f"required source missing: {path}")
        for dependency in local_dependencies(root, path) - selected:
            selected.add(dependency)
            pending.append(dependency)
    for path in selected:
        safe_relative(path)
        if not (root / path).is_file():
            raise PackageError(f"selected file missing: {path}")
        if (root / path).is_symlink() or not (root / path).resolve().is_relative_to(root.resolve()):
            raise PackageError(f"symlink or escaped source path is not eligible: {path}")
    return sorted(selected)


def scan_identity(text, relative):
    for label, pattern in IDENTITY_PATTERNS:
        inspected = text
        if label == "known_compute_host_alias":
            if Path(relative).suffix in {".bib", ".sty", ".bst"}:
                continue  # A cited title may legitimately abbreviate end-to-end.
            if Path(relative).suffix == ".py":
                # In the flow model, `mg` is a modulation-gate variable, not a host.
                inspected = "\n".join(token.string for token in tokenize.generate_tokens(io.StringIO(text).readline)
                                      if token.type in {tokenize.STRING, tokenize.COMMENT})
        match = pattern.search(inspected)
        if match:
            # Report the rule and line, not a potentially sensitive surrounding line.
            line = inspected.count("\n", 0, match.start()) + 1
            raise PackageError(f"identity scan failed [{label}] in {relative}:{line}; review explicitly")


class RemoveDocstrings(ast.NodeTransformer):
    def strip(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node
    visit_Module = strip
    visit_ClassDef = strip
    visit_FunctionDef = strip
    visit_AsyncFunctionDef = strip


def json_signature(text):
    """Preserve every numeric/boolean/null leaf and tree shape; strings may alias."""
    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise PackageError("JSON key collision during anonymization")
        return dict(items)
    data = json.loads(text, parse_int=lambda x: ("integer", x), parse_float=lambda x: ("float", x),
                      parse_constant=lambda x: ("constant", x), object_pairs_hook=pairs)
    def visit(obj):
        if isinstance(obj, dict):
            return ("object", tuple(visit(value) for value in obj.values()))
        if isinstance(obj, list):
            return ("array", tuple(visit(value) for value in obj))
        return "string" if isinstance(obj, str) else obj
    return visit(data)


def embedded_json(text, relative):
    """Only a single reviewed global assignment is evidence, never arbitrary JS."""
    name = Path(relative).name
    if name in STORY_GLOBALS:
        pattern = r"\s*window\." + STORY_GLOBALS[name] + r"\s*=\s*(.*?)\s*;?\s*"
        match = re.fullmatch(pattern, text, re.DOTALL)
        if not match:
            raise PackageError(f"invalid evidence assignment in {relative}")
        return match.group(1)
    return text.split("=", 1)[1].strip().removesuffix(";")


def check_metadata_identity(text, relative):
    """Do not silently retain an unfamiliar host/user field in new reports."""
    allowed = {"anonymous_user", "compute_host_a", "compute_host_b", "localhost", "127.0.0.1"}
    keys = {"host", "hostname", "user", "username", "remote_host", "ssh_host"}
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in keys and isinstance(child, str) and child and child not in allowed:
                    raise PackageError(f"unreviewed identifying metadata field {key!r} in {relative}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(json.loads(text))


def sanitize_text(text, relative):
    result, changes = text, []
    for rule, original, alias in sorted(REPLACEMENTS, key=lambda item: -len(item[1])):
        count = result.count(original)
        if count:
            result = result.replace(original, alias)
            changes.append({"rule": rule, "alias": alias, "count": count})
    if relative.suffix not in {".bib", ".sty", ".bst"}:
        for rule, pattern, alias in ALIAS_RULES:
            if relative.suffix == ".py":
                # Only textual Python tokens are eligible for host shorthand.
                # Executable string edits still fail the AST invariant below.
                offsets, position = [], 0
                for line in result.splitlines(keepends=True):
                    offsets.append(position)
                    position += len(line)
                edits, count = [], 0
                for token in tokenize.generate_tokens(io.StringIO(result).readline):
                    if token.type in {tokenize.STRING, tokenize.COMMENT}:
                        replacement, hits = pattern.subn(alias, token.string)
                        if hits:
                            start = offsets[token.start[0] - 1] + token.start[1]
                            end = offsets[token.end[0] - 1] + token.end[1]
                            edits.append((start, end, replacement))
                            count += hits
                for start, end, replacement in reversed(edits):
                    result = result[:start] + replacement + result[end:]
            else:
                result, count = pattern.subn(alias, result)
            if count:
                changes.append({"rule": rule, "alias": alias, "count": count})
    if Counter(DIGEST.findall(text)) != Counter(DIGEST.findall(result)):
        raise PackageError(f"canonical digest changed in {relative}")
    if relative.suffix in {".bib", ".sty", ".bst"} and result != text:
        raise PackageError(f"bibliography/style would be changed: {relative}")
    if relative.suffix == ".py" and result != text:
        before = ast.dump(RemoveDocstrings().visit(ast.parse(text)), include_attributes=False)
        after = ast.dump(RemoveDocstrings().visit(ast.parse(result)), include_attributes=False)
        if before != after:
            raise PackageError(f"sanitization would change executable Python: {relative}")
    if relative.suffix == ".json":
        if json_signature(text) != json_signature(result):
            raise PackageError(f"JSON measurements changed in {relative}")
        check_metadata_identity(result, relative)
    if relative.suffix == ".jsonl":
        for before, after in zip(text.splitlines(), result.splitlines(), strict=True):
            if before.strip() and json_signature(before) != json_signature(after):
                raise PackageError(f"JSONL measurements changed in {relative}")
            if after.strip():
                check_metadata_identity(after, relative)
    if relative.name in {"evidence.js", *STORY_GLOBALS}:
        payload = lambda value: embedded_json(value, relative)
        if json_signature(payload(text)) != json_signature(payload(result)):
            raise PackageError(f"embedded evidence measurements changed in {relative}")
        check_metadata_identity(payload(result), relative)
    scan_identity(result, relative)
    return result.encode(), changes


def scan_binary(path, relative, data):
    """Read metadata/text without modifying scientific figure or PDF bytes."""
    # Compressed pixel/font streams are not text: scanning arbitrary compressed
    # bytes for short path patterns creates spurious identity matches.
    text = ""
    if relative.suffix == ".pdf":
        for executable in ("pdftotext", "pdfinfo"):
            if shutil.which(executable) is None:
                raise PackageError(f"{executable} is required for PDF privacy checks")
        commands = [["pdftotext", str(path), "-"], ["pdfinfo", "-custom", str(path)],
                    ["pdfinfo", "-meta", str(path)], ["pdfinfo", "-url", str(path)],
                    ["pdfinfo", "-js", str(path)]]
        for command in commands:
            output = subprocess.run(command, check=True, capture_output=True, timeout=60)
            text += "\n" + output.stdout.decode(errors="strict")
    elif relative.suffix == ".png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise PackageError(f"invalid PNG: {relative}")
        offset = 8
        while offset < len(data):
            length = struct.unpack(">I", data[offset:offset + 4])[0]
            kind, payload = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + length]
            if kind == b"zTXt":
                _, compressed = payload.split(b"\0", 1)
                text += "\n" + zlib.decompress(compressed[1:]).decode("latin-1")
            elif kind == b"iTXt":
                _, rest = payload.split(b"\0", 1)
                compressed, _, rest = rest[0], rest[1], rest[2:]
                _, _, content = rest.split(b"\0", 2)
                text += "\n" + (zlib.decompress(content) if compressed else content).decode("utf-8")
            elif kind in {b"tEXt", b"eXIf"}:
                text += "\n" + payload.decode("latin-1")
            offset += length + 12
    else:
        raise PackageError(f"unreviewed binary file type: {relative}")
    scan_identity(text, relative)


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attr in ("href", "src", "poster"):
            if attrs.get(attr):
                resource = attr != "href" or tag == "link"
                self.links.append((attrs[attr], resource, f"html:{tag}:{attr}"))
        if "srcset" in attrs:
            for value in attrs["srcset"].split(","):
                self.links.append((value.strip().split()[0], True, "html:srcset"))


def png_size(data, relative):
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        raise PackageError(f"invalid PNG header: {relative}")
    return struct.unpack(">II", data[16:24])


def validate_links(payloads, pending=()):
    """Check all declared HTML/CSS/TeX links, plus saved-frame sprite references."""
    links, parsed_html = [], {}
    for path, data in payloads.items():
        if path.suffix == ".html":
            parsed = HTMLLinks()
            parsed.feed(data.decode())
            parsed_html[path] = parsed

    def macro_target(compile_dir, name, argument):
        main = payloads.get(compile_dir / "main.tex", b"").decode()
        pattern = r"\\(?:re)?newcommand\s*\{\s*\\" + name + r"\s*\}\s*\[\d+\]\s*\{"
        declaration = re.search(pattern, main)
        if declaration is None:
            raise PackageError(f"cannot verify {name} file macro in {compile_dir}/main.tex")
        depth, end = 1, declaration.end()
        while end < len(main) and depth:
            if main[end] in "{}" and main[end - 1] != "\\":
                depth += 1 if main[end] == "{" else -1
            end += 1
        body = main[declaration.end():end - 1]
        candidates = {match.group(1) for match in re.finditer(
            r"\\(?:IfFileExists|includegraphics|input)(?:\[[^]]*\])?\{([^}]*#1[^}]*)\}", body)}
        if depth or len(candidates) != 1:
            raise PackageError(f"ambiguous or unsupported {name} file macro in {compile_dir}/main.tex")
        return candidates.pop().replace("#1", argument)

    def add(source, target, resource, kind, extensions=(), base=None):
        parts = urlsplit(target)
        if parts.scheme in {"http", "https", "mailto"} or parts.netloc:
            if resource:
                raise PackageError(f"external runtime resource in {source}: {target}")
            links.append({"source": str(source), "target": target, "kind": kind, "status": "external_reference_not_fetched"})
            return
        if parts.scheme == "data":
            return
        if parts.scheme or parts.path.startswith("/"):
            raise PackageError(f"nonrelative local link in {source}: {target}")
        candidate = source if not parts.path else (base or source.parent) / unquote(parts.path)
        # Normalize '..' without letting a link escape the logical bundle root.
        components = []
        for item in candidate.parts:
            if item == "..":
                if not components:
                    raise PackageError(f"link escapes bundle: {source}: {target}")
                components.pop()
            elif item != ".":
                components.append(item)
        candidate = Path(*components)
        if candidate not in payloads:
            candidate = next((candidate.with_suffix(ext) for ext in extensions
                              if candidate.with_suffix(ext) in payloads), candidate)
        if candidate not in payloads:
            if str(candidate) in pending:
                links.append({"source": str(source), "target": target, "resolved": str(candidate),
                              "kind": kind, "status": "awaiting_required_final_artifact"})
                return
            raise PackageError(f"unbundled local link in {source}: {target}; declare the required report/asset")
        if parts.fragment and candidate in parsed_html and unquote(parts.fragment) not in parsed_html[candidate].ids:
            raise PackageError(f"missing HTML anchor in {source}: {target}")
        links.append({"source": str(source), "target": target, "resolved": str(candidate), "kind": kind, "status": "verified_local"})
        return candidate

    def frame_asset(path, target, html_root, expected_size, expected_hash=None, kind="redcube_frame_sprite"):
        candidate = add(path, target, True, kind, base=html_root)
        if candidate is None or candidate.suffix != ".png":
            raise PackageError(f"frame asset must be a bundled PNG in {path}")
        data = payloads[candidate]
        if png_size(data, candidate) != tuple(expected_size):
            raise PackageError(f"frame dimensions disagree with story data in {candidate}")
        if expected_hash is not None and sha256(data) != expected_hash:
            raise PackageError(f"frame digest disagrees with story data in {candidate}")
        return candidate

    for path, data in payloads.items():
        if path in parsed_html:
            for target, resource, kind in parsed_html[path].links:
                add(path, target, resource, kind)
        elif path.suffix == ".css":
            for match in re.finditer(r"url\(\s*['\"]?([^)'\"\s]+)|@import\s+['\"]([^'\"]+)", data.decode()):
                add(path, match.group(1) or match.group(2), True, "css:resource")
        elif path.suffix == ".tex":
            text = re.sub(r"(?<!\\)%[^\n]*", "", data.decode())
            for match in re.finditer(r"\\(input|include|includegraphics|paperfig|inputtable|bibliography|bibliographystyle)(?:\[[^]]*\])?\{([^}]+)\}", text):
                command, target = match.groups()
                if "#" in target or "\\" in target:
                    continue
                compile_dir = Path(*path.parts[:2]) if path.parts[0] == "submission" else path.parent
                if command in {"paperfig", "inputtable"}:
                    target = macro_target(compile_dir, command, target)
                extensions = (".bib",) if command == "bibliography" else (".bst",) if command == "bibliographystyle" else (".pdf", ".png", ".svg") if command in {"paperfig", "includegraphics"} else (".tex",)
                for item in target.split(","):
                    # TeX resolves included paths from the main compile directory,
                    # not from the directory of each included section file.
                    add(path, item.strip(), True, f"tex:{command}", extensions, base=compile_dir)
        elif path.suffix == ".md":
            for match in re.finditer(r"\[[^]]+\]\(([^)\s]+)\)", data.decode()):
                add(path, match.group(1), False, "markdown:link")
        if path.name == "evidence.json" and path.parent.name == "assets":
            evidence = json.loads(data)
            for seed, record in evidence.get("rollouts", {}).items():
                for key in record["keys"]:
                    add(path, f"rollout-s{seed}-{key}.png", True, "saved_rollout_sprite")
        if path.name == "metadata.json" and path.parent.name == "craftax":
            metadata = json.loads(data)
            scenes = metadata.get("scenes", [])
            if [scene.get("episode") for scene in scenes] != metadata.get("selected_episodes"):
                raise PackageError(f"Craftax scene selection metadata disagrees in {path}")
            for scene in scenes:
                if set(scene.get("frames", {})) != {"reset", "start", "goal"}:
                    raise PackageError(f"Craftax frame streams missing in {path}")
                for frame in scene["frames"].values():
                    if frame.get("shape") != [63, 63, 3] or not frame.get("sha256"):
                        raise PackageError(f"Craftax frame metadata invalid in {path}")
                    frame_asset(path, frame["path"], path.parent, (63, 63), frame["sha256"], "craftax_dynamic_frame")
        if path.name in {"fresh-story.js", "historical-story.js"}:
            story = json.loads(embedded_json(data.decode(), path))
            # Paths in the article data are relative to the HTML document,
            # rather than the JavaScript asset directory.
            html_root = Path(*path.parts[:2])
            if not isinstance(story.get("conditions"), dict) or not story["conditions"]:
                raise PackageError(f"missing story conditions in {path}")
            for condition, record in story["conditions"].items():
                for key in ("scenes", "context", "horizon"):
                    if type(record.get(key)) is not int or record[key] <= 0:
                        raise PackageError(f"invalid frame geometry {condition}:{key} in {path}")
                scenes, horizon = record["scenes"], record["horizon"]
                size = (63 * (record["context"] + horizon), 63 * scenes)
                if len(record.get("selectedWindows", [])) != scenes:
                    raise PackageError(f"scene selection length disagrees in {path}:{condition}")
                fresh = path.name == "fresh-story.js"
                expected_keys = {"gt", "jepa", "flow"} | ({"real_encoded", "flow_encoded"} if fresh else set())
                if set(record.get("keys", {})) != expected_keys:
                    raise PackageError(f"missing or unknown frame stream in {path}:{condition}")
                manifest = None
                if fresh:
                    for field in ("actions", "validMasks"):
                        rows = record.get(field, [])
                        if len(rows) != scenes or any(len(row) != horizon for row in rows):
                            raise PackageError(f"invalid {field} shape in {path}:{condition}")
                    if any(type(a) is not int or a not in range(4) for row in record["actions"] for a in row):
                        raise PackageError(f"invalid native action in {path}:{condition}")
                    if any(type(valid) is not bool for row in record["validMasks"] for valid in row):
                        raise PackageError(f"invalid frame-validity flag in {path}:{condition}")
                    if not record.get("manifest"):
                        raise PackageError(f"missing fresh-frame manifest in {path}:{condition}")
                    candidate = add(path, record["manifest"], True, "redcube_frame_manifest", base=html_root)
                    manifest = json.loads(payloads[candidate])
                    if manifest.get("case_id") != condition or manifest.get("scene_ids") != record["selectedWindows"]:
                        raise PackageError(f"fresh-frame selection/condition manifest mismatch in {path}:{condition}")
                    if set(manifest.get("display_sprite_sha256", {})) != expected_keys:
                        raise PackageError(f"fresh-frame manifest digest keys differ in {path}:{condition}")
                for key, target in record["keys"].items():
                    evidence = record.get("assets", {}).get(key, {})
                    if not fresh and (evidence.get("path") != target or not evidence.get("sha256")):
                        raise PackageError(f"historical asset provenance missing in {path}:{condition}:{key}")
                    if not fresh and (evidence.get("width"), evidence.get("height")) != size:
                        raise PackageError(f"historical asset dimensions differ in {path}:{condition}:{key}")
                    expected_hash = manifest["display_sprite_sha256"][key] if fresh else evidence["sha256"]
                    frame_asset(path, target, html_root, size, expected_hash)
                for stream, posters in record.get("posters", {}).items():
                    if stream not in expected_keys:
                        raise PackageError(f"unknown poster stream in {path}:{condition}")
                    for poster in posters.values():
                        if not (type(poster.get("sceneIndex")) is int and 0 <= poster["sceneIndex"] < scenes
                                and type(poster.get("timelineIndex")) is int
                                and 0 <= poster["timelineIndex"] < record["context"] + horizon):
                            raise PackageError(f"poster index outside declared timeline in {path}:{condition}")
                        if (poster.get("width"), poster.get("height")) != (63, 63) or not poster.get("sha256"):
                            raise PackageError(f"poster metadata invalid in {path}:{condition}")
                        frame_asset(path, poster["path"], html_root, (63, 63), poster["sha256"], "redcube_frame_poster")
    return links


README = """# Anonymous research draft

This is a local anonymous review draft, not a conference submission. It has not
been submitted or published. No acceptance or independent 9/10 rating is claimed.

- [V1 manuscript](submission/v1/main.pdf) preserves the first rewrite.
- [V2 manuscript](submission/v2/main.pdf) contains the separately identified revision.
- [Revised red-cube article](submission/html-redcube/index.html) presents the final visual story.
- [V1 HTML companion](submission/html/index.html) and [earlier V2 companion](submission/html-v2/index.html)
  preserve the earlier drafts with their neighboring assets and report links.
- [File manifest](manifest.json) records original and bundled SHA-256 identities.
- [Inventory source provenance](inventory-provenance.json) records the original
  and derived identities of the frozen inventory dependencies without private
  source locations.
- [Link manifest](link-manifest.json) records checked local references. External
  literature links are optional and were not fetched by the packager.

Open the HTML files directly. Their local scripts, styles and frame sprites do
not require internet services. If a browser restricts local files, serve this
bundle root with `python -m http.server 8000` and open the same relative page.
For TeX, run `tectonic -X compile main.tex` from the chosen version directory;
a working TeX installation and its packages are prerequisites.

Source files needed by the reported measurements and their local imports are
included. Datasets, model checkpoints, raw rollout archives, machine configuration,
experiment launch lanes, screenshot QA, caches, auxiliary TeX files and downloaded
third-party papers are excluded. The bundle alone therefore does not reproduce
training without the separately identified data and checkpoint inputs.

For a fresh inventory rerun, start with
`python -m studies.revision.portable_inventory --help`. This entrypoint verifies
the bundle and the three user-supplied input hashes, then produces a dry-run
command plan and attestation. It performs no downloads or experiment execution.
The plan fits fresh frozen JEPA readouts and creates a fresh reset bank before
the six fixed policy runs and final native evaluations. These runs obtain their
own source attestations in this checkout; the frozen validators are unchanged.
Do not substitute the original fitted-readout manifest or policy final.json.
The former records original input paths and model/readout source hashes; the
latter additionally attests private source-location records and original
queue/protocol bytes that are intentionally not the public runtime.
Exact historical replay requires those original inputs from the internal
research worktree. The public source-to-derived manifest documents this boundary
and is not a replacement accepted by the historical validators.

The inventory runtime uses its bundled source directory through
`inventory_flow.py`. Its preserved generic `~/personal` fallback and generic
`/dev/shm` paths are storage conventions, not machine identities. The copied
`inventory_source/train_ppo_real.py` and `wm_gym.py` are historical provenance;
use the new `inventory_ppo.py` and `inventory_eval.py` entrypoints. The historical
trainer's external `craftax_env` helper is not supplied or needed by this route.

Copied audit/provenance text uses consistent aliases for author-specific paths
and machines. Recorded artifact paths describe omitted inputs; they are not
promises that a file is bundled. Original canonical artifact hashes and every
JSON numeric result are retained. Source and derived-copy hashes distinguish
anonymized files. Executable Python is unchanged; only comments/docstrings can
be anonymized. Bibliographic authors and official style sources are unchanged.

The archival evidence ledger retains its historical rows for traceability;
their presence does not promote excluded experiments into manuscript claims.
Automated identity/link checks supplement, and do not replace, a human anonymity
and scientific review before any eventual submission.
"""


def inventory_provenance(root, records):
    """Public hash-only derivative; never copy the internal source-location map."""
    original_path = root / "studies/revision/inventory_source/SOURCE_MANIFEST.json"
    original = original_path.read_bytes()
    locations = json.loads(original)
    if set(locations) != set(INVENTORY_SOURCE):
        raise PackageError("inventory source-location attestation has unexpected members")
    by_path = {record["path"]: record for record in records}
    result = []
    for name in INVENTORY_SOURCE:
        relative = "studies/revision/inventory_source/" + name
        record = by_path[relative]
        if locations[name].get("sha256") != record["source_sha256"]:
            raise PackageError(f"frozen inventory dependency differs from its original attestation: {relative}")
        result.append({"path": relative, "source_sha256": record["source_sha256"],
                       "bundle_sha256": record["bundle_sha256"],
                       "role": "historical_reference" if name in {"train_ppo_real.py", "wm_gym.py"} else "runtime_dependency"})
    return {"schema_version": 1, "status": "public_hash_only_derivative",
            "original_source_location_manifest_sha256": sha256(original),
            "source_locations_included": False, "historical_validator_substitute": False,
            "records": result,
            "fresh_rerun": "Generate new readout and policy attestations in this checkout; do not reuse historical runtime attestations."}


def prepare(root, reports=()):
    payloads, records = {}, []
    for relative in select_files(root, reports):
        original = (root / relative).read_bytes()
        scan_identity(str(relative), "bundle filename")
        if relative.suffix in TEXT_SUFFIXES:
            derived, changes = sanitize_text(original.decode("utf-8"), relative)
        else:
            scan_binary(root / relative, relative, original)
            derived, changes = original, []
        payloads[relative] = derived
        records.append({"path": str(relative), "source_sha256": sha256(original),
                        "bundle_sha256": sha256(derived), "source_bytes": len(original),
                        "bundle_bytes": len(derived), "sanitization": changes})
    provenance = inventory_provenance(root, records)
    payloads[Path("inventory-provenance.json")] = (json.dumps(provenance, indent=2) + "\n").encode()
    scan_identity(payloads[Path("inventory-provenance.json")].decode(), "inventory-provenance.json")
    missing = [name for name in REQUIRED if Path(name) not in payloads]
    links = validate_links(payloads, pending=missing)
    return payloads, records, links, missing


def build(root, out, reports=(), v2_ready=False):
    if not v2_ready:
        raise PackageError("building requires --v2-ready after the V2 artifacts are final")
    if out.exists():
        raise PackageError("refusing to overwrite an existing bundle directory")
    payloads, records, links, missing = prepare(root, reports)
    if missing:
        raise PackageError("required final artifacts missing: " + ", ".join(missing))
    payloads[Path("README.md")] = README.encode()
    # Generated indexes are intentionally outside their own hash list.
    manifest = {"schema_version": 1, "status": "anonymous_local_draft_not_submitted",
                "v2_readiness_asserted": True, "records": records,
                "checks": {"identity_scan": "passed", "local_links": "passed",
                           "canonical_digest_tokens": "unchanged", "json_numeric_results": "unchanged",
                           "executable_python_ast": "unchanged", "bib_and_style_bytes": "unchanged"},
                "generated_readme_sha256": sha256(payloads[Path("README.md")]),
                "generated_files": {name: sha256(payloads[Path(name)])
                                    for name in ("README.md", "inventory-provenance.json")}}
    payloads[Path("manifest.json")] = (json.dumps(manifest, indent=2) + "\n").encode()
    payloads[Path("link-manifest.json")] = (json.dumps({"links": links}, indent=2) + "\n").encode()
    # Include the generated README's own relative links in the final validation.
    all_links = validate_links(payloads)
    payloads[Path("link-manifest.json")] = (json.dumps({"links": all_links}, indent=2) + "\n").encode()
    for relative in (Path("README.md"), Path("manifest.json"), Path("link-manifest.json")):
        scan_identity(payloads[relative].decode(), relative)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".anonymous-draft-staging-", dir=out.parent))
    try:
        for relative, data in payloads.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        if out.exists():
            raise PackageError("bundle destination appeared during preparation")
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return {"status": "anonymous_local_draft_not_submitted", "files": len(payloads),
            "local_links": sum(link["status"] == "verified_local" for link in all_links),
            "manifest_sha256": sha256(payloads[Path("manifest.json")])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", action="append", default=[], help="additional finalized report, relative to root")
    parser.add_argument("--check", action="store_true", help="read-only preflight; no bundle is created")
    parser.add_argument("--v2-ready", action="store_true", help="assert that V2 is finalized for this local draft")
    args = parser.parse_args()
    try:
        if args.check:
            payloads, records, links, missing = prepare(args.root.resolve(), args.report)
            print(json.dumps({"status": "ready" if not missing else "awaiting_final_artifacts",
                              "selected_files": len(payloads), "sanitized_files": sum(bool(r["sanitization"]) for r in records),
                              "checked_links": len(links), "pending_links": sum(link["status"] == "awaiting_required_final_artifact" for link in links),
                              "missing_required": missing}, indent=2))
            return 0 if not missing else 2
        if args.out is None:
            parser.error("--out is required unless --check is used")
        print(json.dumps(build(args.root.resolve(), args.out.resolve(), args.report, args.v2_ready), indent=2))
        return 0
    except (PackageError, UnicodeError, subprocess.SubprocessError, ValueError, OSError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
