# Anonymous local draft packaging protocol

The builder is `studies/revision/package_submission.py`. It makes a separate local
review bundle only when the author explicitly supplies `--v2-ready` and both V1
and V2 have manuscript source, PDF and HTML artifacts. This flag records artifact
readiness; it is not permission to submit or publish. The builder performs no
network access, experiment execution, manuscript editing or publishing.

## Layout and inclusion

The bundle preserves `submission/{v1,v2,shared,shared-v2,html,html-v2,html-redcube}` under its root. It
retains each version separately, official TeX source/style/bibliography, final PDF,
figure assets, HTML/CSS/JavaScript and local frame/evidence assets. It excludes
screenshots, QA directories, caches, generated TeX auxiliaries and editing scripts.

The fixed report allowlist includes the evidence ledger, Craftax and nuisance
audits and numeric evidence, relevant methods/protocol reviews, and the completed
Craftax follow-up JSON when present. The V1 HTML's relative links to
`research/evidence-ledger.json`, `research/craftax_audit.md` and
`research/redcube_audit.md` therefore continue to work. Final V2 reports can be
added explicitly with repeatable `--report research/NAME.json` arguments. An
unbundled local hyperlink aborts packaging; the builder never silently drops it.

Included source starts from root `train.py`, `probe.py`, `plan.py`, `viz.py`,
`eval.py`, the environment/control/readout dependencies in `studies/redcube` and
`studies/stoch`, the separate flow recipe, and the explicit `REVISION_SOURCE`
allowlist of measured/admitted revision experiment and analysis scripts. New
scripts do not enter merely because they exist in the directory. Host-specific
and queue launchers, unlaunched last-only objective lanes, and their tests are
excluded. Existing local Python imports are followed transitively; this includes
`studies/compare.py`, required by the nuisance benchmark. The flow recipe's
`train.py`, `eval.py` and `kernels.py` are explicitly included because their loader
uses dynamic file imports. The packaging script itself and standalone unit tests
are internal tooling and are excluded.

The current visual narrative is `html-redcube`; the other two HTML copies are
retained as earlier drafts. Its fresh and historical JavaScript data payloads
receive the same numeric/digest preservation checks as the evidence data, and
every declared frame sprite and frame manifest must resolve in the bundle.
The native 63-pixel sprite geometry, scene counts, action and validity-mask
shapes, native action range, selection identities and sprite SHA-256 values are
checked against the declared payload and manifest. Historical posters are also
checked for size, digest and timeline index. The earlier Craftax companion's
dynamic scene images are checked through all nine entries of its metadata,
including scenes not visible in the initial HTML. Its embedded `craftax-data.js`
numbers receive the same preservation check as the other evidence payloads.
The inventory-study adapter's frozen Python dependencies are explicitly included
because they are imported through a dynamic source-directory path. The internal
source-location manifest is excluded. Final inventory-achievement reports can
be explicitly admitted; machine GPU inventories remain excluded.

The generated `inventory-provenance.json` is a new hash-only derivative: it
contains the original source-location manifest's SHA-256 and original/derived
hashes for the seven frozen source files, without any source-location strings.
The dependency hashes are checked against the original internal attestation
before this derivative is generated. It explicitly distinguishes five runtime
dependencies from the two historical source references. It does not substitute
for the private manifest in any frozen validator.

`portable_inventory.py` provides a separate, dry-run-by-default fresh-rerun
entrypoint. It verifies the bundle, requires the exact three user-supplied
data/model hashes, and produces an attested command plan. It does not download
inputs or run experiments. A fresh decoder/head fit records the anonymous
checkout's actual source map; fresh policy training and evaluation then use the
same map under the unchanged validators. Historical fitted-readout and final
policy manifests must not be substituted. The fitted readout attests original
input paths and four model/readout source files; the final policy additionally
attests omitted private source records and original queue/protocol bytes. Exact historical-runtime replay
therefore requires the original research worktree and inputs. The public
README states this boundary and identifies the portable entrypoints. The
historical `train_ppo_real.py`'s external `craftax_env` helper is not bundled or
needed by the new route; `inventory_flow.py` selects the bundled source directory.

Do not add third-party reading PDFs/text/screenshots, downloaded paper caches,
original VM source snapshots, GPU inventories, credentials, private configuration,
datasets, checkpoints, raw rollout archives, legacy launch lanes or amplification
experiment output files. The archival evidence ledger remains complete and
numerically unchanged, including its clearly identified historical rows; their
presence does not make them new paper claims. The source and canonical hashes
support provenance, but omitted data/checkpoints remain prerequisites for reruns.

## Anonymization and scientific invariants

Internal originals are never modified. The builder applies a fixed, reviewed map
to copied text: original/revision checkouts, author-specific artifact storage
roots, known compute hosts and a small number of historical report
paths become consistent relative aliases. It records the rule identifier, alias
and replacement count, without recording the identifying original string in the
public manifest. Unknown identity matches abort; there is no generic redact-all
fallback. A new identifier must be reviewed explicitly before extending the map.
Generic `/dev/shm` and `~/personal` storage conventions are preserved, including
runtime defaults. These strings alone contain no author or host identity.
Known identities nested under them still trigger the usual checks; this is not
a general exception for identifying paths or executable-string modification.

Every canonical 40- or 64-character hexadecimal digest token is preserved. JSON
and JSONL numeric, boolean and null leaves and container structure are checked
before/after, including all four recognized embedded evidence JavaScript payloads.
Source and derived-file
SHA-256 values are both recorded. Python edits must preserve its executable AST
after removing docstrings, restricting anonymization to comments/docstrings;
runtime strings cannot be silently changed. Bibliography and official style
files are byte-preserved. Public literature authors and third-party source/data
URLs are not anonymity targets.

Identity scans cover known user/corporate strings and reviewed short host aliases,
private absolute path roots,
named-home/file/SSH references, private IPv4 addresses and recognizable private
keys/tokens. Unfamiliar structured host/user fields in JSON reports also fail.
Short aliases in Python are checked only in text tokens, so a mathematical
variable that happens to share a host abbreviation is not renamed. PDF bytes stay unchanged; Poppler extracts their text, metadata,
links and JavaScript for scanning. PNG text metadata, including compressed text,
is scanned without changing image bytes. Unreviewed binary types fail. These are
mechanical checks, not a proof of anonymity: human review of final PDF/HTML
appearance and content remains required before an eventual submission.

## Offline links and completeness

The builder validates HTML resources and anchors, CSS resources, concrete TeX
inputs/figures/bibliographies, Markdown links and all declared saved-frame sprites.
Fresh and historical story scripts must contain one recognized global JSON
assignment; trailing executable JavaScript cannot masquerade as evidence data.
TeX figure/table macro paths are read from each version's actual main source;
V2 figures cannot be accidentally checked against V1's shared directory.
Local references must resolve inside the selected bundle. External citation links
are recorded but not fetched. External runtime resources are rejected. Keep
the complete directory layout when moving the bundle; HTML can be opened
directly or served from the bundle root with a local HTTP server.

The root README explicitly labels the output an anonymous research draft, not a
submission, and makes no acceptance or independent 9/10 claim. `manifest.json`
provides source/derived identities and hashes for the generated README and public
inventory provenance; `link-manifest.json` provides verified local
links and unfetched external references. The destination must not exist. A
temporary staging directory is renamed only after successful preparation and
validation, so an error cannot leave a partially labeled final bundle.

## Commands

Read-only preflight, which is permitted before V2 is complete:

```sh
python -m studies.revision.package_submission --check
```

This returns exit status 2 and lists missing required artifacts while V2 is
pending; validation errors return 1. A ready complete preflight returns 0.
"Ready" means the currently present bytes pass mechanical checks. It does not
assert that the manuscript, pending experiments, independent reviews or visual
QA are final. Re-run preflight after all concurrent authoring has stopped.

After V2 and all intended reports are final, create a new local bundle:

```sh
python -m studies.revision.package_submission --v2-ready \
  --report research/FINAL_V2_REPORT.json \
  --out anonymous-review-draft
```

Replace the example additional-report name with an actual finalized report, or
omit that argument when the default allowlist covers every linked report. The
builder must not be run in build mode merely to test the current incomplete V2.
