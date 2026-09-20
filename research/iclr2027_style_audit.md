# Official ICLR 2027 style audit

Verified against the public [ICLR 2027 style archive](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip), downloaded 19 September 2026. Canonical extracted files are retained in [formatting/iclr2027/](formatting/iclr2027/).

## Finding and action

The pre-existing repository style was not byte-identical to the official style. Its entire anonymous-review line-number ruler block was commented out. The bibliography style differed only in its opening filename comment. No margin or text-area differences were found in the style-file diff. The manuscript author replaced both submission copies with the official files. A subsequent byte-for-byte check confirmed both now match the official hashes below; see [the current hash check](formatting/style_hash_check.json). The original repository was not edited by this workstream.

| File | Official SHA-256 | Initial submission copy SHA-256 | Difference |
|---|---|---|---|
| `iclr2027_conference.sty` | `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6` | `96b73285a1f9595bbd4308b804bb582b6ce99fb29e88c46073847bf131dbdbfb` | Review line-number ruler disabled |
| `iclr2027_conference.bst` | `2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5` | `979801f5dd1175bba2f3e2f088b74a669db8f290d97d72fa2817a33f7cb8f825` | Opening comment renamed |

## Authority for colored tables and visual changes

The official template explicitly allows color figures at `iclr2027_conference.tex:191` and asks for meaning to survive grayscale printing at lines 192–194. Its table requirements (lines 203–211) are centering, neatness, legibility, consecutive numbers, and captions before the table. It does not explicitly mention colored table cells. Thus pale table tints with readable dark text are a reasonable interpretation of the legibility requirements, not a separately stated permission. Actual accepted ICLR 2026 papers in the reading archive use colored cells, which demonstrates practice but does not override the 2027 rules.

Preserve official geometry, typography, and page numbering. The template's final instructions (lines 352–357) prohibit changes to formatting parameters and font sizes except perhaps references. Initial manuscript uses of `\small` in the global table-input macro and one appendix table were flagged conservatively; author should keep default-size table text unless a compliant reason is established. No `geometry`, negative spacing, line-spacing, width/height, resizebox, or scalebox overrides were found in the inspected submission `.tex` files at audit time. Recheck after edits.

## Submission facts

- Initial main-text limit is 9 pages; rebuttal/camera-ready limit is 10 (`iclr2027_conference.tex:131`). The author-guide FAQ contains stale wording; the explicit rule and template agree on 9.
- AI-use statement is required, outside the page limit, at most one page (`iclr2027_conference.tex:397–412`). It must describe actual substantive research assistance, including methodology, implementation, processing, interpretation, mathematics when applicable, as specified in the [AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors). Do not claim human verification that has not happened.
- Reproducibility and ethics statements are recommended and outside the main-text limit (`iclr2027_conference.tex:416`, `:430`).
- Keep review mode enabled, author identity absent, and page numbers/ruler intact. HTML is a companion, not a substitute for the official PDF format.

This audit checks source/template fidelity. Final page count, legibility, overflow, and anonymity require rendering and inspection of the final compiled artifact.
