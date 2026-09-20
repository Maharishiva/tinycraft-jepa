# V1: frozen existing evidence

This is the separate rewritten manuscript. It includes existing results only; follow-up experiments belong in V2.

- Build: `bash build.sh` (Tectonic). Figures are in `../shared/figures/`.
- Main text: 9 pages; references start on page 10; complete PDF: 21 pages.
- The ICLR 2027 `.sty` and `.bst` match the official files byte-for-byte. Review line numbers, normal font sizes, and page geometry are preserved. Color is confined to figures and pale table row backgrounds.
- Main storyline: Craftax task split; controlled gate retention/prediction/control; matched three-seed nuisance interaction; scoped population limits.
- The residual-quality diagnostic uses 8 NFE, while the policy runs use 4. Head-only and residual-only holdouts are distinguished from independent model evaluation data. Model fits and policy seeds are reported separately.
- Exact empirical SIGReg scaling, coordinate-mean MSE, and full proofs appear in the appendix. The Gaussian nuisance construction is attributed to prior work; the small-amplitude construction is stated as a narrow population boundary.
- `qa/checks.json` records PDF/style hashes and structural checks. Page renders are QA material, not submission artifacts.

The manuscript does not claim a causal dynamics-only control deficit, matched flow superiority, long-horizon Craftax success, or a guaranteed acceptance score. Those require evidence beyond the existing results.
