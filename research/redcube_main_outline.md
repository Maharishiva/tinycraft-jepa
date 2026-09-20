# Redcube main-paper rewrite

User-directed rewrite started 2026-09-19 23:09 UTC. The preceding audit-centered V2 and its exact figures are preserved under `submission/archive/20260919T230951Z-audit-v2/`, with file hashes. V1 is untouched.

## Main claim architecture

The paper is about two requirements of an abstract world model: preserve the state needed for decisions, and preserve the uncertainty needed to evaluate those decisions. Three minimal redcube cases make these requirements concrete.

1. **A model that learns and plans.** Primary setup is gray per-frame background with deterministic dynamics, using the native 512-dimensional grid JEPA. Existing three-seed position readout is 0.115±0.012 pixel L1. This alone is not planning; the new fixed visual-goal MPC test supplies actual trajectories and goal hit/distance outcomes. Admit its result only from final validated artifacts. Goals are 6–12 native actions away; 24 real steps; fixed histories and blind/random/privileged controls.
2. **A learned representation that fails to expose useful state.** Gray per-frame noise plus stochastic wind raises the same three-seed position readout to 21.33±0.06 pixels. Persistent gray noise with wind gives 0.668±0.049 pixels. The main visible evidence will be true observations, decoded true encodings, and a compact behavior comparison. The empirical failure has high prediction loss (~0.733), so it is not claimed to instantiate the static-nuisance zero-loss optimum or prove information-theoretic deletion.
3. **Usable state with a difficult stochastic future.** Candidate environments are persistent-gray wind and the late-reveal gate. A fair same-CNN learned-model comparison and real-decoded gate controls are running. Until they qualify the interface and show a replicated contrast, there is no empirical claim that flow succeeds where an otherwise sufficient JEPA fails. Independently, the exact oracle-variance audit tests loss of expected latent-goal cost under point compression in a fixed informative encoder.

## Analytical spine, written before empirical outcomes

- The user's exact A: ±1, B:0.1, goal0 example: mean-goal costs0/.01 choose A; true expected costs1/.01 choose B. The missing term is action-dependent variance.
- The user's exact recursion X'=X²+ξ, iid ξ=±1, X0=0: true means0,1,2 versus recursive conditional-mean predictions0,0,0. Finite-support calculation; no Gaussian substitution.
- Known Gaussian episode-nuisance trap: a recoverable constant nuisance can yield a noncollapsed zero-loss representation that carries no controlled state. Attribute the mechanism to Sobal2022. Keep existing exact Gaussian small-amplitude and other proof details in appendix.
- These establish failures possible under perfect point prediction or population objectives. They do not by themselves diagnose a particular trained model or require every useful stochastic controller to sample.

## Nine-page target

1. Motivation and the two requirements, with the three-case roadmap (~1 page).
2. Exact stochastic-planning mechanisms, two-panel analytical figure, static-nuisance proposition (~1.5 pages).
3. Minimal environments, native model dimensions, data and comparison contract (~0.75 page).
4. Case1: actual multigoal trajectories and concise control table (~1 page).
5. Case2: observation reconstruction plus matched nuisance×dynamics contrast (~1 page).
6. Case3: actual matched JEPA/flow trajectories/control and exact variance audit (~1.5 pages).
7. Related work, measured boundaries, reproducibility and AI statements (~1 page).

No nearby Craftax goal results in the main paper. A Craftax secondary result is eligible only if the fair inventory-acquisition experiment completes; the primary outcome is wooden-pickaxe acquisition with all achievements retained. No existing unpaired rate is promoted.

## Appendix and artifact only

Keep all completed MLP/native-grid/termination negative controls, forecast readout and conditional-constant audits, exact headers and provenance, earlier adverse flow conditions, historical goal results, and full proofs. They bound interpretation and support reproducibility; they are not the paper's main narrative.

## Pending evidence labels

Draft source uses explicit internal `\pendingresult{...}` blocks for the new visual-goal outcomes, same-CNN model/control comparison, and exact-variance table. These are not empirical results. No pending claim appears in the abstract. Remove or resolve every block before the final manuscript is frozen.

## Completed goal-control correction

The unchanged three-model H3 block is complete and independently audited: exact hits5/5/9 of128, horizon-one0/0/0, random3, fixedcycle0, privilegedreference128. H3 minimum-distance means22.24/22.95/22.03px are worse than random19.41px. Case1 now states readable state with weak goal-directed control under the declared distant-goal/short-horizon recipe. Any longer CEM or distance-response study is a separately declared follow-up; no positive outcome is assumed.

Current internal title: **What Does a JEPA World Model Preserve?** The established empirical novelty is the noise-persistence×wind reversal, with full Sobal2022 benchmark credit; the analytical mean examples explain mechanisms rather than assert novelty.

## Exact variance audit completed

Both frozen encoders pass the exact nine-branch calculation and identity/affine/deterministic controls. In the same843safecontexts×25goals, robust latent-cost action disagreements are2.65% (black+wind) and1.08% (persistentgray+wind). Mean latent regrets are7.50e-5 and3.80e-6 in checkpoint-specific coordinate-MSE units. Physical-cost differences have intervals containing zero. Main role is one short illustrative sentence; full numeric scope is appendix-only. Mean plus scalar variance trace suffices for this quadratic one-step objective, so the result does not prove generative modeling is necessary.

## Distance-response follow-up completed and independently admitted

The final audit covers all3models and passes8,179checks/176,334native transitions. The unchanged H3planner reaches all one/two-action-distance goals and121/123/124 of128 atdistance3; pairedrandom reaches19 andH1 reaches22/16/21. Atdistance6 H3 reaches25/32/34, atdistance9 reaches3/2/1, and atdistance12 none. The main case now honestly reports nearby goal-directed control and its operating range, using one all-model/all-shell curve. The failed original6–12goal primary remains explicit and separately tabulated. No global defect of goal geometry or long-horizon impossibility is inferred from this short-search diagnostic.
