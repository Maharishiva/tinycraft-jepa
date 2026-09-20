# Assessment after the frozen hard-goal control block

All three prospectively admitted `gray_frame_still` model seeds completed the
unchanged protocol. Exact goal hits out of the same 128 cases were **5, 5, 9** for
three-step JEPA MPC, **0, 0, 0** for one-step JEPA, **3** for paired random actions,
**0** for the fixed cycle, and **128** for the privileged position-cost oracle.
These are not a successful positive control. All outcomes, including the three
predeclared illustrative episodes, are retained.

The task asks for goals 6–12 actions away with a three-action search. Therefore
success requires the latent squared-distance objective to supply useful local
direction before the goal is reachable within the search. Merely encoding
position does not imply that metric property. Even a lossless spatial image
representation can have flat squared distance between two non-overlapping cube
locations. The oracle changes the cost to physical squared distance and thus
does not isolate transition-model error. These results concern this complete
frozen latent-distance control recipe; they do not prove that JEPA cannot plan,
that position is missing, or that stochasticity caused the failure.

## Conventional longer-horizon CEM: feasible, but not a clean diagnosis

Existing `plan.py:87` implements generic categorical CEM with default population
512, six iterations, 32 elites, temperature 1, and `log(counts + 0.5)` categorical
refitting. It uses terminal latent MSE. A separately labeled H12 variant could
use these conventional proposal settings, a fresh fixed planner key, and the
same summed latent stage cost as the frozen H3 evaluation. It would use no true
coordinates, goal-directed proposal policy, latent amplification, reward heads,
or special environment features. It would change both horizon and search type.

Measured H3 seed-0 steady-state work was 2.31–2.41 seconds per 16 cases × 24
decisions, after a 30.35-second first chunk including compilation. That chunk
evaluates 16 × 24 × 64 × 3 = 73,728 predictor transitions. H12 with 512 candidates
and six iterations evaluates 192 times as many transitions per decision.
Linear extrapolation gives approximately **60 GPU minutes per 128-case model**,
or **15 minutes per 32-case model**, plus compilation and sampling overhead.
Actual batching efficiency is unmeasured; this is a feasibility estimate, not a
benchmark. A three-model block on the first 32 fixed existing cases would be
roughly 45–60 GPU minutes. The alternative 256-candidate/four-iteration package
would be about 20 minutes per 128-case model, but is less generous than the
existing conventional defaults.

Independent review found that longer CEM would be a useful engineering
robustness check, but a negative result would remain difficult to interpret.
For a purely axial goal 12 actions away, one uniform 12-action sequence reaches
the exact goal with probability 4^-12; 512 initial candidates almost always
miss. If the latent cost is locally flat until overlap, elite refitting need not
receive a useful directional signal even with exact dynamics. Longer horizon
alone therefore does not certify competent search under an arbitrary latent
goal metric. No CEM follow-up has been launched.

## Recommended single bounded follow-up: fixed distance-response curve

Treat this explicitly as a **post-outcome characterization of operating range**.
Retain the failed hard-goal primary unchanged. Use the same 128 fresh valid
prefixes and all three frozen models, the same H3 controller and 24-action
budget, and exact goal-distance shells **1, 2, 3, 6, 9, 12 actions**. Before any
follow-up run, fix a new goal-sampling seed and uniformly draw one safe reachable
goal per start and shell, independently of model outputs. Keep the same
episode-specific goal background across the six goals, so only the cube
position changes. Reuse execution noise across all shells/models. Record
directional offsets because the safe border truncates the available directions.

Retain random, fixed-cycle, and privileged oracle references at every distance.
Report exact hits/128 separately for every model × distance, with per-cell Wilson
intervals and paired-case records. Do not pool the 768 correlated cases into an
IID interval. Show all six distances and all three seeds; never select the best
distance as a replacement primary. The ≤3-pixel endpoint is already satisfied
at the initial state for one-action goals and must not headline that stratum.

Using one process per checkpoint and reusing the fixed batch-shape compiled
planner across all six shells should take approximately **12–15 GPU minutes for
all three models**, including references and compilation, by the measured H3
throughput. A conservative slot would be 20 minutes, with explicit runtime
measurement and no outcome-dependent stopping. This estimate assumes shared
compilation rather than six fresh Python processes per checkpoint.

This follow-up can reveal whether the same controller has a short operating
range. It cannot turn nearby empty-arena navigation into strategic planning,
prove horizon causality, or rescue the original hard-goal outcome. Independent
review recommended this bounded curve over CEM for clearer interpretation and
lower cost. It remains a proposal; the parent must schedule any GPU execution.
