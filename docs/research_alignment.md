# Research alignment before RL

Working assessment: 2026-11-08. This is a planning record, not an approved research protocol or a new empirical result.

## Sources and precedence

- Advisor email supplied by Thaewa, dated 2026-11-05: the revised manuscript argues that wasps achieve efficiency without route optimization; TSP-style agents should not be the target. The suggested direction is RL as an optimizer comparator on real nest maps, tested against field statistics. The email also asks whether the TSP modes actually ran given the case mismatch.
- Local capstone report: `C:/Users/Lenovo/Downloads/capstone-report.pdf`, 11 pages. This documents the previous model and proposed extensions. It is not a verified copy of the revised biological manuscript.
- Report reference [32] names an unpublished 2025 Sharma and Gadagkar manuscript. Its title alone cannot establish the revised manuscript's conclusions or metric definitions.

Where the report and email disagree, use the email to guide the new project framing. Do not infer the missing manuscript's methods or numerical findings.

## Advisor requests versus present evidence

| Email item | Present status | Remaining work |
| --- | --- | --- |
| Check case mismatch and actual TSP dispatch | Corrected canonical mode selection, planner/route tests and actual solver execution are documented in bugfix_progress.md. | Report the correction and evidence accurately; this does not establish the size of its effect on historical figures. |
| Re-anchor around efficiency without route optimization | Direction is recorded here; the report's older optimization framing remains historical. | Agree on the operational meaning of efficiency and the biological comparison question. |
| Use RL as an optimizer comparator | Proposed role, not implemented. | Define observations, actions, objective and evaluation after identifying available field measurements. RL is not a presumed model of wasp cognition. |
| Use real nest maps and test against field statistics | The current experiment runner generates synthetic instances. No empirical map/statistics integration is established by the present audit. | Locate usable data, specify units and definitions, implement mapping, and validate the comparison. |

Controlled buggy-versus-fixed historical replay is an additional investigation proposed during this collaboration, not an explicit requirement in the email. It is deferred. Baseline commit `f64d70c9a02782ab2840e74708b0f4a7f624b0d2` is a reproduction candidate, not proof that every report figure was generated from that commit.

## What can be reused from the report

| Report section | Useful starting point | Restriction |
| --- | --- | --- |
| Methods, pp. 3-5 | Agent roles, food exchange, hunger, local movement and five hand-crafted policies | Compare each retained assumption with corrected code and biological measurements; the report is not a current executable specification. |
| Results, pp. 5-8 | Existing hunger, spatial and role-allocation figure types | Historical algorithm rankings and explanations are not validated results of the corrected simulator. |
| Limitations, pp. 8-9 | Simplified geometry, stylized rules and incomplete calibration | These limitations remain relevant; do not treat synthetic geometry as a real nest map. |
| Future Work, p. 9 | Real nest maps, empirical observations and learning-based comparators | This supports preparation, but supplies neither the actual dataset nor its statistical definitions. |

## Provisional comparison question

On the same real nest geometry, under comparable resource and observation constraints, how do measured wasp feeding patterns and non-optimizing local policies compare with an RL policy trained to optimize a specified feeding objective?

This is our proposed formulation, pending confirmation. It does not assume that real wasps optimize routes, that an RL advantage implies wasps are inefficient, or that matching one simulated pattern identifies a biological mechanism. Comparator information access and constraints must be stated explicitly.

## Measurement bridge to establish

| Candidate quantity | Current simulator support | Empirical definition still needed |
| --- | --- | --- |
| Feeding activity | Successful feeding event counts and rates | What counts as a feed, observation duration, and which adults/larvae were observed |
| Travel cost | Euclidean travel in grid-coordinate units per successful event | Physical coordinate scale, observed trajectory/route definition and missing observations |
| Spatial feeding distribution | Larval positions and hunger/event logs | Nest/cell coordinates, spatial reference and measured outcome per cell |
| Adult participation inequality | Per-adult feeding counters can support analysis | Identity tracking, zero-event individuals, observation exposure and chosen inequality statistic |
| Larval hunger | Internal modeled state | Whether a comparable field measurement exists; otherwise retain as a simulation diagnostic |

These are candidates drawn from the report and current logging, not verified statistics from the revised manuscript. See metrics.md for current units and formulas. The final low-hunger fraction is not evidence of route or energetic efficiency. Calibration must not aim merely to reproduce the old graph shapes.

## Next work, in order

1. Inventory locally available nest maps, cell coordinates, feeding observations and supporting metadata. Mark synthetic examples separately. Do not assume any particular field measurement exists.
2. Prepare a concise list of unresolved inputs for the advisor: revised manuscript or relevant methods/results, available maps/statistics, coordinate/time units, observation scope, data access/use conditions and what efficiency means in this study. No message has been sent.
3. Using the available evidence, draft a comparison protocol specifying hypotheses, map representation, empirical endpoints, calibration/evaluation separation, comparator constraints and replication. Identify unsupported parameters explicitly.
4. Implement and validate empirical data integration and the agreed baseline comparisons before RL training. Select replication and uncertainty reporting appropriate to the available nests and observations.

Until these inputs are established, generated-nest runs are software/model diagnostics. Passing tests and completing plots do not establish biological validity or algorithm superiority. This record does not authorize changing parameters, sending email, or claiming advisor approval.
