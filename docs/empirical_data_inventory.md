# Local empirical-data inventory

Inspection date: 2026-11-08. Read-only source inspection; no simulator changes or empirical data import.

## Scope and limitations

Inspected the project parent directory `C:/Users/Lenovo/Documents/capstone_project`, including `manuscripts`, `Final Project Report`, sibling simulator copies and `analysis_outputs - backup`. Listed relevant Downloads filenames. The recursive data-file scan excluded Git, environments, dependency/cache directories and `exports` (which contains prior generated/recovered artifacts). This is not a full-computer search.

The scan found 178 CSV files outside those exclusions and no XLS/XLSX/JSON/R/RDS/RData/MAT/TSV files. Paths and representative headers identify simulation logs, aggregates and a nest-log statistical summary; they are not verified field observations. Every CSV row was not independently audited. Downloads includes `Extended_References_FeedingDynamics.xlsx`; its contents have not been inspected, so it is not classified as field data.

## Important manuscript discovery

Source: `C:/Users/Lenovo/Documents/capstone_project/manuscripts/[32] .pdf`

- 27 pages; title: *Heuristic Route Optimization in a Social Wasp Reflects Decentralized Solutions to the Traveling Salesman Problem*.
- Authors: Nitika Sharma and Raghavendra Gadagkar.
- SHA-256: `a0593b7bf34469799b0e76e68864042eeba1a08db6f1b1b955532bf6af30abea`.
- This is a biological manuscript, distinct from the capstone report and local capstone drafting files. Its optimization framing differs from the advisor's 2026-11-05 description of the revised argument. Its revision/date relationship to that email is unverified; do not label it the latest manuscript.
- Page 16 says data are available on request. No usable raw nest-coordinate/feeding-observation dataset was identified in the inspected locations.

The local `Manuscripts Reinforcement Learning for Larval Feeding Routes*.docx` filenames are additional document candidates, not evidence of a current biological manuscript. Their contents have not been inspected in this step. Nest photographs and manuscript figures are not substitutes for calibrated coordinates or event tables.

## Methods recoverable from the local manuscript

These describe this local version only; they are not assumed to be the revised protocol.

| Pages | Method or observation unit | Consequence for the simulator comparison |
| --- | --- | --- |
| 14 | 36 complete feeding bouts across three colonies; individually identified wasps; timed and located adult/adult and adult/larval exchanges | Need colony, individual, time, location and recipient identities; generic per-step counts alone do not reproduce these observations. |
| 14 | A bout spans forager arrival through exhaustion of brought food; overlapping arrivals extend the bout until the food is exhausted | A successful adult transfer is an event within a bout, not a complete biological bout. Current transfer-event objects and bout tracking need a separate definition audit before field comparison. |
| 14 | Division of labour and inequality of unloading/receiving food | Larval-feed inequality and food-receiving inequality are different endpoints. Do not substitute one for the other. |
| 15 | Larval feeding by developmental stage and distance from nest centre; feeding-area overlap and within-bout cell sharing | Need cell coordinates/stages, event identities and bout membership. Modeled hunger is not an observed feeding frequency. |
| 16 | Route analysis preserves a forager-alighting origin and the observed larval targets; distance normalized by number of larvae fed; compares observed routes with permuted, shortest-route and greedy alternatives | This conditions on observed targets and origin. It differs from comparing whole simulated colonies with different target selection and food supply. Define the intended comparison before implementing a metric. |

The local manuscript provides no verified conversion from simulation steps/grid coordinates/hunger to physical time/distance/field demand. The current config audit remains a model-behavior audit, not empirical calibration.

## Other inspected candidates

| Location | Finding | Classification |
| --- | --- | --- |
| `manuscripts/Graphs.zip` | Three PNG charts, including hunger plots | Simulation visualization artifacts, not raw field observations |
| `manuscripts/[14] .zip` | 22 PDF book chapters | Reference literature, not a coordinate/event dataset |
| `analysis_outputs - backup/3_nest_summary_stats.csv` | Descriptive statistics over 1000 nest-log rows with simulated role, hunger and event columns | Simulation summary; not empirical validation data |
| `manuscripts/Photos` | Nest/wasp/larva images | Image assets; calibration, cell labels and provenance unverified |
| `Final Project Report` | Capstone reports and evaluation document | Historical model documentation |

## Concrete unresolved inputs for the advisor

1. Is `[32] .pdf` the older manuscript? Request the current version, or the current relevant methods/results if sharing the full manuscript is impractical.
2. Which real nests and observations can be used? Request cell IDs and coordinates, coordinate scale, larval occupancy/stage, and nest/colony IDs where available.
3. Are time-stamped food-transfer and larval-feeding records available, including adult/recipient IDs, bout membership, arrival origin and observation exposure? Ask what was actually measured rather than assuming quantities or hunger observations exist.
4. Which field endpoints and null comparisons should the RL optimizer and local policies be tested against, and how is efficiency defined in the revised study?
5. What access, attribution and publication-use conditions apply to shared data?

These are prepared questions; no email has been sent. Next implementation should follow the agreed data representation and endpoints. Do not manufacture empirical inputs from plots or tune synthetic hunger to match historical figures.
