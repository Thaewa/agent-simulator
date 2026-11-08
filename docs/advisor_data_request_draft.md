# Advisor follow-up draft

Draft only. Not sent. Confirm the wording and any data-sharing constraints before sending.

Subject: Wasp simulator: TSP fix and inputs for the revised comparison

Hi Nitika,

I checked the case mismatch you flagged and corrected the mode selection. Tests and instrumented runs now confirm that the TSP modes reach the planner and execute planned moves. This confirms the implementation fix; I have not yet isolated how much that bug affected the historical figure.

I am re-anchoring the next phase around your suggestion: RL as an optimizer comparator on real nest maps, evaluated against field statistics, rather than treating TSP-like behavior as the biological target.

I found a local 27-page copy titled *Heuristic Route Optimization in a Social Wasp Reflects Decentralized Solutions to the Traveling Salesman Problem*. Could you confirm whether this is an older version and share the revised manuscript, or the relevant updated methods and results?

To set up the comparison, could you also point me to:

- Available nest/cell maps, coordinate scales and larval information.
- Feeding and food-transfer observations, including existing bout labels, arrival records and observation windows where available.
- The field statistics and definition of efficiency you would prioritize, and any conditions for using the data in this project and a future publication.

The simulator now records individual successful food events. I also found that its legacy bout counter uses food thresholds, so I have explicitly labeled it as a proxy rather than equating it with the manuscript's biological feeding bouts. I would like to confirm the observation protocol before implementing that mapping.

Best,
Thaewa
