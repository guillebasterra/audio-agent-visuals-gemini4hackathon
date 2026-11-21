# Agent Crowd — Experiment Report (Gemini 4 Neon)

- **Track:** `getuphereletsdance.mp3` | **Seed:** `42`
- **Analyst model:** `gemini-4-neon` (`thinking_level="high"`, latency `87705 ms`, tokens `28816`)

## Five Key Takeaways

### 1. Audio-only agents begin positioning earlier ahead of drops than the scripted heuristic, while acoustic filtering creates a steep front-to-back anticipation gradient where rear rings miss builds and drop cues heard clearly up front.
- **Numbers:** anticipation_mean_lead_s: 12.75 (silent), 12.0 (talk), 10.5 (shove); pre_15s_v_gain: 0.0144 (silent), 0.014 (talk), 0.0813 (shove); mean_v_quiet: 0.4782 (silent), 0.4728 (talk), 0.3986 (shove); Ring 1 drop_recall: 1.0 and Ring 2 build_anticipation_hit: 0.5556 vs Ring 4 build_anticipation_hit: 0.0 and Ring 5 drop_recall: 0.6429, build_anticipation_hit: 0.1111
- **Evidence quote:** A94 -> A95 (no): "Drop's hitting right now, mind if I squeeze past you?"

### 2. Replacing blind shoving with audio-guided decisions modestly lifts mean crowd welfare while substantially reducing inequality by raising dancer and between welfare and trimming diehard front-row extraction; adding natural-language negotiation does not improve welfare or fairness over silent minds.
- **Numbers:** mean_H: 0.3133 (silent), 0.3125 (talk), 0.3082 (shove); gini_H: 0.1737 (silent), 0.1783 (talk), 0.2212 (shove); by_archetype_mean_H (diehard/between/dancer): 0.406/0.3176/0.2149 (silent), 0.4081/0.3174/0.2102 (talk), 0.4374/0.2976/0.1931 (shove); drop_front_minus_back_H: 0.2843 (silent), 0.291 (talk), 0.3222 (shove); drop_front_H/drop_back_H: 0.4576/0.1734 (silent), 0.4597/0.1687 (talk), 0.4719/0.1498 (shove) 
- **Evidence quote:** A73 -> A76 (yes): "Mind if I slip past you? Just trying to find a little more room up ahead."

### 3. Negotiation in talk is largely honest with minimal keenness shading and zero extreme bluffs, while blockers reward past generosity with higher acceptance rates yet still enforce a high refusal baseline.
- **Numbers:** asks_total: 604 (asks_yes: 201, asks_no: 403, ask_acceptance_rate: 0.3328); mean_overstatement: 0.0329, max_overstatement: 0.1168, share_overstated_gt_0_1: 0.005, count_overstated_gt_0_3: 0, yes_rate_when_overstated_gt_0_1: 0.0 vs yes_rate_when_honest_within_0_1: 0.3344; reciprocity yes_rate: 0.4211 (mean_p: 0.4003, n: 57) vs no_prior_yield: 0.3236 (mean_p: 0.3202, n: 547) 
- **Evidence quote:** A22 -> A3 (no): "I let you through earlier, mind if I slip past for this groove?"

### 4. Audio-driven decision-making slashes crowd gridlock by more than half, multiplies forward progress per push, and preserves higher energy at drops and in the rear of the floor without relying on breakdown resting or triggering exhaustion.
- **Numbers:** gridlock_rate: 0.8496 (shove, 4281/5039 push ticks stalled, mean_push_dv: 0.00456) vs 0.3272 (silent, 1002/3062, mean_push_dv: 0.05586) and 0.3342 (talk, 1014/3034, mean_push_dv: 0.05516); mean_E_at_drop_onsets: 0.7852 (shove), 0.8015 (silent), 0.8068 (talk); drop_back_E: 0.5781 (shove), 0.6777 (silent), 0.682 (talk); breakdown_rest_share: 0.0161 (shove), 0.0 (silent), 0.0011 (talk); exhaustion_events: 0
- **Evidence quote:** A83 -> A91 (no): "Hey, I'm fresh and really trying to catch this groove up front—mind if I squeeze past?"

### 5. Intrinsic preferences (high keenness k and low space sensitivity s) explain roughly two-thirds of welfare across all conditions, with diehards dominating front-row drop occupancy and resisting yields; in talk, persuasive ask success independently raises welfare and removes the starting-lane advantage seen in silent.
- **Numbers:** OLS r_squared: 0.6302 (shove), 0.6724 (silent), 0.6752 (talk); k coef: 0.39284 (p=1e-05, shove), 0.23131 (p=0.00034, silent), 0.25529 (p=0.00019, talk); s coef: -0.2547 (p=0.00325, shove), -0.33207 (p=0.0, silent), -0.26021 (p=0.00013, talk); start_lane coef: 0.00598 (p=0.00259, silent) vs 0.00303 (p=0.13342, talk); ask_success_rate coef: 0.06665 (p=0.01563, talk); drop_top20_diehard_share: 0.6687 (shove), 0.5969 (silent), 0.6094 (talk); blocker yield_rate: 0.2771 (diehard, 433), 0.4562 (between, 160), 0.7273 (dancer, 11) 
- **Evidence quote:** A87 -> A83 (yes): "Yo, can I squeeze past? I'm desperate to get closer to the booth."

## Methodological Caveats

- Single track and seed: all three conditions ran once on getuphereletsdance.mp3 (duration_s: 339.1, n_ticks: 113, n_drops: 4) with seed 42 and n=100 agents, so narrow silent-vs-talk gaps such as mean_H (0.3133 vs 0.3125) and gini_H (0.1737 vs 0.1783) cannot be separated from single-seed variance or generalized across tracks.
- Probabilistic yield sampling and OLS endogeneity: blocker responses in talk are sampled stochastically from answer_p (mean_answer_p: 0.3278, realized ask_acceptance_rate: 0.3328), introducing realization noise on individual asks (for example, A22 -> A3 rejected at answer_p 0.422 while A84 -> A87 accepted at 0.223) and making the positive ask_success_rate OLS coefficient (0.06665, p=0.01563) sensitive to both draw luck and behavioral selection into asking.
- Small subgroup sizes and untested fatigue mechanics: the reciprocity lift rests on just 57 prior-yield asks against 547 non-prior-yield asks (0.4211 vs 0.3236), dancer asks (count: 2) and dancer blocks (count: 11) are tiny, overstatements above 0.1 represent only 0.005 of asks (with 0 above 0.3), and breakdown_rest_share is 0.0 in silent and 0.0011 in talk (with exhaustion_events: 0 across all conditions), leaving fatigue and rest trade-offs effectively unstressed.
- API reliability vs compute cost and reference-condition anchoring: although runs experienced 0 api_failures (api_failure_rate: 0.0), 0 rate_limit_errors, and 0 hydrogen_failures across 113 (shove), 651 (silent), and 1263 (talk, including 608 hydrogen_calls) API calls, talk incurred heavy latency and token overhead (tick_latency_p50_ms: 1184.7 vs 1042.8 silent and 833.1 shove; hydrogen_latency_p50_ms: 2772.2, p95: 4028.8; total_tokens: 6223275), and house_drop_onsets are anchored to talk as reference_house_condition (p_drop: 0.298 to 0.318).

## Stratified Sample of 15 Asks Sent to Neon

| t (s) | Pair | Archetypes | True k -> Claimed | Outcome (p) | Ask |
| ---: | --- | --- | --- | --- | --- |
| 24.1 | A83 → A91 | diehard → diehard | 0.71 → 0.78 (+0.07) | no (0.28) | "Hey, I'm fresh and really trying to catch this groove up front—mind if I squeeze past?" |
| 54.1 | A11 → A1 | diehard → diehard | 0.71 → 0.78 (+0.07) | no (0.44) | "Hey, I let you through earlier—mind letting me slide up as the groove builds?" |
| 69.1 | A94 → A95 | diehard → between | 0.87 → 0.90 (+0.03) | no (0.33) | "Drop's hitting right now, mind if I squeeze past you?" |
| 78.1 | A54 → A77 | dancer → between | 0.54 → 0.58 (+0.04) | no (0.51) | "Mind if I slip past you? Trying to find a better spot to get into this groove." |
| 81.1 | A84 → A87 | between → diehard | 0.56 → 0.65 (+0.09) | yes (0.22) | "Drop's here! Can I slip by you for a second?" |
| 102.1 | A59 → A95 | between → between | 0.58 → 0.60 (+0.02) | yes (0.39) | "Hey, mind if I slip past you? Just trying to get a bit closer to the booth." |
| 111.1 | A61 → A11 | diehard → diehard | 0.73 → 0.82 (+0.09) | yes (0.25) | "The drop is hitting right now, mind if I slip past you again?" |
| 159.1 | A87 → A83 | diehard → diehard | 0.80 → 0.85 (+0.05) | yes (0.35) | "Yo, can I squeeze past? I'm desperate to get closer to the booth." |
| 168.1 | A20 → A61 | between → diehard | 0.64 → 0.72 (+0.08) | no (0.31) | "Hey, mind if I slip past you again? I'm feeling this groove and trying to get closer." |
| 189.1 | A27 → A45 | diehard → diehard | 0.70 → 0.80 (+0.10) | yes (0.29) | "The drop is hitting right now, mind if I squeeze past you?" |
| 198.1 | A84 → A35 | between → diehard | 0.56 → 0.58 (+0.02) | no (0.26) | "Hey, mind if I slip past you during this groove? Just trying to get a bit closer." |
| 300.1 | A73 → A76 | dancer → between | 0.36 → 0.38 (+0.02) | yes (0.39) | "Mind if I slip past you? Just trying to find a little more room up ahead." |
| 327.1 | A99 → A47 | diehard → between | 0.83 → 0.85 (+0.02) | yes (0.31) | "Hey, this groove is infectious—mind if I slip past you toward the front?" |
| 327.1 | A72 → A12 | between → diehard | 0.61 → 0.65 (+0.04) | yes (0.25) | "Hey, mind if I squeeze past? Just want to get a little closer for this groove." |
| 333.1 | A22 → A3 | diehard → diehard | 0.85 → 0.85 (+0.00) | no (0.42) | "I let you through earlier, mind if I slip past for this groove?" |
