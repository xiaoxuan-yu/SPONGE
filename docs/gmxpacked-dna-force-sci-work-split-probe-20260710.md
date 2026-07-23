# gmxpacked DNA force SCI work-split probe - 2026-07-10

This note records the DNA-first force-kernel experiment that followed
`gmxpacked-force-kernel-handoff-20260710.md`.

## Production target correction

The three production systems do not consume the same template instance.
Current-peak NSYS and the dispatch gate show:

| system | LJ parameter path | main force instance average | tuning role |
|---|---|---:|---|
| dna_cou | packed triangular AB table | 146.453 us | primary target |
| wat160k | geometric per-atom LJ combination | 245.290 us | guardrail |
| wat600k | geometric per-atom LJ combination | 779.630 us | guardrail |

DNA prints the incompatible-combination fallback notice and instantiates
`use_lj_comb=false`. Both water systems instantiate `use_lj_comb=true`.
Therefore the actual DNA production replay payload is:

```text
/tmp/sponge-force-table-microbench-20260709/dna_cou/table_payload.sponge_gmxpacked_forceonly.bin
```

The larger comb-format DNA snapshot under
`/tmp/sponge-collect-distribution-20260708/dna_peak_padding/` is useful for
diagnostics, but it is not the parameter path consumed by DNA production.

## Baseline diagnosis

Fresh full NCU on the DNA AB-table replay measured:

| metric | value |
|---|---:|
| duration | 162.05 us |
| SM / memory / DRAM throughput | 16.81% / 10.84% / 1.57% |
| registers/thread | 68 |
| theoretical / achieved occupancy | 58.33% / 15.92% |
| eligible warps/scheduler | 0.37 |
| grid | 996 CTAs, 0.56 waves/SM |
| L1 / L2 hit rate | 61.84% / 87.94% |

The kernel was latency-bound. The dominant normalized sampled stalls were
wait 32.3%, long scoreboard 28.2%, and short scoreboard 22.3%. There were no
spills. The immediate problem was insufficient independent CTA work, not DRAM
bandwidth.

## Rejected AB matrix probe

The existing default-off row-major AB matrix gate reduced NCU instructions by
2.73% and NCU duration by 0.8%, but did not survive production measurement:

| measurement | packed table | row-major matrix | delta |
|---|---:|---:|---:|
| six-run DNA force mean | 3.631423 s | 3.614687 s | -0.46% |
| six-run DNA speed mean | 330.810521 ns/day | 332.044352 ns/day | +0.37% |
| paired NSYS safe-kernel average | 146.209 us | 146.433 us | +0.15% |

This is below run-to-run noise. The matrix gate remains default-off.

## SCI work partition experiment

One production CTA originally consumed all `cjpacked` records for one SCI.
Force output is already atomic, so the replay microbench partitioned the record
loop across multiple CTAs using interleaved record indices.

| parts/SCI | normal time | full-NCU time | tolerance result | decision |
|---:|---:|---:|---|---|
| 1 | 0.139090 ms | 162.05 us | reference | baseline |
| 2 | 0.079722 ms | 92.83 us | 0 mismatches; max scaled 8.57e-6 | accept |
| 3 | 0.064293 ms | 75.74 us | 2 mismatches; max scaled 1.166e-5 | reject |
| 4 | 0.059292 ms | 69.02 us | intermittent mismatch; max scaled 1.075e-5 | reject |

The validation threshold is `1e-5 * (1 + abs(reference))`. Parts 3 and 4 are
faster but are not production candidates under the existing force tolerance.

## Production gate

The accepted implementation adds a trailing compile-time `sci_work_parts`
parameter to the force kernel. Its default is one, so existing specializations
retain the original mapping. The default-off gate is:

```text
SPONGE_CLUSTERED_GMXPACKED_FORCE_SCI_SPLIT2_PROBE=1
```

It only activates when all of the following hold:

- force-only atom-order `VECTOR` output;
- AB-table parameters (`use_lj_comb=false`);
- fast full-local dense layout;
- existing SCI-shift safe/unsafe split is active;
- no compact/float4 force target, AB matrix, component-atomic, or skipped
  writeback probe is active.

Safe and unsafe SCI launches both use two CTAs per SCI. Energy and virial calls
remain on the original one-CTA specialization.

## Production results

Full production NCU on the safe force instance:

| metric | packed baseline | split2 | delta |
|---|---:|---:|---:|
| duration | 180.06 us | 104.26 us | -42.1% |
| SM throughput | 15.6% | 28.2% | +12.6 pp |
| memory throughput | 13.4% | 24.9% | +11.5 pp |
| achieved occupancy | 16.1% | 30.6% | +14.5 pp |
| eligible warps/cycle | 0.3 | 0.7 | about 2x |
| registers/thread | 69 | 69 | unchanged |

The final rebuild NSYS repeat measured the safe instance at 146.688 -> 84.366
us and the unsafe instance at 26.514 -> 15.874 us. Their combined GPU time
fell from 1.732 s to 1.002 s across 10000 steps. An earlier paired capture
measured 146.660 -> 84.733 us and 27.561 -> 13.410 us; both captures agree on
the work-split direction and magnitude.

Four-run AB/BA 10000-step comparisons:

| comparison | Calculate_Force | speed |
|---|---:|---:|
| packed gmxpacked | 3.768978 +/- 0.041096 s | 311.293877 +/- 2.757110 ns/day |
| split2 gmxpacked | 3.035348 +/- 0.026780 s | 361.047897 +/- 1.656741 ns/day |
| current mainline baseline | 3.520570 +/- 0.014745 s | 320.486397 +/- 0.820629 ns/day |
| split2 paired with mainline | 3.029613 +/- 0.021473 s | 361.680825 +/- 2.160516 ns/day |

Against the mainline binary, split2 improves 10000-step speed by 12.85% and
reduces `Calculate_Force` by 13.95%.

A fresh three-pair final-binary repeat, ordered both baseline->split2 and
split2->baseline, measured `Calculate_Force` at 3.758465 -> 3.092816 s
(-17.71%) and speed at 311.874980 -> 352.038361 ns/day (+12.88%). Population
standard deviations were 0.021186/0.030246 s and 1.661137/4.443837 ns/day.
All six final states were finite (298.52-301.42 K).

Two-run AB/BA 50000-step comparisons:

| path | Calculate_Force | speed |
|---|---:|---:|
| mainline baseline | 17.495117 s | 324.400711 ns/day |
| split2 | 15.236982 s | 360.568177 ns/day |

The long-run speed gain is 11.15%, with a 12.91% force-time reduction. All
paired final states were finite. One preliminary unpaired split2 50000-step
run measured 262.61 ns/day; it is retained as an outlier artifact and was not
included in the predeclared AB/BA pair means.

## Water guardrails

With the split2 flag requested, both water systems reported `lj_comb=1` and
`sci_work_split2=0` in the production gate trace.

| system | 10000-step speed | final temperature |
|---|---:|---:|
| wat160k | 151.682510 ns/day | 294.49 K |
| wat600k | 49.056854 ns/day | 294.22 K |

The water LJ-combination fast path is unchanged.

## Decision

Use split2 in the DNA-specific peak environment. Keep it default-off outside
that environment and do not add it to the shared water peak environment.
Keep split3/split4 as microbench diagnostics only. The next DNA force-kernel
iteration should start from split2 and target the remaining latency stalls,
not revisit the rejected AB matrix or relax the force tolerance.

Artifacts:

```text
/tmp/sponge-dna-sci-split2-production-20260710
.tmp/dna-sci-split2-nsys-20260710
.tmp/dna-sci-split2-production-20260710
ncu_reports/force_dna_ab_20260710/dna_ab_production_sci_split2_dedicated_full.ncu-rep
ncu_reports/force_dna_ab_20260710/dna_ab_production_sci_split2_dedicated_diff.md
ncu_reports/force_dna_ab_20260710/dna_ab_sci_split{2,3,4}_full.ncu-rep
```
