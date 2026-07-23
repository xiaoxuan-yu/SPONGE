# gmxpacked DNA NPT virial SCI split2 - 2026-07-23

This experiment follows the same-ensemble public baseline correction in
`gmxpacked-public-baseline-ensemble-benchmark-20260723.md`. NVE remains locked;
the new gate only targets DNA's NPT virial direct-force specialization.

## Baseline profile

Full NCU on the DNA AB-table virial specialization:

| metric | value |
|---|---:|
| duration | 245.57 us |
| grid | 996 CTAs, 0.56 waves/SM |
| SM / memory / DRAM throughput | 20.34% / 37.93% / 5.10% |
| theoretical / achieved occupancy | 58.33% / 17.35% |
| eligible warps/scheduler | 0.30 |
| registers/thread | 72 |
| local spill requests | 2,559,530 |
| long / wait / short scoreboard | 33.0% / 27.6% / 24.1% |

The native CUDA kernel was latency-bound. The grid was too small to expose the
theoretical occupancy, while the per-thread virial state also spilled.

## Change

Default-off gate:

```text
SPONGE_CLUSTERED_GMXPACKED_VIRIAL_SCI_SPLIT2_PROBE=1
```

It requires all of:

- `need_virial && !need_energy`;
- AB-table parameters, not the water LJ-combination path;
- full-local-dense fast layout and SCI-shift split;
- compact force storage;
- no AB-matrix, float4, component-atomic, or skipped-writeback probe.

Each SCI's packed-record interval is interleaved across two CTAs. Force and
per-atom virial outputs are already atomic. The existing force-only split2
gate and all NVE specializations are unchanged.

## NCU verification

Mandatory full-NCU diff:

| metric | one CTA/SCI | two CTAs/SCI | delta |
|---|---:|---:|---:|
| duration | 245.57 us | 192.90 us | -21.4% |
| grid | 996 | 1992 | 2x |
| waves/SM | 0.56 | 1.11 | +0.55 |
| SM throughput | 20.34% | 29.37% | +9.03 pp |
| memory throughput | 37.93% | 64.82% | +26.89 pp |
| achieved occupancy | 17.35% | 37.10% | +19.75 pp |
| registers/thread | 72 | 72 | unchanged |
| L2 hit rate | 94.55% | 98.49% | +3.94 pp |

The kernel speedup is 1.27x. The remaining bottleneck changes to L2/local
memory latency: long scoreboard rises to about 55%, and the virial accumulator
still spills. Do not add more SCI partitions before addressing that state.

## Correctness and end-to-end

At 20 steps, current and split2 have identical temperature, potential energy,
and density at printed precision. Pressure differs by 0.04 bar and individual
energy components differ only by normal atomic-order rounding.

Three paired 10000-step DNA NPT runs:

| path | Calculate_Force (s) | wall (s) | speed (ns/day) |
|---|---:|---:|---:|
| current | 3.104260 +/- 0.003748 | 4.828815 +/- 0.007985 | 357.888224 +/- 0.591879 |
| virial split2 | 3.059537 +/- 0.007744 | 4.785828 +/- 0.008668 | 361.102976 +/- 0.653380 |

Paired speed gain is `+0.8984% +/- 0.2492%`; force time falls 1.44%.

Final same-ensemble public acceptance, three paired 10000-step runs:

| path | Calculate_Force (s) | wall (s) | speed (ns/day) |
|---|---:|---:|---:|
| public SPONGE 2.0 | 3.508105 +/- 0.006500 | 6.011970 +/- 0.017534 | 287.456940 +/- 0.837088 |
| virial split2 | 3.058080 +/- 0.032462 | 4.794256 +/- 0.039616 | 360.483877 +/- 2.989289 |

The paired candidate speed advantage over public is
`+25.4044% +/- 0.9534%`.

Both water NPT guardrails retain `use_lj_comb=true` and `sci_work_parts=1`.
Single 10000-step current/candidate checks were finite; their small negative
differences were consistent with order noise because the device specialization
is identical.

## Decision

Accept the gate for the DNA-specific NPT benchmark environment. Keep it
default-off, do not add it to NVE/NVT, and do not add it to either water path.
The next NPT kernel experiment should reduce the spilled eight-entry
`LTMatrix3` virial accumulator or otherwise improve its local-memory access;
it must begin from a new full NCU profile of this split2 specialization.

Artifacts:

```text
ncu_reports/gmxpacked_ensemble_20260723/dna_cou_virial_full.ncu-rep
ncu_reports/gmxpacked_virial_split2_20260723/dna_cou_virial_full.ncu-rep
ncu_reports/gmxpacked_virial_split2_20260723/dna_cou_virial_full_diff.md
.tmp/dna-virial-split2-ab10000-20260723/results.tsv
.tmp/dna-virial-split2-public-ab10000-20260723/results.tsv
.tmp/water-virial-split2-guardrail10000-20260723/results.tsv
```
