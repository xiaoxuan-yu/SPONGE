# gmxpacked DNA energy+virial SCI split2 - 2026-07-23

This experiment continues DNA optimization after checkpoint `1e5561e`. The
water LJ-combination paths remain frozen. The change targets only the AB-table
kernel used when both per-atom energy and virial are requested.

## Baseline profile

Full NCU on the one-CTA-per-SCI energy+virial specialization:

| metric | value |
|---|---:|
| duration | 342.69 us |
| grid | 996 CTAs |
| SM / memory / DRAM throughput | 18.02% / 42.84% / 2.61% |
| achieved occupancy | 17.83% |
| eligible warps/scheduler | 0.31 |
| registers/thread | 72 |
| local-store spill sectors | 6,022,572 |
| L1 / L2 hit rate | 64.36% / 98.64% |
| long / wait / short scoreboard | 36.8% / 27.0% / 20.1% |

The one-CTA grid did not expose enough parallelism. The launch bound also
forced the large force, energy, and virial thread state into local memory.

## Change

Default-off gate:

```text
SPONGE_CLUSTERED_GMXPACKED_ENERGY_VIRIAL_SCI_SPLIT2_PROBE=1
```

It requires all of:

- `need_energy && need_virial`;
- AB-table parameters, not the water LJ-combination path;
- full-local-dense fast layout and SCI-shift split;
- compact force storage;
- no AB-matrix, float4, component-atomic, or skipped-writeback probe.

Each SCI's packed-record interval is interleaved across two CTAs. Force,
per-atom energy, and per-atom virial already use atomic output, so this does
not change their storage or downstream consumption. The split2 specialization
uses the existing virial launch bound of 10 blocks/SM instead of 13.

The energy+virial gate is independent from the existing virial-only gate. This
allows end-to-end A/B tests to keep the accepted virial-only optimization on
both sides.

## NCU verification

Mandatory full-NCU comparison:

| metric | one CTA/SCI | two CTAs/SCI | delta |
|---|---:|---:|---:|
| duration | 342.69 us | 193.34 us | -43.6% |
| grid | 996 | 1992 | 2x |
| SM throughput | 18.02% | 27.69% | +9.67 pp |
| memory throughput | 42.84% | 51.92% | +9.08 pp |
| achieved occupancy | 17.83% | 29.12% | +11.29 pp |
| eligible warps/scheduler | 0.31 | 0.49 | +58.1% |
| registers/thread | 72 | 96 | +24 |
| local-store spill sectors | 6,022,572 | 3,392,467 | -43.7% |
| L1 / L2 hit rate | 64.36% / 98.64% | 65.52% / 97.57% | +1.16 / -1.07 pp |

The kernel is 1.77x faster. It remains L2/local-memory latency bound, with
long, wait, and short-scoreboard stalls at 28.6%, 27.9%, and 23.4%.

The force-only specialization was re-profiled from the same final binary at
101.89 us, versus 102.46 us before the change. Its grid, 69 registers/thread,
and 30.6% achieved occupancy are unchanged.

## Correctness and end-to-end

At 20 DNA NPT steps, one-CTA and split2 have identical potential energy,
density, and LJ short-range energy at printed precision. Pressure differs by
0.02 bar, consistent with atomic-order rounding.

Three paired 10000-step DNA NPT runs with output only at the final step:

| path | Calculate_Force (s) | wall (s) | speed (ns/day) |
|---|---:|---:|---:|
| one CTA/SCI | 2.962044 | 4.581623 | 377.198069 |
| energy+virial split2 | 2.944978 | 4.561712 | 378.843801 |

The measured speed difference is `+0.4367%`. Existing 2000-step nsys data
shows 1800 fast and 1800 slow force-only launches, 200 fast and 200 slow
virial-only launches, but only one fast and one slow energy+virial launch.
Consequently, the standard-run difference is below a defensible attribution
threshold and should be treated as run noise rather than claimed as an
end-to-end gain.

As a frequency stress test, writing energy and virial every 10 steps gives:

| path | Calculate_Force (s) | wall (s) | speed (ns/day) |
|---|---:|---:|---:|
| one CTA/SCI | 3.168136 | 5.020156 | 344.247772 |
| energy+virial split2 | 3.050085 | 4.905855 | 352.277099 |

The paired speed gain is `+2.3335%`; force time falls 3.73%. This confirms that
the kernel gain transfers when full-output calls are frequent.

Both frozen water NPT guardrails remained finite with the new gate enabled:
wat160k reached 142.66 ns/day and wat600k reached 45.87 ns/day over 1000 steps.
The gate rejects their LJ-combination specializations.

## Decision

Accept the default-off DNA energy+virial split2 gate. Enable it for DNA
full-output benchmarks in any ensemble, but do not claim a standard
sparse-output end-to-end gain. The steady NVE and NVT force-only calls remain
governed by the existing split2 path and are unchanged.

The next force-kernel experiment should focus on the frequently called
force-only specialization. Its full NCU profile remains latency-bound at
101.89 us, with 27.8% long-scoreboard, 30.5% wait, and 21.5%
short-scoreboard stalls. A small AB-table cache experiment is appropriate;
the table has only 153 `float2` entries for DNA, but the result must be
evaluated as a general AB-table optimization rather than assumed universal.

Artifacts:

```text
ncu_reports/dna_kernel_variants_20260723/dna_cou_energy_virial.ncu-rep
ncu_reports/dna_kernel_variants_energy_virial_split2_final_20260723/dna_cou_energy_virial.ncu-rep
ncu_reports/dna_kernel_variants_energy_virial_split2_final_20260723/dna_cou_force_only.ncu-rep
.tmp/dna-full-split2-correctness-20260723
.tmp/dna-energy-virial-split2-ab-20260723/results.tsv
.tmp/dna-energy-virial-split2-output10-ab-20260723/results.tsv
.tmp/water-energy-virial-split2-guardrail1000-20260723/results.tsv
```
