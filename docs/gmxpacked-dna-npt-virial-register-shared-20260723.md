# gmxpacked DNA NPT virial register/shared experiments - 2026-07-23

This iteration starts from the accepted default-off DNA NPT virial split2
probe. It preserves the external force and per-atom virial layouts and their
consumers.

## NCU baseline

The split2 virial kernel was faster than one CTA/SCI, but its eight-entry
`LTMatrix3` accumulator still spilled heavily:

| metric | split2 baseline |
|---|---:|
| duration | 192.90 us |
| registers/thread | 72 |
| local-memory spilling requests | 2,949,688 |
| static shared memory/block | 1.28 KB |
| achieved occupancy | 37.10% |
| eligible warps/scheduler | 0.35 |
| L1 / L2 hit rate | 51.16% / 98.49% |
| long-scoreboard stall | about 55.4% |

DRAM throughput was only about 7%. This made register spill latency a stronger
first target than global-atomic cache policy.

## Experiment A: relaxed launch bounds

Only the AB-table, per-atom virial, `sci_work_parts == 2` specialization changes
from `minBlocksPerMultiprocessor=13` to `10`. All one-part, NVE, NVT, energy,
total-output, and LJ-combination specializations retain their existing launch
bounds.

Full NCU:

| metric | split2 baseline | launch bounds 10 | delta |
|---|---:|---:|---:|
| duration | 192.90 us | 126.91 us | -34.2% |
| registers/thread | 72 | 96 | +24 |
| local spilling requests | 2,949,688 | 469,486 | -84.1% |
| static shared memory/block | 1.28 KB | 1.28 KB | unchanged |
| achieved occupancy | 37.10% | 28.28% | -8.82 pp |
| eligible warps/scheduler | 0.35 | 0.57 | +62.9% |
| L1 hit rate | 51.16% | 59.18% | +8.02 pp |
| long-scoreboard stall | about 55.4% | 23.1% | -32.3 pp |

The occupancy reduction is profitable: allowing 96 registers removes most
spill traffic and increases scheduler eligibility.

## Experiment B: two-warp shared virial merge

The combined force/virial kernel previously let both warps issue a six-scalar
`LTMatrix3` atomic contribution for each active i atom. Under split2 that can
produce four contributions per SCI/atom: two warps times two CTAs.

The new split2-only path packs each warp's final reduced virial as `float4` plus
`float2` into:

```text
shared_split_virial_{lo,hi}[2][8][8]
```

After one block barrier, split 0 combines the two warp results and emits one
`LTMatrix3` atomic contribution. This reduces the maximum contributors from
four to two without changing `atom_virial`, force storage, or downstream
consumption.

Full NCU against launch-bounds 10:

| metric | launch bounds 10 | + shared merge | delta |
|---|---:|---:|---:|
| duration | 126.91 us | 124.45 us | -1.94% |
| registers/thread | 96 | 96 | unchanged |
| local spilling requests | 469,486 | 398,464 | -15.1% |
| static shared memory/block | 1.28 KB | 4.35 KB | +3.07 KB |
| achieved occupancy | 28.28% | 27.32% | -0.96 pp |
| eligible warps/scheduler | 0.57 | 0.60 | +5.3% |
| L1 hit rate | 59.18% | 66.78% | +7.60 pp |
| long-scoreboard stall | 23.1% | 22.6% | -0.5 pp |
| wait / short-scoreboard stall | 30.8% / 26.1% | 30.7% / 25.1% | no regression |

The merge is a small positive increment. The barrier and shared traffic do not
erase the atomic reduction, but the main gain remains the relaxed register
constraint.

Cumulative split2 baseline to final kernel:

```text
duration: 192.90 -> 124.45 us (-35.5%, 1.55x)
local spilling requests: 2,949,688 -> 398,464 (-86.5%)
```

## Correctness and end-to-end

Both changes pass a 20-step DNA NPT comparison. Temperature and total potential
energy match at printed precision. Pressure and individual force components
only show the expected last-digit differences from atomic accumulation order.

End-to-end timing was collected on a shared RTX 4090. Several runs overlapped
an external Python compute process using about 18 GB GPU memory and up to tens
of percent of the SMs; those runs are invalid and are not included below.

Three uncontaminated launch-bounds pairs:

| path | Calculate_Force mean | wall mean | speed mean |
|---|---:|---:|---:|
| split2 baseline | 3.113822 s | 4.840657 s | 357.029612 ns/day |
| launch bounds 10 | 3.050736 s | 4.777846 s | 361.713725 ns/day |

Mean paired speed improvement is about `+1.32%`; force time falls about
`2.03%`.

Two currently uncontaminated shared-merge pairs:

| path | Calculate_Force mean | wall mean | speed mean |
|---|---:|---:|---:|
| launch bounds 10 | 3.148806 s | 5.010134 s | 344.953110 ns/day |
| + shared merge | 3.125265 s | 4.986262 s | 346.587799 ns/day |

The shared merge reduces force time by about `0.75%`. Its wall/speed signal is
small (`about +0.48%`) and should be interpreted together with the unambiguous
full-NCU result.

One uncontaminated final public-baseline pair:

| path | Calculate_Force | wall | speed |
|---|---:|---:|---:|
| public SPONGE 2.0 | 3.608697 s | 6.382168 s | 270.781464 ns/day |
| final candidate | 3.186185 s | 5.086110 s | 339.782837 ns/day |

The final candidate is `25.48%` faster in this pair, consistent with the prior
three-pair split2 result of `+25.40%`. Additional attempted pairs were invalid
because the intermittent external compute job affected only one side of each
pair.

## Water guardrails

The two water NPT systems still dispatch the LJ-combination,
`sci_work_parts=1` specialization. Binary resource inspection for its fast and
slow variants is identical before and after the shared merge:

```text
fast: REG=96, STACK=32 B, SHARED=1536 B
slow: REG=96, STACK=48 B, SHARED=1536 B
```

Final 10000-step guardrails:

| system | Calculate_Force | wall | speed | final temperature / pressure |
|---|---:|---:|---:|---:|
| wat160k NPT | 5.126573 s | 6.179741 s | 139.825684 ns/day | 300.35 K / 5.98 bar |
| wat600k NPT | 15.840239 s | 19.696973 s | 43.868996 ns/day | 299.93 K / -16.93 bar |

Both are finite. The saved launch-bounds binary produced 134.082977 and
43.771252 ns/day respectively. Because the water compiled resource contract is
unchanged and these are single runs on a shared GPU, the wat160k timing
difference is treated as runtime noise rather than an optimization effect.

## Decision

Keep both changes inside the existing default-off DNA NPT virial split2 probe.
Do not broaden them to NVE, NVT, energy, or water LJ-combination paths.

The next virial iteration should not add more SCI partitions. Remaining work is
latency-bound at 124.45 us, 96 registers/thread, 27.3% achieved occupancy, and
0.60 eligible warps/scheduler. Further work should target the lifetime or
scalarization of the eight-entry virial accumulator, one NCU-driven change at a
time.

Artifacts:

```text
ncu_reports/gmxpacked_virial_split2_lb10_20260723
ncu_reports/gmxpacked_virial_split2_lb10_merge_20260723
.tmp/dna-virial-lb10-smoke20-20260723
.tmp/dna-virial-merge-smoke20-20260723
.tmp/dna-virial-lb10-ab10000-20260723
.tmp/dna-virial-lb10-ab10000-clean1-20260723
.tmp/dna-virial-merge-ab10000-clean1-20260723
.tmp/dna-virial-merge-ab10000-clean23-20260723
.tmp/dna-virial-register-shared-public-ab10000-20260723
.tmp/water-virial-merge-final-guardrail10000-20260723
```
