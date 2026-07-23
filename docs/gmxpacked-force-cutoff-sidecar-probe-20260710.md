# gmxpacked force cutoff-sidecar probe - 2026-07-10

This note records the first force replay experiment after
`gmxpacked-force-kernel-handoff-20260710.md`. All modes remain microbench-only.
No production dispatch is enabled by this work.

## Baseline diagnosis

Payload:

```text
/tmp/sponge-collect-distribution-20260708/wat600k_peak/wat600k_builder_footer.sponge_gmxpacked_forceonly.bin
```

Fresh RTX 4090 `ncu --set full` baseline:

| metric | value |
|---|---:|
| duration | 3.378912 ms |
| SM throughput | 50.73% |
| memory throughput | 48.27% |
| DRAM throughput | 3.34% |
| registers/thread | 70 |
| theoretical/achieved occupancy | 58.33% / 49.76% |
| eligible warps/scheduler | 0.91 |
| total instructions | 1.951B |

Top sampled warp states were wait 23.75%, short scoreboard 21.35%, long
scoreboard 19.24%, and branch resolving 8.12%. The kernel is latency-bound and
register-limited, not DRAM-bandwidth-bound.

Artifacts:

```text
ncu_reports/force_vote_20260710/wat600k_baseline_full.ncu-rep
ncu_reports/force_vote_20260710/wat600k_baseline_full.csv
```

## Rejected warp-vote recompute

The first probe performed an active ballot and cutoff ballot for every
`split/JM/I`, then recomputed distance only on cutoff-pass lanes. It regressed:

| metric | baseline | vote-recompute | delta |
|---|---:|---:|---:|
| NCU duration | 3.378912 ms | 5.137792 ms | +52.1% |
| instructions | 1.951B | 3.656B | +87.4% |
| branch instructions | 376M | 708M | +88.4% |
| registers/thread | 70 | 72 | +2 |

The extra votes, control flow, and distance recomputation dominated. This mode
was removed after profiling.

## Oracle payload ceiling

`production-gmxpacked-sorted-force-oracle-imask` computes an exact host-side
cutoff-pass bit for every `cjpacked split/JM/I`, then clears source `imask` bits
with no passing lane before upload. Host classification is outside timing and
there is no extra device-side metadata load. It is the optimistic payload
contract ceiling.

| payload | source sites kept | baseline | oracle imask | delta |
|---|---:|---:|---:|---:|
| wat160k | 9.156% | 0.986880 ms | 0.208589 ms | -78.9% |
| wat600k | 9.194% | 2.994 ms | 0.727 ms | -75.7% |
| dna AB-table | 59.855% | 0.139525 ms | 0.106534 ms | -23.6% |

The result proves that cutoff-fail payload sites are large enough to justify a
classification contract. The AB-table payload has a much smaller ceiling
because most sites must be retained.

## Sidecar consumer

`production-gmxpacked-sorted-force-oracle-sidecar` keeps the original payload
and loads one 64-bit pass mask per `cjpacked` record. On wat600k it measured
0.745835 ms normally and 0.822880 ms under NCU. The sidecar load is only about
1-3% slower than in-place oracle pruning, and registers remain at 70/thread.

NCU measured 509.8M consumer instructions versus 1.951B in the baseline.

## Device producer plus consumer

`production-gmxpacked-sorted-force-device-sidecar` generates the exact mask on
GPU and immediately consumes it in the force replay kernel. The reported time
includes both launches. The classifier uses a conservative four-FP32-ULP cutoff
guard; the force consumer still applies the original cutoff.

| payload | cjpacked | baseline | producer + consumer | delta |
|---|---:|---:|---:|---:|
| wat160k | 367,963 | 0.986880 ms | 0.930714 ms | -5.7% |
| wat600k | 1,556,747 | 2.999 ms | 2.703 ms | -9.9% |
| dna comb-format diagnostic snapshot | 74,219 | 0.601291 ms | 0.597606 ms | -0.6% |
| dna AB-table | 17,951 | 0.139525 ms | 0.202850 ms | +45.4% |

The comb-format DNA row is not the current production dispatch. Production
detects that the DNA pair table is incompatible with geometric combination
and consumes the 17,951-record AB-table payload. The comb-format row remains
useful only as a payload-size diagnostic.

Final wat600k NCU split:

| kernel | duration | regs/thread | instructions | SM | memory |
|---|---:|---:|---:|---:|---:|
| cutoff-mask producer | 2.197344 ms | 29 | 1.750B | 69.70% | 55.49% |
| sidecar force consumer | 0.818656 ms | 70 | 0.510B | 55.53% | 32.40% |
| combined | 3.016000 ms | n/a | 2.260B | n/a | n/a |

The NCU combined result is 10.7% faster than the 3.378912 ms baseline. The
producer dominates the new path. A one-ballot-per-I producer was faster than a
one-warp-reduction-per-JM probe; the shuffle reduction raised producer time
from 2.190 ms to 2.855 ms and was reverted.

## Correctness

The final device producer was compared outside the timed region with the host
oracle and a one-launch baseline force replay.

| payload | mask missing/extra bits | force tolerance mismatches | max abs force error |
|---|---:|---:|---:|
| wat160k | 0 / 0 | 0 | 1.52587891e-05 |
| wat600k | 0 / 0 | 0 | 1.52587891e-05 |
| dna comb-format diagnostic snapshot | 0 / 0 | 0 | 1.14440918e-05 |
| dna AB-table | 0 / 0 | 0 | 1.14440918e-05 |

Bitwise force differences remain because pruning changes global atomic update
order. The validation tolerance is `1e-5 * (1 + abs(reference))`.

## Decision

Keep the three microbench modes and NCU reports. Do not enable the device
producer globally: small and high-retention payloads regress.

The next production experiment should be default-off and gated before launch.
A provisional `cjpacked >= 300000` threshold separates the two water wins from
the two dna non-wins in this data, but it is not yet a committed policy. A
production gate must run wat160k, wat600k, and dna_cou 10000-step alternating
e2e tests plus an nsys split. The classifier should be fused into an existing
per-step geometry/refresh pass if that removes most of its 2.2 ms producer
cost; otherwise the separate two-launch path is only justified for large water
payloads.
