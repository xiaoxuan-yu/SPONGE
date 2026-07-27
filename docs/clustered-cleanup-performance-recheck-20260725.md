# Clustered cleanup performance recheck - 2026-07-25

## Decision

**Performance pass: the cleaned production path is now a valid migration
reference.**

The clustered rebuild horizon, the MD neighbor-list default used by clustered
direct, and the production DNA SCI partition have been restored without
bringing back their probe gates. A true alternating-order recheck shows water
flat within 0.3%, the complete ensemble matrix shows the DNA partition is
21--26% faster end to end, and a 30,000-step wat600k guardrail remains within
0.4%. No retained CUDA change lacks a post-change NCU profile.

The remaining sections retain the discovery chronology. This section and
“Current repair state” below supersede their earlier interim conclusions.

## Current repair state

The exact pre-cleanup source at `02272de` was rebuilt independently as:

```text
/tmp/sponge-cleanup-baseline-head-02272de/build-dev-cuda13/SPONGE
```

Current repairs:

1. Clustered LJ defaults to a 10 Angstrom rebuild horizon again; an explicit
   global `skin` no longer collapses that horizon.
2. When clustered direct is selected and global `skin` is omitted, MD neighbor
   maintenance uses the pre-cleanup 10 Angstrom default. Explicit global skin
   remains authoritative.
3. AB-table force-only production dispatch uses contiguous SCI split3; AB-table
   full output uses SCI split2. Water LJ-combination remains one-part.
4. The externally launched gmxpacked variants remain force-only and full. The
   full entry selects a compiler-pruned virial-only body for pressure updates
   instead of doing energy arithmetic on every virial step.

Five paired 10,000-step NPT cycles were run with true alternating old/current
order and fine timers enabled on both binaries:

| system | paired current/old `Calculate_Force` deltas | median |
|---|---|---:|
| wat160k | +1.99%, -0.76%, -0.66%, +0.43%, +0.24% | **+0.24%** |
| wat600k | +0.27%, +1.03%, +0.26%, +0.79%, +0.01% | **+0.27%** |

Raw results:

```text
/tmp/sponge-scoped-finetimers-alternating-water-5cycle-20260726/results.tsv
```

The same scoped-output-buffer binary passed the final three-cycle matrix:

| system | ensemble | median force ratio | median wall ratio | median speed ratio |
|---|---|---:|---:|---:|
| DNA_COU | NVT | 0.7379 | 0.8192 | 1.2207 |
| DNA_COU | NPT | 0.7487 | 0.8231 | 1.2149 |
| wat160k | NVT | 0.9887 | 0.9884 | 1.0117 |
| wat160k | NPT | 1.0012 | 1.0014 | 0.9986 |
| wat600k | NVT | 1.0085 | 1.0083 | 0.9918 |
| wat600k | NPT | 1.0067 | 1.0054 | 0.9946 |

Ratios are current/old; lower is better for force/wall and higher is better for
speed. All 36 runs completed 10,000 steps without OOM, CUDA errors or
non-finite output:

```text
/tmp/sponge-scoped-final-ensemble-matrix-3cycle-20260726/results.tsv
```

The small wat600k difference was checked with 30,000-step, three-cycle,
alternating fine-timer runs:

| ensemble | median force delta | median wall delta | median speed delta |
|---|---:|---:|---:|
| NVT | +0.326% | +0.390% | -0.389% |
| NPT | +0.368% | +0.257% | -0.257% |

The direct kernel was slightly faster in both long runs; gmxpacked launch was
within +0.15%. This is the accepted measurement-noise/fixed-overhead envelope,
not a kernel regression:

```text
/tmp/sponge-scoped-wat600k-long-30k-3cycle-20260726/results.tsv
```

## Current NCU attribution

All reports use the same wat160k NPT input, 3,901-block production kernel and
the `dev-cuda13` pixi NCU. The full-kernel regex selected the first virial
pressure update rather than the force-only launch.

| production path | duration | regs/thread | achieved occupancy | local spill requests |
|---|---:|---:|---:|---:|
| old virial-only | 410.05 us | 96 | 31.61% | 2,626,347 |
| first internal-mode full | 418.18 us | 96 | 32.16% | 4,154,894 |
| scoped/two-body internal full | 415.39 us | 96 | 31.89% | 3,211,723 |
| old force-only | 316.38 us | 69 | 40.84% | 0 |
| current force-only | 296.99 us | 69 | 41.33% | 0 |

Reports:

```text
/tmp/sponge-ncu-baseline-virial-wat160k-npt-20260725/report.ncu-rep
/tmp/sponge-ncu-full-modes-refined-wat160k-npt-20260725/report.ncu-rep
/tmp/sponge-ncu-full-modes-scoped-wat160k-npt-20260725/report.ncu-rep
/tmp/sponge-ncu-baseline-forceonly-wat160k-npt-20260725/report.ncu-rep
/tmp/sponge-ncu-current-forceonly-wat160k-npt-20260725/report.ncu-rep
```

The main force-only kernel is faster than the old baseline. The current full
symbol still contains two compiler-specialized pair bodies; its sm80 text
section is 188,928 bytes, versus 104,576 bytes for the old virial-only symbol
and 146,816 bytes for the old full symbol. The full entry is 1.30% slower than
the removed virial-only entry, but it runs only on pressure/output steps and
does not produce a measurable end-to-end regression in the alternating or
30,000-step checks. Further kernel edits are stopped here: NCU supplies no
evidence for a safe gain, and a third externally dispatched virial variant
would violate the cleanup contract.

An earlier matched fine-timer pair, before the scoped internal-mode repair,
also showed non-kernel cost:

| wat160k NPT, 10,000 steps | old | cleanup at that point | delta |
|---|---:|---:|---:|
| payload build | 1.066444 s | 1.146812 s | +0.080368 s |
| coordinate gather | 0.365516 s | 0.367722 s | +0.002206 s |
| gmxpacked launch | 2.749455 s | 2.953146 s | +0.203691 s |
| sorted-force scatter | 0.023289 s | 0.047962 s | +0.024673 s |
| `Calculate_Force` | 5.117490 s | 5.523336 s | +0.405846 s |

This pair cannot be used as the final current measurement because the kernel
has since changed. It establishes that the next paired run must retain stage
timers and must not focus only on the LJ kernel.

## Initial failed-state tested state (chronology)

```text
HEAD: 02272de7e38bda238bd639da0b3cf1dc87b8eed1
worktree: dirty, 19 tracked paths changed
diffstat: 1659 insertions, 8070 deletions
GPU: NVIDIA GeForce RTX 4090, compute capability 8.9
CUDA: 13.0
SPONGE binary SHA-256:
  802f43ec3b9940c776e143c41fe2758bfe818e37bf66f97ef7596f6f15ec1e33
NBNXM_MICROBENCH SHA-256:
  9fefa660d59b93d4e8e7084d2511df2ad1e61486c8911b91417dac76588bb4ba
```

The build was current:

```sh
cmake --build build-dev-cuda13 \
  --target SPONGE NBNXM_MICROBENCH --parallel 4
```

Ninja reported no pending work.

## Initial failed-state end-to-end protocol (chronology)

The recheck used the existing accepted driver:

```text
.tmp/run_public_comparison_ensemble_matrix_20260723.sh
```

Invocation:

```sh
OUT_ROOT=/tmp/sponge-cleanup-ensemble-recheck-20260725 \
IMPLEMENTATIONS=current RUNS=3 STEPS=10000 \
bash .tmp/run_public_comparison_ensemble_matrix_20260723.sh
```

It preserved the accepted RTX 4090, one-process, 10,000-step inputs and the
full environment recorded by the driver, including clustered direct, outer
lifecycle, active view, rolling-source-cache off, fixed-shift/subgroup builder,
fill-prune reuse, cooperative fixed-light count, queue2 fused count, cached
inner-active fill and 128-thread shift refresh. DNA additionally requested
full-dense padding and the former force split2 gate.

Input hashes:

```text
DNA NVT mdin:
  532a038375215554f5b1ce64924034da5401125ec58ca26e4d446e63fb228df8
water160k NVT mdin:
  7041e64043918275d75a132d5c93d0e820820e546d6a3cd8ff7b82de60f2c7a8
```

## First-cycle result

| system | ensemble | force (s) | wall (s) | speed (ns/day) | historical current | delta |
|---|---|---:|---:|---:|---:|---:|
| DNA_COU | NVT | 28.104386 | 30.579638 | 56.513844 | 368.115509 | -84.65% |
| DNA_COU | NPT | 28.070890 | 30.844982 | 56.027683 | 349.552999 | -83.97% |
| water160k | NVT | 28.447577 | 30.080432 | 28.725864 | 148.528737 | -80.66% |
| water160k | NPT | 28.243945 | 29.943026 | 28.857685 | 144.324870 | -80.01% |

All four runs were finite. The failure is performance, not an immediate NaN
or launch failure.

Raw table:

```text
/tmp/sponge-cleanup-ensemble-recheck-20260725/results.tsv
```

## Attribution 1: clustered rebuild skin semantics

The accepted ensemble inputs set global `skin = 2.0` or inherit it. Before the
cleanup, enabled clustered LJ retained a separate 10 Angstrom rebuild skin
unless `[LJ].clustered_rebuild_skin` was explicitly set. The cleaned code:

- changes `kDefaultClusteredRebuildSkin` from 10 to 2;
- copies every global `skin` setting into `rebuild_skin`;
- removes the separate `halo_skin` behavior.

Current locations are
`SPONGE/Lennard_Jones_force/clustered_lj.cpp:25147` and
`SPONGE/Lennard_Jones_force/clustered_lj.cpp:25252`.

An input-only A/B added:

```toml
[LJ]
clustered_rebuild_skin = 10.0
```

No source was changed.

| system | ensemble | force (s) | wall (s) | speed (ns/day) | change vs broken 2 A |
|---|---|---:|---:|---:|---:|
| DNA_COU | NVT | 4.979849 | 7.317662 | 236.164612 | +317.9% |
| water160k | NVT | 5.992001 | 7.329754 | 117.887512 | +310.4% |

This establishes the lifecycle semantic change as the dominant common
regression. It must be repaired before the formal matrix is repeated. It does
not make either system pass: after this correction, water160k remains 20.63%
below its historical current speed and DNA remains 35.84% below its historical
current speed.

## Same-machine public sanity

The public executable was run once on the same native inputs:

```text
/home/youmans/sidereus/P8_short_screening_2ps/.pixi/envs/default/bin/SPONGE
```

| system | force (s) | wall (s) | speed (ns/day) |
|---|---:|---:|---:|
| DNA_COU NVT | 4.231638 | 6.835968 | 252.805893 |
| water160k NVT | 10.309362 | 12.637525 | 68.374657 |

The machine was slower than the historical public water result, so historical
absolute speed alone is not a sufficient diagnosis. It is also not a waiver
for the water regression. Same-machine relative results are:

- current water160k with explicit 10 Angstrom rebuild skin is 72.41% faster
  than public;
- current DNA with the same repair is 6.58% slower than public.

For water160k, the historical current/public speed ratio was 1.795640, while
the same-machine recheck ratio is 1.724140. The normalized advantage regressed
3.98%, outside the 3% gate. The current/public `Calculate_Force` ratio also
regressed by 6.48%:

| water160k NVT ratio | historical | recheck | normalized change |
|---|---:|---:|---:|
| speed, current/public | 1.795640 | 1.724140 | -3.98% |
| force time, current/public | 0.545831 | 0.581219 | +6.48% |

Therefore water160k is independently unqualified even after compensating for
the slower same-machine public run. Its residual regression still needs
builder/lifecycle versus force-kernel attribution. DNA is also independently
unqualified and is slower than public outright.

## Attribution 2: water kernel status

The fresh water160k payload replay measured:

| mode | average |
|---|---:|
| comb-gmxpacked | 0.277435 ms |
| sorted-force | 0.270448 ms |

These numbers confirm the current payload and replay are operational, but
there is no accepted historical raw replay for this exact regenerated payload.
They cannot be used to waive the end-to-end failure or to claim that the
residual regression is outside the kernel. Water remains failed until a
matched pre-cleanup binary/payload A/B or an equivalent NCU-attributed repair
passes the complete end-to-end gate.

## Attribution 3: DNA force kernel

Fresh force-only payloads were dumped with explicit 10 Angstrom rebuild skin:

| payload | atoms | SCI | CJ packed | exclusions | SHA-256 |
|---|---:|---:|---:|---:|---|
| water160k | 164544 | 3901 | 90109 | 30449 | `c4b68eace9ae48ad2d71aab234e9e8754f0d027db0201255b6f45dc5867645ce` |
| DNA_COU | 31662 | 996 | 17951 | 6109 | `3a81f3a27445842a4da2b0c029f8b038361e2a8e39e5fdd47b5c202edc1354f0` |

Raw payloads:

```text
/tmp/sponge-cleanup-wat160k-payload.sponge_gmxpacked_forceonly.bin
/tmp/sponge-cleanup-dna-payload.sponge_gmxpacked_forceonly.bin
```

Unprofiled replay used 200 warmup and 2,000 measured iterations:

| payload/mode | average |
|---|---:|
| water160k comb-gmxpacked | 0.277435 ms |
| water160k sorted-force | 0.270448 ms |
| DNA current one-part sorted-force | 0.157905 ms |
| DNA SCI split2 | 0.093590 ms |
| DNA SCI split3 | 0.068508 ms |

Both DNA partitioned variants passed replay tolerance:

```text
split2 force_tolerance_mismatches=0, max_abs=1.90734863e-05
split3 force_tolerance_mismatches=0, max_abs=2.28881836e-05
```

Split2 is 40.73% faster and split3 is 56.61% faster than the current one-part
kernel on the same payload. The microbench still contains the algorithms, but
the cleanup removed their production gates and associated force scratch/
writeback path.

## NCU comparison

NCU was provided by the `dev-cuda13` pixi environment. Both reports used the
same DNA payload and covered SpeedOfLight, memory hierarchy, scheduler,
warp-state, instruction, launch and occupancy sections.

| metric | current one-part | split3 |
|---|---:|---:|
| duration | 161.95 us | 79.58 us |
| grid blocks | 996 | 2988 |
| waves/SM | 0.56 | 1.67 |
| registers/thread | 68 | 68 |
| local spill requests | 0 | 0 |
| compute throughput | 16.85% | 36.23% |
| memory throughput | 10.87% | 24.58% |
| achieved occupancy | 15.88% | 36.11% |
| active warps/scheduler | 2.27 | 4.64 |
| eligible warps/scheduler | 0.38 | 0.99 |
| no eligible | 69.25% | 47.35% |
| executed instructions | 30,791,902 | 31,986,478 |
| active threads/warp | 21.91 | 22.20 |
| branch efficiency | 81.54% | 80.99% |

The split kernel executes slightly more instructions and has similar branch
efficiency. Its 50.86% profiled duration reduction comes from exposing enough
CTA work to hide latency, not from lower register use, spilling or a different
arithmetic shortcut.

Reports:

```text
/tmp/sponge-cleanup-dna-current-full-20260725.ncu-rep
/tmp/sponge-cleanup-dna-split3-full-20260725.ncu-rep
/tmp/sponge-cleanup-dna-current-stalls-20260725.ncu-rep
/tmp/sponge-cleanup-dna-split3-stalls-20260725.ncu-rep
```

## Migration gate

Completed:

1. Scoped full candidate compared against the exact old binary with five
   alternating fine-timer pairs.
2. Payload, gather, direct launch and scatter separated and checked.
3. Production force-only and full kernels re-profiled after the retained code
   change, including roofline, cache, stalls, occupancy, registers and spills.
4. Final three-cycle DNA/wat160k/wat600k NVT+NPT matrix completed.
5. wat600k residual checked with 30,000-step alternating runs.
6. Both CPU skin-contract tests passed.

Performance Phase 0 is passed. Before starting a migration commit series, run
the remaining correctness-only gates: GPU minimization, full-output numerical
comparison and the retained microbench tolerance checks. Keep only the
production force-only/full algorithms and retained microbench; do not restore
probe gates, native LJ fallback, solvent-LJ work or SPONGE Manager.
