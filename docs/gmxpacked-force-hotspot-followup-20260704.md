# gmxpacked force hotspot follow-up - 2026-07-04

This note records the next optimization direction after the fixed-shift
candidate-leaf cooperative traversal probe failed its stop rule.

## Current Decision

Do not continue the candidate-leaf traversal line for now.

The cooperative traversal probe made the traversal path slower:

| kernel | duration | regs/thread | eligible warps/sched | issued warp/sched | spill |
|---|---:|---:|---:|---:|---:|
| traversal probe baseline | 28.038 ms | 31 | 2.67 | 0.58 | 0 |
| coop traversal probe | 29.673 ms | 38 | 2.80 | 0.60 | 0 |

The transient cooperative production collector also regressed:

| kernel | duration | regs/thread | eligible warps/sched | issued warp/sched | spill |
|---|---:|---:|---:|---:|---:|
| production onepass collect baseline | 30.905 ms | 48 | 2.09 | 0.56 | 0 |
| transient coop production collect | 32.946 ms | 47 | 2.20 | 0.58 | 0 |

Artifacts:

```text
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_coop_traversal.log
/tmp/sponge-fixed-light-probes-20260704/ncu_candidate_leaf_coop_production.log
```

The default-off cooperative probes remain useful diagnostics, but there is no
production `COOP_COLLECT` gate in the final code. Keep
`SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_NODEBOX_OPT` and all cooperative
candidate-leaf envs out of the peak env.

## Hotspot Order

`nsys` is not available in the current PATH or the `dev-cuda13` pixi env, so the
ranking below uses the latest existing probes-off CUDA kernel summary:

```text
/tmp/sponge-fixed-light-probes-20260704/nsys_candidate_leaf_default_10000_stats_cuda_gpu_kern_sum.csv
```

Top grouped totals from that run:

| group | total time | instances | note |
|---|---:|---:|---|
| force kernels | 2773.881 ms | 20002 | largest remaining target |
| candidate collect | 630.884 ms | 24 | current line just failed stop rule |
| dedicated count | 441.436 ms | 24 | already on fixed-light path |
| pair-shift refresh | 439.317 ms | 10025 | prior simple variant improved kernel but regressed e2e |
| fill active sources | 108.730 ms | 10001 | smaller target |
| count active sources | 102.134 ms | 10001 | smaller target |
| candidate leaf masks | 61.026 ms | 25 | smaller target |

Next optimization should target the force path first, but only with end-to-end
gating. The sorted/float4 force kernel is faster as a single launch, yet its
current support path loses the gain over 10000 steps.

## Force A/B

NCU single-launch comparison, current final binary:

Artifacts:

```text
/tmp/sponge-fixed-light-probes-20260704/ncu_force_current_main_6launch.log
/tmp/sponge-fixed-light-probes-20260704/ncu_force_sorted_float4_main_6launch.log
```

Env for current baseline:

```text
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_DEDICATED=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_PARALLEL=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_ONEPASS=1
```

Extra sorted/float4 env:

```text
SPONGE_CLUSTERED_GMXPACKED_FORCE_SORTED_SCRATCH=1
SPONGE_CLUSTERED_GMXPACKED_FUSED_SORTED_FORCE=1
SPONGE_CLUSTERED_GMXPACKED_FLOAT4_SORTED_FORCE=1
```

| variant | main force template | duration avg | regs/thread | eligible warps/sched | issued warp/sched | L2 | spill |
|---|---|---:|---:|---:|---:|---:|---:|
| current | `<0,0,0,0,1,1,1,1,VECTOR,0>` | 299.659 us | 69 | 1.04-1.05 | 0.55-0.56 | ~30.5% | 0 |
| sorted/float4 | `<0,0,0,1,1,1,1,1,float4,0>` | 276.416 us | 71 | 1.18 | 0.59 | ~16.5-17.1% | 0 |

The single main force launch improves by about 7.8%. That is real, but not
sufficient by itself.

10000-step e2e comparison:

Artifacts:

```text
/tmp/sponge-fixed-light-probes-20260704/force_baseline_10000_current.out
/tmp/sponge-fixed-light-probes-20260704/force_baseline_10000_current.err
/tmp/sponge-fixed-light-probes-20260704/force_sorted_10000_current.out
/tmp/sponge-fixed-light-probes-20260704/force_sorted_10000_current.err
```

Both stderr files are empty.

| variant | Calculate_Force | Core Run Wall Time | Core Run Speed | final temp | final LJ | final PM |
|---|---:|---:|---:|---:|---:|---:|
| current | 6.048021 s | 6.533890 s | 132.246857 ns/day | 294.95 K | 78774.83 | 3131409.25 |
| sorted/float4 | 6.152228 s | 6.640481 s | 130.124069 ns/day | 293.50 K | 79223.68 | 3131362.00 |

Decision: do not add the sorted/float4 env trio to the peak env. The current
implementation is a local kernel win but an end-to-end regression.

## Cost Path To Inspect

The sorted/float4 path is not a single-kernel replacement. Relevant code:

- `SPONGE/Lennard_Jones_force/clustered_lj.cpp`: `Gather_Sorted_LJ_Direct_Scratch_From_Plain`
- `SPONGE/Lennard_Jones_force/clustered_lj.cpp`: `LJ_CLUSTERED_DIRECT_CACHE::Gather_Plain`
- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`: sorted force scratch
  memset/reuse before force launch
- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`:
  `Scatter_And_Clear_Sorted_Clustered_Force_Float4`

The current gather path refreshes cluster centers, then gathers sorted atom ids,
shifted `float4` coordinates/charge, LJ type, and optional LJ combination data.
The sorted/float4 force path also needs `d_sorted_frc4` and a scatter-and-clear
back into `frc`.

This makes the next useful question:

Can the current production `VECTOR` force path borrow the sorted/float4 kernel's
lower L2 traffic and issue-rate benefit without paying an extra per-step force
scratch/scatter cost?

## Recommended Next Experiments

1. Profile support kernels under the sorted/float4 env with NCU:

```text
Gather_Sorted_LJ_Direct_Scratch_From_Plain
Refresh_Current_Cluster_Centers_From_Crd
Scatter_And_Clear_Sorted_Clustered_Force_Float4
```

Acceptance gate: the measured extra support cost must be below the main-force
launch saving. Current saving is about `23.2 us` per main launch, so any support
cost that repeats every step must be well below that after accounting for both
force launches.

2. Try an in-place `VECTOR` force micro-optimization before more sorted scratch
plumbing.

Focus on the current production template
`Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device<0,0,0,0,1,1,1,1,VECTOR,0>`.
The target symptoms are low issue rate, no spill, register-limited theoretical
occupancy, and high L2 throughput. Good candidates are narrower load/temporary
lifetimes, fewer per-pair branches in the active-view path, and avoiding extra
metadata loads in the common fixed-light case.

Acceptance gate: single main force launch improves by at least 5% with no extra
per-step kernels, then run 2000-step verify and 10000-step e2e.

3. If sorted/float4 is revisited, make it a cost-model experiment first.

Do not enable the existing three env gates directly in peak. Instead, first
measure whether one of these can remove enough overhead:

- reuse already gathered `d_sorted_xq` without an additional force scratch
  lifecycle;
- fuse scatter/clear with an existing mandatory per-step pass;
- produce forces directly in final atom order while retaining the faster
  `float4` source-load pattern.

Acceptance gate: 10000-step e2e must recover at least 150 ms or improve overall
speed by at least 2% versus the current peak env, with empty stderr and finite
outputs.

## PTX/SASS/NCU Observation - Force Kernel

Artifacts from the follow-up inspection:

```text
/tmp/sponge-fixed-light-probes-20260704/sponge_dump_ptx_20260704.ptx
/tmp/sponge-fixed-light-probes-20260704/sponge_dump_sass_20260704.sass
/tmp/sponge-fixed-light-probes-20260704/force_current_vector.ptx
/tmp/sponge-fixed-light-probes-20260704/force_sorted_float4.ptx
/tmp/sponge-fixed-light-probes-20260704/ncu_force_peak_warp_instr_1launch.log
```

The full SASS dump was interrupted after it exceeded 1 GiB. This build does not
contain native `sm_89` cubins; the RTX 4090 run reports compiled CUDA arch 8.0
and runtime arch 8.9, so runtime NCU is the source of truth for SASS-level
behavior. Static `cuobjdump --dump-resource-usage` is still useful for template
resource shape, but not for exact runtime scheduling.

The current production force launch is:

```text
Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device
<0,0,0,0,1,1,1,1,VECTOR,0>
```

Runtime NCU, one launch, full stable peak env:

| metric | value |
|---|---:|
| duration | 302.432 us |
| registers/thread | 69 |
| static shared memory | 1536 B |
| inst_executed | 146,262,508 |
| DRAM throughput | 5.88% |
| L2 throughput | 30.43% |
| eligible warps/scheduler | 1.05 |
| issue active/scheduler | 0.56 |
| active warps/scheduler | 5.18 |
| warp latency / issued inst | 9.32 cycles |
| long scoreboard stall | 2.11 |
| wait stall | 2.13 |
| short scoreboard stall | 1.42 |
| branch resolving stall | 0.66 |
| not selected stall | 0.89 |
| register spilling | 0 |

Static PTX comparison of current `VECTOR` force versus sorted/float4 force:

| PTX feature | current VECTOR | sorted/float4 |
|---|---:|---:|
| PTX lines | 2854 | 2786 |
| `.reg .b32 %r` | 496 | 484 |
| `.reg .b64 %rd` | 134 | 108 |
| `ld.global*` | 34 | 22 |
| `ld.shared*` | 64 | 64 |
| `shfl*` | 120 | 116 |
| `bra*` | 127 | 123 |
| `setp*` | 141 | 139 |
| `fma*` | 420 | 420 |
| `mul*` | 628 | 615 |
| `add*` | 259 | 243 |
| `atom.global*` | 12 | 11 |

Interpretation:

- The math body is almost unchanged. Both templates have the same 420 PTX
  `fma` instructions and the same repeated `rsqrt.approx`/`rcp.approx` pattern
  from the eight unrolled i-cluster lanes.
- The sorted/float4 launch is faster because it removes address/index work from
  the force kernel. In the current `VECTOR` launch, force writes use
  `sorted_atom_ids[sorted_i]` / `sorted_atom_ids[sorted_j]` before atomic adds
  into final atom order. In the sorted/float4 launch, `compact_force_storage`
  writes to sorted scratch with `sorted_i` / `sorted_j` directly.
- That explains both the PTX delta and the prior NCU delta: current `VECTOR`
  has more global loads, more 64-bit address registers, and higher L2
  throughput; sorted/float4 has lower L2 and better issue, but pays an
  end-to-end scatter/clear cost outside the force launch.

The next force-kernel optimization should therefore target the current
`VECTOR` output/index path, not the LJ/Coulomb arithmetic body:

1. Prototype a force-output path that avoids `sorted_atom_ids` inside the hot
   force loop without adding a full per-step scatter tax. The narrowest
   experiment is a production-compatible sorted-force scratch cost model:
   `VECTOR` scratch first, then float4 scratch only if the support kernels can
   be fused or amortized.
2. If direct final-order output must remain, try caching or restructuring
   sorted-to-atom id loads around the two write sites in
   `CLUSTERED_GMXPACKED_PROCESS_JM` and `CLUSTERED_GMXPACKED_REDUCE_I`. The
   target is fewer `ld.global.u32`, fewer 64-bit address ops, and lower L2
   throughput without changing math.
3. Do not spend the next iteration on changing `rsqrtf`, PME correction math,
   or LJ combination math. NCU shows no math-pipe throttle pressure, and the
   sorted/float4 win appears even with the same arithmetic structure.

Acceptance gate for this line:

- single main force launch improves by at least 5% versus the current
  `~300 us` baseline;
- no increase in per-step support kernels unless the full 10000-step e2e recovers
  at least 150 ms or improves overall speed by at least 2%;
- 2000-step verify remains zero-mismatch and 10000-step finite with empty
  stderr.

## Atomic-Order Force Optimization Plan

Decision after checking the GROMACS force-buffer design: do not pursue the
GROMACS-style grouped ordered-buffer path in this branch. In GROMACS, GPU
short-range nonbonded and GPU listed forces can share the NBNXM atom-data
buffer, while PME and other force sources are merged later through a force
reduction layer. SPONGE's current force ABI is different: other force kernels
write atom-order `dd.frc` directly, usually with atomics, and there is no
central force reduction stage to amortize a new sorted/clustered force buffer.

The next round should therefore keep atom-order `frc` as the production output
and improve the current atomic path in-place. Any experiment that requires a
sorted force scratch plus a scatter/clear epilogue is out of scope for this
round, except as a diagnostic comparison.

### Default-Off Probe Gates

Add probes only if needed, all default-off:

```text
SPONGE_CLUSTERED_GMXPACKED_FORCE_RAW_COMPONENT_ATOMIC_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_FORCE_STAGGERED_ATOMIC_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_FORCE_FIXED_LIGHT_COMMONCASE_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_FORCE_ACTIVE_CLUSTER_BUCKET_PROBE=1
```

Do not add these to the peak env unless the full correctness and e2e gates pass.

### Experiment 1: Raw Component Atomic

Goal: reduce final-order write address overhead without changing the force ABI.

Current write sites use `atomicAdd(frc + atom, VECTOR{...})` or the equivalent
helper. Prototype a force-only template path that computes a raw component base
once:

```text
float* frc_raw = reinterpret_cast<float*>(frc);
int atom3 = atom_id * 3;
atomicAdd(frc_raw + atom3 + 0, fx);
atomicAdd(frc_raw + atom3 + 1, fy);
atomicAdd(frc_raw + atom3 + 2, fz);
```

Apply this only around the current production write sites in
`CLUSTERED_GMXPACKED_PROCESS_JM` and `CLUSTERED_GMXPACKED_REDUCE_I`. Keep the
same atom ids, local/ghost semantics, energy, virial, and central halfshell
behavior.

Expected signal:

- fewer 64-bit address temporaries in PTX;
- fewer `IMAD`/address-generation instructions near force writeback;
- no additional global memory traffic;
- no new support kernels.

Gate:

- single main force launch improves by at least 3%;
- register count does not increase above the current 69 by more than 2;
- no correctness mismatch in 2000-step verify.

### Experiment 2: Staggered Component Atomic

Goal: reduce atomic collision and scoreboard pressure while still writing
atom-order `frc`.

Use a GROMACS-like component order rotation for final force atomics. Consecutive
lanes should not all issue x, then y, then z for the same writeback pattern.
Implement as a small helper that takes `atom_id`, `fx/fy/fz`, and a stable lane
or thread id, rotates component order by `lane % 3`, and writes the same final
values.

Expected signal:

- lower long-scoreboard stall and/or lower L2 pressure;
- same or similar instruction count;
- no change in force values beyond existing atomic-order nondeterminism.

Gate:

- single main force launch improves by at least 3%;
- if raw component atomic also passes, test raw+staggered combined;
- combined variant must improve at least 5% before e2e testing.

### Experiment 3: Fixed-Light Common-Case Force Kernel

Goal: remove branches and metadata loads that are not needed in the current
peak fixed-light force-only path.

Prototype a specialized production-compatible force-only template for the
current peak env:

```text
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_DEDICATED=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_PARALLEL=1
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_ONEPASS=1
```

Focus only on the hot force kernel. Do not change builder payload format in this
round. Candidate specializations:

- split central-shift and non-central-shift writeback paths if runtime data show
  one dominates;
- separate local-j and mixed local/ghost records if payload stats show enough
  local-only records;
- remove inactive output variants from the force-only instantiation when energy
  and virial are off.

Expected signal:

- fewer `setp`/`bra` instructions in PTX;
- lower branch resolving stall;
- no increase in payload build time.

Gate:

- single main force launch improves by at least 5% by itself, or at least 7%
  when combined with Experiments 1 and 2;
- no added per-step kernels.

### Experiment 4: Active-Cluster Bucket Probe

Goal: test whether register pressure from eight unrolled i-cluster accumulators
is hurting issue rate enough to justify a narrow specialized kernel.

Add a probe path that dispatches a reduced active-cluster variant only if the
existing payload metadata can identify a dense bucket without rebuilding the
payload. Do not add a new sort or compaction pass for this experiment.

Expected signal:

- registers drop from the current 69 toward 64 or lower;
- active warps per scheduler improves;
- no loss from extra dispatch overhead.

Gate:

- only continue if NCU shows a real occupancy or issue-rate gain;
- stop if the probe needs a new per-step classification kernel.

### Measurement Protocol

Use the current baseline kernel:

```text
Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device
<0,0,0,0,1,1,1,1,VECTOR,0>
```

Baseline numbers to compare against:

```text
single launch: ~302 us
registers/thread: 69
L2 throughput: ~30%
eligible warps/scheduler: ~1.05
issue active/scheduler: ~0.56
spill: 0
```

For each probe:

1. Build with `ninja -C build-dev-cuda13 SPONGE`.
2. Run NCU on one stable main force launch and record duration, registers,
   `ld.global`, atomics, L2 throughput, eligible warps, issue rate, scoreboard
   stalls, and branch resolving stall.
3. If the single-launch gate passes, run 2000-step verify.
4. If verify passes, run 10000-step finite/e2e.
5. Run `git diff --check`.

Final acceptance:

- no probe is kept unless the production main force launch improves by at least
  5% with no extra per-step support kernels;
- no probe is considered for peak env unless 10000-step e2e recovers at least
  150 ms or improves total speed by at least 2%;
- verify must keep `flag_mismatch=0`, `cj_mismatch=0`, `excl_mismatch=0`,
  `row_count_mismatch=0`, and `field_mismatch=0`;
- 10000-step finite run must have empty stderr and no NaN/Inf.

### Stop Conditions

Stop the atom-order atomic line for this round if:

- raw/staggered atomics do not reach a combined 5% single-launch improvement;
- common-case specialization lowers branch count but does not improve launch
  time;
- any promising kernel win is offset by new per-step launch, classification, or
  memory-clear overhead;
- register count rises enough to reduce active warps without a matching latency
  reduction.

If these stop conditions are met, move to a different remaining hotspot instead
of returning to sorted force scratch.

## Atomic-Order Probe Results

Implementation added two default-off force-only gmxpacked full-local dense
probes:

```text
SPONGE_CLUSTERED_GMXPACKED_FORCE_RAW_COMPONENT_ATOMIC_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_FORCE_STAGGERED_ATOMIC_PROBE=1
```

Gate trace confirmed both probes only affect the non-compact force-only direct
call. Energy/virial and compact/sorted scratch calls keep the baseline
component-atomic path.

Artifacts:

```text
/tmp/sponge-fixed-light-probes-20260704/raw_component_probe_peak_smoke.out
/tmp/sponge-fixed-light-probes-20260704/raw_component_probe_peak_smoke.err
/tmp/sponge-fixed-light-probes-20260704/ncu_force_raw_component_main_6launch.log
/tmp/sponge-fixed-light-probes-20260704/staggered_atomic_probe_peak_smoke.out
/tmp/sponge-fixed-light-probes-20260704/staggered_atomic_probe_peak_smoke.err
/tmp/sponge-fixed-light-probes-20260704/ncu_force_staggered_atomic_main_6launch.log
/tmp/sponge-fixed-light-probes-20260704/staggered_atomic_verify2000.out
/tmp/sponge-fixed-light-probes-20260704/staggered_atomic_verify2000.err
/tmp/sponge-fixed-light-probes-20260704/staggered_atomic_10000.out
/tmp/sponge-fixed-light-probes-20260704/staggered_atomic_10000.err
```

NCU comparison against
`/tmp/sponge-fixed-light-probes-20260704/ncu_force_current_main_6launch.log`,
filtered to the hot SCI-safe force-only launch:

| variant | template suffix | duration avg | regs/thread | eligible warps/sched | issued warp/sched | L2 | spill |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `<...,1,VECTOR,0>` | 299.659 us | 69 | 1.047 | 0.557 | 30.727% | 0 |
| raw component | `<...,1,VECTOR,0,1>` | 301.520 us | 69 | 1.055 | 0.560 | 30.505% | 0 |
| staggered component | `<...,1,VECTOR,0,0,1>` | 288.192 us | 69 | 1.235 | 0.595 | 32.095% | 0 |

Raw component addressing failed the single-launch gate (`+0.62%` slower). This
confirms the current component helper is already close to the raw pointer form
for the peak writeback.

Staggered component atomics passed the single-launch 3% gate (`-3.83%`) without
raising registers or spilling. The 2000-step verify also passed with zero
mismatches at every sampled step:

```text
flag_mismatch=0
cj_mismatch=0
excl_mismatch=0
row_count_mismatch=0
field_mismatch=0
```

However, the 10000-step e2e run did not recover the kernel-level win:

| variant | Calculate_Force | Core Run Wall Time | Core Run Speed | stderr | final temp | final LJ | final PM |
|---|---:|---:|---:|---|---:|---:|---:|
| current baseline | 6.048021 s | 6.533890 s | 132.246857 ns/day | empty | 294.95 K | 78774.83 | 3131409.25 |
| staggered component | 6.050280 s | 6.531224 s | 132.300842 ns/day | empty | 294.79 K | 79334.45 | 3130853.50 |

Decision: do not add raw component or staggered component atomic probes to the
peak env. Staggered atomics are a real isolated kernel win, but not large enough
to survive full-run noise/support costs under the current acceptance rule
(`>=150 ms` or `>=2%` e2e). The next force-path experiment should not spend more
time on atomic address/order tweaks unless paired with a larger specialization
that also reduces non-writeback instruction count.

## Force Kernel Hotspot Localization

Added three more default-off isolation gates for the force-only gmxpacked
full-local dense path:

```text
SPONGE_CLUSTERED_GMXPACKED_FORCE_SKIP_WRITEBACK_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_FORCE_SKIP_I_WRITEBACK_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_FORCE_SKIP_J_WRITEBACK_PROBE=1
```

These gates are probes only. `SKIP_I` removes only final i-force component
atomics; `SKIP_J` removes only final j-force component atomics. `SKIP_WRITEBACK`
currently proves compiler-elimination sensitivity rather than a valid
"math-only" measurement, because removing both i and j writes allows ptxas to
drop most force accumulation despite the consume barrier.

Artifacts:

```text
/tmp/sponge-fixed-light-probes-20260704/ncu_force_isolate_baseline_6launch.log
/tmp/sponge-fixed-light-probes-20260704/ncu_force_skip_i_writeback_6launch.log
/tmp/sponge-fixed-light-probes-20260704/ncu_force_skip_j_writeback_6launch.log
/tmp/sponge-fixed-light-probes-20260704/ncu_force_skip_writeback_consume_6launch.log
/tmp/sponge-fixed-light-probes-20260704/ncu_force_instructionstats_source_hot1.log
/tmp/sponge-fixed-light-probes-20260704/ncu_force_memorytables_source_hot1_details.log
/tmp/sponge-fixed-light-probes-20260704/force_current_vector.ptx
```

NCU comparison filtered to the hot SCI-safe force-only launch:

| variant | template suffix | duration samples | mean | regs/thread | interpretation |
|---|---|---:|---:|---:|---|
| baseline | `<...,VECTOR,0,0,0,0,0>` | 292.736, 308.320 us | 300.528 us | 69 | current path |
| skip i writeback | `<...,VECTOR,0,0,0,0,1>` | 299.232, 294.048 us | 296.640 us | 70 | no material speedup |
| skip j writeback | `<...,VECTOR,0,0,0,1,0>` | 302.592, 290.336 us | 296.464 us | 69 | no material speedup |
| skip both writebacks | `<...,VECTOR,0,0,0,1,1>` | 162.592, 156.320 us | 159.456 us | 22 | invalid as math-only; DCE detected |

The single-side skip results are the useful ones: removing either final atomic
half does not move the hot launch beyond normal NCU launch variance. That
matches the earlier raw/staggered result: atom-order component atomics are a
real cost, but not the dominant cost that explains the ~300 us force kernel.

Instruction/source counters for the hot launch:

| metric | value |
|---|---:|
| duration | 302.432 us |
| executed instructions | 146,256,112 |
| issued instructions | 146,785,455 |
| fused FP32 instructions | 31,484,949 |
| non-fused FP32 instructions | 31,946,757 |
| branch instructions | 19,421,126 |
| branch efficiency | 82.20% |
| avg active threads/warp | 23.40 |
| avg predicated-on threads/warp | 22.34 |
| warp cycles per issued instruction | 9.31 |

Warp stall sampling from the same run:

| stall reason | all samples | not-issued samples |
|---|---:|---:|
| wait | 8,020 | 3,884 |
| long scoreboard | 7,863 | 4,262 |
| short scoreboard | 5,412 | 2,935 |
| selected | 3,844 | 0 |
| branch resolving | 2,504 | 1,356 |
| not selected | 2,523 | 0 |
| math pipe throttle | 453 | 106 |
| imc miss | 198 | 182 |

Memory tables for the hot launch:

| metric | value |
|---|---:|
| DRAM throughput | 4.77% |
| L1TEX throughput | 35.64% |
| L2 throughput | 30.70% |
| L1TEX global-load hit rate | 40.71% |
| L2 load hit rate | 82.98% |
| L2 store/reduction hit rate | 82.97% |
| average bytes per global-load sector | 22.16 / 32 |
| global load sectors | 3,773,706 |
| global reduction sectors | 2,837,962 |
| global reduction requests at L1TEX | 659,041 |
| L2 reduction requests | 2,138,011 |
| DRAM bytes read | 14.39 MiB |
| DRAM bytes written | 128 B |
| L2 theoretical global excessive sectors | 1,434,652 / 6,588,776 |
| shared excessive wavefronts | 355,382 / 11,355,339 |

This points to latency and instruction pressure inside the pair loop, not DRAM
bandwidth saturation and not final global atomic serialization as the primary
bottleneck. The red/reduction traffic is visible in L1/L2, but the actual DRAM
writeback is negligible because reductions hit in L2. The bigger signals are
uncoalesced global load sectors, long/short scoreboard stalls, low active
threads from predication/divergence, and a very large instruction body.

Static PTX for the current `VECTOR` force template shows the same shape:

| PTX feature | count |
|---|---:|
| PTX lines | 2,854 |
| `.reg .pred` | 319 predicates |
| `.reg .b32` | 496 integer registers |
| `.reg .f32` | 2,631 float registers |
| `.reg .b64` | 134 address registers |
| `fma.rn` | 420 |
| `mul.ftz` | 581 |
| `setp.*` | 141 |
| `bra` | 127 |
| `shfl.sync` | 120 |
| `ld.shared` | 64 |
| `ld.global` / `ld.global.nc` | 34 / 10 |
| `atom.global` | 12 |
| `rsqrt.approx` / `rcp.approx` | 32 / 32 |

The visible repeated pattern is eight unrolled i-cluster tests per j lane:
shared i-coordinate load, cutoff/exclusion predicates, `rsqrt.approx`, LJ/PME
polynomial arithmetic, fci/fcj accumulation, then component reductions. The PME
correction polynomial and the cutoff/exclusion predication dominate the
instruction body; atomics are only the final 12 PTX operations.

### Optimization Direction

Do not pursue this round as an atomic-buffer redesign. Under the atom-order
force constraint, the most plausible remaining wins are:

1. **Common-case force kernel specialization.** Add a default-off force-only
   template for the measured peak case: full-local dense, `sci_shift_safe`,
   fixed LJ-comb, no per-pair shift bits, no exclusions in the common inner
   loop. Route only rows proven to satisfy those conditions. Acceptance gate:
   reduce branch/predicate count and NCU duration by at least 5% without changing
   force values.

2. **Uncoalesced load reduction.** The main memory warning is global load sector
   waste, not DRAM bandwidth. Probe caching or regrouping of `cjpacked`/j
   coordinate and LJ-comb loads so adjacent lanes consume fewer sectors. This
   should be measured by the `Average Bytes Per Sector For Global Loads` and
   `L2 Theoretical Sectors Global Excessive` counters.

3. **PME correction math specialization.** The PTX has 32 repeated
   `rsqrt.approx`/`rcp.approx` blocks and hundreds of FMA/mul instructions. A
   probe can specialize water/fixed-charge or approximate-polynomial variants
   behind a strict verification gate, but this has higher numerical risk than
   the branch/load cleanup.

4. **Active-lane compaction is lower priority.** Warp active threads are only
   23.4/32 and branch efficiency is 82.2%, but changing row grouping risks
   disturbing the existing payload/order machinery. Treat this as a second-stage
   experiment after the common-case specialization.

Stop conditions for this line:

- common-case specialization does not reduce hot-launch duration by at least 5%;
- global-load sector waste stays near 22% after load-layout/caching probes;
- any math approximation changes force verification beyond accepted tolerances;
- e2e 10000-step improvement is below the existing acceptance bar.

## Build And Verification State

Code after the force-kernel probe work:

```text
ninja -C build-dev-cuda13 SPONGE
git diff --check
```

Build passed after adding the force writeback probes. `git diff --check` passed.
The earlier `pixi run -e dev-cuda13 compile` also passed and emitted the existing
nvcc `compiler-bindir` redefinition warning.

## AB-Table Full-Dense Kernel Completion

Follow-up implementation after comparing the GROMACS NBNXM LJ kernel dispatch:
the full-local-dense gmxpacked fast path no longer requires geometric LJ-comb
compatibility. The force kernel template already had a `use_lj_comb=false`
AB-table path that loads `sorted_lj_type` and fetches packed `A/B` parameters
from `LJ_type_AB_packed`; the missing part was dispatch.

Implemented in `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`:

- removed `use_gmxpacked_lj_comb_kernel` from the
  `gmxpacked_fast_full_local_dense_compatible` gate;
- added `use_lj_comb ? comb : AB-table` dispatch for full-local-dense VECTOR
  kernels;
- added the same dual dispatch for sci-shift split/runtime variants;
- added AB-table full-local-dense float4 sorted-force variants so the
  experimental sorted-force path does not silently force comb semantics.

Offline verification:

```text
ninja -C build-dev-cuda13 SPONGE
git diff --check
nm -C build-dev-cuda13/SPONGE
```

Build and diff check passed. `nm -C` confirms the binary now contains
`use_lj_comb=false, dense_offsets=true, full_local_dense=true` instantiations for
the force-only VECTOR and float4 paths, including sci-shift-only/runtime
combinations.

## Full-Dense Padding Probe

DNA_COU has 31,662 atoms. Without padding it produced 3,958 clusters, so the
fast full-local-dense gate rejected it because the cluster count was not a
multiple of the 8-cluster supercluster shape. Added a default-off gate:

```text
SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING=1
```

The gate pads only the clustered LJ sorted scratch domain. It does not change
the real atom count, atom-order force scatter count, domain decomposition atom
metadata, or any non-LJ module. It is only enabled for the single-rank,
full-local, no-ghost, fixed 8x8 gmxpacked direct shape. Padded clusters have
zero valid/local masks and padded sorted slots are filled with zero charge,
type 0, and zero LJ-comb values.

Runtime validation on DNA_COU with peak env plus gate trace:

```text
call=0    lj_comb=0 fast=1 clusters=3960 total_atoms=31662 padded_atoms=31680 full_local_dense=1
call=2000 lj_comb=0 fast=1 clusters=3960 total_atoms=31662 padded_atoms=31680 full_local_dense=1
```

The second line also verifies the lifecycle fix: `Refresh_Metadata()` and
cache-hit `Build()` paths must preserve `padded_total_atom_numbers` when atom
metadata is unchanged. Before that fix, the step-2000 energy call reset to
`padded_atoms=31662` and fell out of full-local-dense dispatch.

2000-step DNA_COU e2e, same peak env, no gate trace:

| build | Calculate_Force | Core Run Speed |
|---|---:|---:|
| lifecycle-fix baseline | 0.868814 s | 290.818207 ns/day |
| AB-table fast, no padding | 0.839357 s | 298.065735 ns/day |
| AB-table fast + padding | 0.842890 s | 296.746552 ns/day |

Conclusion: keep padding default-off. It fixes coverage for near-full dense
AB-table systems and proves the dispatch path is correct, but this short
DNA_COU e2e run does not show a stable improvement beyond the existing AB-table
dispatch completion.

## AB-Table Padding Force Kernel Tuning

Artifacts:

```text
/tmp/sponge-ab-padding-force-20260705/resource_ab_full_dense_forceonly_sm80.txt
/tmp/sponge-ab-padding-force-20260705/sass_ab_full_dense_forceonly_sm80.sass
/tmp/sponge-ab-padding-force-20260705/sass_comb_full_dense_forceonly_sm80.sass
/tmp/sponge-ab-padding-force-20260705/resource_ab_minmax_full_dense_forceonly_sm80.txt
/tmp/sponge-ab-padding-force-20260705/sass_ab_minmax_full_dense_forceonly_sm80.sass
/tmp/sponge-ab-padding-force-20260705/ncu_dna_ab_padding_force_defaultsplit_6launch.log
/tmp/sponge-ab-padding-force-20260705/ncu_dna_ab_padding_force_skipempty_6launch.log
/tmp/sponge-ab-padding-force-20260705/ncu_dna_ab_minmax_skipempty_6launch.log
/tmp/sponge-ab-padding-force-20260705/ncu_dna_ab_padding_force_skipempty_forceonly_hot1_warpdetails.log
/tmp/sponge-ab-padding-force-20260705/dna_cou_minmax_noskip_2000.stdout
/tmp/sponge-ab-padding-force-20260705/dna_cou_minmax_skipempty_10000.stdout
/tmp/sponge-ab-padding-force-20260705/dna_cou_minmax_noskip_10000.stdout
```

The current release binary does not retain PTX for these filtered cubin
functions: `cuobjdump --dump-ptx --function ...` emitted only fatbin headers.
For this round the instruction-level evidence is therefore SASS plus NCU. A
future PTX source comparison needs a dedicated `-keep` or lineinfo build.

### Static AB vs Comb Shape

Static `sm_80` resource usage for full-local-dense force-only VECTOR kernels:

| variant | sci-shift path | regs/thread | shared | spill/local |
|---|---|---:|---:|---:|
| AB-table | unsafe | 68 | 1280 B | 0 |
| AB-table | safe | 69 | 1280 B | 0 |
| comb | unsafe | 72 | 1536 B | 0 |
| comb | safe | 69 | 1536 B | 0 |

The AB path is not register or shared-memory worse than comb. Its static cost is
instruction shape: before the min/max rewrite, the two AB kernels combined had
more LJ pair-index/address work than comb:

| SASS count | AB-table | comb | delta |
|---|---:|---:|---:|
| `LDG` | 136 | 72 | +64 |
| `SHF` | 264 | 136 | +128 |
| `IMAD` | 452 | 200 | +252 |
| `FMUL` | 811 | 939 | -128 |
| `FFMA` | 1447 | 1447 | 0 |
| `BRA` | 388 | 388 | 0 |

The extra `SHF/IMAD` came from the generic `Get_LJ_Type(a,b)` packed-triangle
index expression in the hot AB path.

### Min/Max Pair-Index Rewrite

Implemented a gmxpacked-local equivalent helper:

```text
hi = max(a, b)
lo = min(a, b)
pair = (hi * (hi + 1) >> 1) + lo
```

This only replaces the AB-table lookup inside
`Nbnxm_Clustered_Lennard_Jones_And_Direct_Coulomb_ForceOnly_Warp_Record_Device`.
It does not change the packed LJ table layout or any non-gmxpacked caller.

Static SASS effect:

| AB SASS count | before | after | delta |
|---|---:|---:|---:|
| total SASS lines | 16947 | 16307 | -640 |
| `SHF` | 264 | 136 | -128 |
| `IMAD` | 452 | 373 | -79 |
| `IMNMX` | 0 | 128 | +128 |
| `LDG` | 136 | 136 | 0 |
| regs/thread | 68/69 | 68/69 | 0 |

NCU on DNA_COU, peak env + full-dense padding + skip-empty, hot force-only
safe launch:

| kernel | duration samples | mean | regs | shared | eligible | issued |
|---|---:|---:|---:|---:|---:|---:|
| AB packed-index | 186.66, 186.37, 186.62, 186.94, 186.21 us | 186.56 us | 69 | 1.28 KiB | 0.35 | 0.29 |
| AB min/max-index | 182.27, 182.21, 181.76, 182.75, 182.40 us | 182.28 us | 69 | 1.28 KiB | 0.34 | 0.28 |

Single-launch improvement is about `2.3%`. The stall shape remains latency
bound rather than bandwidth bound. The hot force-only safe launch reports:

```text
Stall Wait            2.17 inst
Stall Long Scoreboard 1.81 inst
Stall Short Scoreboard 1.39 inst
Stall Branch Resolving 0.58 inst
Avg active threads/warp 22.32
```

### Split Skip-Empty Result

DNA_COU with padding is all sci-shift-safe:

```text
split_skip_empty=1 split_counts_valid=1 split_safe=992 split_unsafe=0
```

With skip-empty off, NCU sees an extra unsafe launch of about `2.8 us` after
each hot safe launch. Enabling
`SPONGE_CLUSTERED_GMXPACKED_SCI_SHIFT_SPLIT_SKIP_EMPTY=1` removes that launch,
but the 2000/10000-step e2e results are too noisy to justify adding this flag
to the global peak env. Keep it as a case-specific diagnostic/tuning env for
now.

### E2E Readout

DNA_COU finite/e2e checks with the min/max-index build:

| run | extra env | Calculate_Force | Core Run Wall Time | Core Run Speed | stderr |
|---|---|---:|---:|---:|---|
| 2000-step prior padding baseline | none | 0.868814 s | 1.188965 s | 290.818207 ns/day | AB fallback notice |
| 2000-step min/max | none | 0.830006 s | 1.151830 s | 300.194336 ns/day | AB fallback notice |
| 2000-step min/max | `SPLIT_SKIP_EMPTY=1` | 0.880432 s | 1.198400 s | 288.528778 ns/day | AB fallback notice |
| 10000-step min/max | none | 4.857783 s | 6.459754 s | 267.529205 ns/day | AB fallback notice |
| 10000-step min/max | `SPLIT_SKIP_EMPTY=1` | 4.504756 s | 6.125005 s | 282.150452 ns/day | AB fallback notice |

All runs were finite with no NaN/Inf. The only stderr line is the expected
notice that the system is not geometric-comb compatible and therefore uses the
AB-table path.

Decision: keep the min/max pair-index rewrite because it is semantics-preserving,
does not increase registers/shared memory, and gives a measured NCU hot-launch
win. Do not promote `SPLIT_SKIP_EMPTY` to the stable peak env based on this
round; its launch-level benefit is real but small and the e2e data is not
stable enough.

## AB Row-Major Matrix Probe

Implemented a default-off probe gate:

```text
SPONGE_CLUSTERED_GMXPACKED_FORCE_LJ_AB_MATRIX_PROBE=1
```

The probe adds a row-major `atom_type_numbers^2` `float2` AB table derived from
the existing packed triangular LJ table. The packed table remains present for
all legacy/non-probe paths. The matrix probe is intentionally narrow: AB-table
gmxpacked force-only, full-local-dense `VECTOR` kernels, including the current
runtime safe/unsafe split launches. It does not assume that every SCI is safe;
split still uses the existing runtime safe flags. Energy/virial,
runtime-sci-shift, F4/scratch, and writeback/atomic probes stay on the packed
table.

Artifacts:

```text
/tmp/sponge-ab-matrix-probe-20260705.WvpdXL/matrix_trace20.stderr
/tmp/sponge-ab-matrix-probe-20260705.e2e.012224/packed
/tmp/sponge-ab-matrix-probe-20260705.e2e.012224/matrix
/tmp/sponge-ab-matrix-probe-20260705.ncu.012339/packed/ncu.log
/tmp/sponge-ab-matrix-probe-20260705.ncu.012339/matrix/ncu.log
/tmp/sponge-ab-matrix-probe-20260705.e2e10000.012420/packed
/tmp/sponge-ab-matrix-probe-20260705.e2e10000.012420/matrix
```

Gate trace on DNA_COU with peak env plus full-dense padding:

```text
call=0 need_energy=1 compact=1 split=1 lj_ab_matrix=0
call=1 need_energy=0 compact=0 split=1 lj_ab_matrix=1
```

This confirms the matrix probe is applied only to the force-only split launch,
not to the step-0 energy path.

2000-step DNA_COU e2e, current build, same peak env plus full-dense padding:

| run | Calculate_Force | Core Run Wall Time | Core Run Speed | stderr |
|---|---:|---:|---:|---|
| packed min/max | 0.824659 s | 1.145730 s | 301.792633 ns/day | AB fallback notice |
| row-major matrix probe | 0.805234 s | 1.122334 s | 308.083710 ns/day | AB fallback notice |

The 2000-step same-build comparison shows a small positive signal:
`Calculate_Force` improves by `19.425 ms` (`2.36%`) and wall time by
`23.396 ms` (`2.04%`).

NCU 6-launch comparison on the same DNA_COU 20-step setup:

| launch | packed min/max | row-major matrix | resource |
|---|---:|---:|---|
| force-only safe, sample 1 | 183.30 us | 180.22 us | 69 regs, 1.28 KiB shared |
| force-only safe, sample 2 | 183.55 us | 179.68 us | 69 regs, 1.28 KiB shared |
| force-only unsafe, sample 1 | 2.82 us | 2.85 us | 68 regs, 1.28 KiB shared |
| force-only unsafe, sample 2 | 2.78 us | 2.78 us | 68 regs, 1.28 KiB shared |

The hot safe launch improves by about `1.9%` with no register/shared-memory
increase. The unsafe launch is tiny and unchanged.

The 10000-step same-build e2e run is not usable as a promotion gate in this
round: both packed and matrix runs reached `NaN` at step 10000 and both slowed to
about 150 seconds wall time:

| run | Calculate_Force | Core Run Wall Time | Core Run Speed | validity |
|---|---:|---:|---:|---|
| packed min/max | 2.440269 min | 148.150590 s | 11.664974 ns/day | NaN at step 10000 |
| row-major matrix probe | 2.487272 min | 150.895659 s | 11.452767 ns/day | NaN at step 10000 |

Decision: keep `SPONGE_CLUSTERED_GMXPACKED_FORCE_LJ_AB_MATRIX_PROBE` default-off.
The kernel-level and 2000-step signals are real but small. It should not enter
the stable peak env until a finite 10000-step run on a stable mdin reproduces an
end-to-end gain.

## Finite Lifecycle Probe

Added a default-off lifecycle probe for the DNA_COU NaN diagnosis:

```text
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_BEGIN=<step>
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_END=<step>
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_INTERVAL=<n>
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_SITE=<site>
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_ABORT=1
```

The probe is host-side and synchronous: it copies `dd.crd`, `dd.vel`, `dd.frc`,
`dd.acc`, `md_info.crd`, `md_info.vel`, and `md_info.frc` to host and reports the
first nonfinite vector per array. For local `dd` arrays it also copies
`dd.atom_local` and prints the mapped global atom id.

Output format:

```text
SPONGE_FINITE_LIFECYCLE_PROBE step=<step> site=<site> array=<array> \
bad_atoms=<count> first_local=<local> first_global=<global> \
value=(<x>,<y>,<z>)
```

Important boundary: this probe synchronizes CUDA work. Continuous probing before
the failing step can mask the timing-sensitive NaN, matching the earlier
cuda-gdb observation. Use `BEGIN=END=<step>` and `SITE=<site>` for low-disturbance
single-boundary checks.

Validation:

```text
ninja -C build-dev-cuda13 SPONGE
```

DNA_COU single-boundary probe with peak env plus full-dense padding:

```text
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_BEGIN=4707
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_END=4707
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_SITE=force_entry
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_ABORT=1
```

Result:

```text
SPONGE_FINITE_LIFECYCLE_PROBE step=4707 site=force_entry array=dd.crd \
bad_atoms=291 first_local=1395 first_global=1395 value=(nan,nan,nan)
SPONGE_FINITE_LIFECYCLE_PROBE step=4707 site=force_entry array=dd.vel \
bad_atoms=291 first_local=1395 first_global=1395 value=(nan,nan,nan)
SPONGE_FINITE_LIFECYCLE_PROBE step=4707 site=force_entry array=dd.frc \
bad_atoms=291 first_local=1395 first_global=1395 value=(nan,nan,nan)
```

This confirms the earlier cuda-gdb finding without modifying source state at
runtime: by the next step's force entry, the local `dd` view already contains
NaNs in coordinates, velocities, and forces. The first bad local id maps directly
to the same global atom id in this single-rank DNA_COU run.

## Async Finite Lifecycle Sentinel

The host-side lifecycle probe can still mask the bug even when used at a single
boundary. A `force_entry@14000` host probe completed finite on a run where the
same input without host copies reached NaN. Added a lower-disturbance device-side
sentinel:

```text
SPONGE_CLUSTERED_GMXPACKED_FINITE_ASYNC_LIFECYCLE_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_BEGIN=<step>
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_END=<step>
SPONGE_CLUSTERED_GMXPACKED_FINITE_LIFECYCLE_PROBE_SITE=<site>
```

It reuses the lifecycle probe filters, but only launches scan kernels and records
the first nonfinite vector in a device record. The device record is copied back
once during finalization:

```text
SPONGE_FINITE_ASYNC_LIFECYCLE_PROBE step=<step> site=<site> site_id=<id> \
array=<array> array_id=<id> first_local=<local> first_global=<global> \
value=(<x>,<y>,<z>)
```

The basic DNA_COU NVT input that reproduces the issue is the force-matrix case,
not the sinkmeta helper default:

```text
cutoff = 8.0
velocity_in_file = "Pmin_velocity.txt"
print_zeroth_frame = 0
[PM]
MPI_size = 0
```

Fresh sinkmeta-style input with `cutoff=10`, no initial velocity, CV, and
restraint completed 10000 steps finite under the same peak env plus
`SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING=1`.

Current async-sentinel evidence on the basic+velocity input:

| run | result |
|---|---|
| 10000-step basic+velocity | finite |
| 15000-step basic+velocity | first printed NaN at step 14000 |
| 20000-step, output only at 20000 | abnormal long execution; interrupted after no mdout progress |
| async all-sites 13999-14000 | first record: `step=13999 site=run_step_begin array=dd.crd first_global=5160` |
| run-step bisection | `run_step_begin@13376` finite, `run_step_begin@13377` nonfinite |
| async `after_step_increment`, step 13375 -> 13376 | nonfinite `dd.crd`, first global 2403 |
| async `after_print@13375` | nonfinite `dd.crd`, first global 573 |
| async `after_iteration@13375` | nonfinite `dd.crd`, first global 797 |
| async `iteration_entry@13375` | nonfinite `dd.crd`, first global 21507 |

Interpretation: the bad state is already in the local `dd` coordinate view before
the next step's force kernels. The strongest current signal is still a local-view
lifecycle/asynchronous dependency issue. The exact first bad step and first atom
move with probe placement, so independent site-filtered runs must not be treated
as one strict timeline.

Next diagnostic target: add finer low-disturbance boundaries before
`force_entry` in the previous step, especially the end of `Main_Calculate_Force`
and any non-default-stream work that can still write `dd.crd` after force/print
boundaries. The async sentinel should stay default-off and diagnostic-only.

## DNA_COU NaN Active-View Root Slice

Follow-up async sentinel work added two diagnostic improvements:

```text
SPONGE_CLUSTERED_GMXPACKED_FINITE_ASYNC_LIFECYCLE_PROBE_STOP=1
SPONGE_CLUSTERED_GMXPACKED_FINITE_ASYNC_LIFECYCLE_PROBE_ARRAY=dd.crd
```

The async sentinel now records first-hit data per site and per array, rather than
only a single global first-hit record. This made same-run site comparisons
possible without treating independent STOP runs as a strict timeline.

Key observations from the basic DNA_COU NVT input:

| run | result |
|---|---|
| STOP `after_step_increment@13375`, two repeats | both nonfinite `dd.crd` |
| STOP `run_step_begin@13375`, two repeats | one finite, one nonfinite |
| single-site first-hit `run_step_begin`, 0-12000 | first bad at step 1741 in that run |
| narrow same-run all-site window 1740-1742 | finite |
| same-run all-site window 13374-13375 | already bad at `run_step_begin@13374` |

Interpretation: first bad step is highly phase-sensitive under probes. The more
actionable result came from gate ablation:

| variant | result |
|---|---|
| peak env, active-view current mask on | NaN observed, e.g. mdout step 3000/4000/9000 depending on run |
| `ACTIVE_VIEW=0` | 2/2 finite to 15000 steps |
| `ACTIVE_VIEW=1`, `ZERO_DIRTY_SOURCE_REUSE=0` | mixed: one finite, one NaN at step 7000 |
| `ACTIVE_VIEW=1`, `DIRTY_INDEX_REFRESH=0` | 2/2 NaN at step 1000 |
| `ACTIVE_VIEW=1`, `CURRENT_MASK=0` | 2/2 finite to 15000 steps |
| `ACTIVE_VIEW=1`, `CURRENT_MASK=1`, zero-dirty contract-gated off | NaN at step 9000 |

The minimum stable slice found in this round is to keep active-view enabled but
make current-coordinate active masks explicit opt-in:

```text
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_CURRENT_MASK=1
```

Default behavior now leaves current-mask disabled. Zero-dirty source reuse is
not fixed by an extra runtime source-displacement scan. Instead it is restricted
by source contract:

```text
ZERO_DIRTY_SOURCE_REUSE is allowed only when CURRENT_MASK=0.
```

With current-coordinate masks, source membership can change even when the
candidate dirty index is zero, so `dirty_candidate_sci == 0` does not imply the
active source/payload is reusable. With stable target-layout active sources,
that implication is the intended contract and no extra per-refresh source scan is
needed.

Validation after the change, using the original peak env that still contains
`SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW=1` and
`SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_ZERO_DIRTY_SOURCE_REUSE=1`:

```text
ninja -C build-dev-cuda13 SPONGE
DNA_COU basic+velocity 15000-step: no mdout NaN
Calculate_Force: 6.563854 s
```

This does not prove the current-mask path is irreparable. It means current-mask
is not a safe default contract for non-water DNA_COU today, while zero-dirty
reuse can be retained only under stable-source active-view semantics.

## Current-Mask Anchor/Generation Contract Attempt

Follow-up implementation made the current-mask payload ownership explicit:

- `gmxpacked_inner_active_anchor_generation` increments whenever the inner
  active anchor coordinate buffer is refreshed.
- `gmxpacked_inner_active_source_generation` increments when active source rows
  are rewritten, including active-view dirty refresh, full dirty refresh, and
  coverage append.
- compact payload reuse now checks both anchor and source generation when
  current-mask semantics are active.
- all active-view source refresh calls now pass the current-mask anchor when
  `use_current_active_mask` is true.
- current-mask dirty-index partial refresh is now a separate experimental gate:

```text
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_CURRENT_MASK_PARTIAL_REFRESH=1
```

This supersedes the earlier zero-dirty conclusion above: zero-dirty source reuse
is no longer hard-disabled only because `CURRENT_MASK=1`; instead it is guarded
by the compact payload anchor/source generation key. However, current-mask
itself still failed validation, so it remains experimental and must not enter
peak env.

Validation on DNA_COU basic+velocity with peak env:

| variant | result |
|---|---|
| stable-source active view, 15000 steps | finite, `Calculate_Force=6.433809 s`, `288.372162 ns/day` |
| current-mask with generation key, dirty partial enabled | mixed; one finite 15000-step run at `6.854192 s`, repeat NaN by mdout step 2000/12000 depending on phase |
| current-mask with partial disabled, full current-mask refresh | NaN already by mdout step 1000, `Calculate_Force=2.716502 min` |

Interpretation: the remaining current-mask bug is not just stale compact payload
reuse. Even a full current-coordinate active prune over cached outer sources can
produce a bad payload. The likely contract gap is geometric source ownership:
the current mask is evaluated with current coordinates while cluster
centers/extents/layout metadata still belong to the cached build geometry. The
next repair should either recompute the geometry used by current-mask active
prune or keep current-mask disabled and continue optimizing the stable-source
active-view path.

## Current-Mask Geometry Source Repair

The next repair targeted the geometric source ownership issue above. The
current-mask path no longer derives active payloads from a cached outer source on
cache-hit steps, and it no longer uses incremental source merge under
`SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_CURRENT_MASK=1`. Instead, cache-hit
current-mask steps fall through to a full gmxpacked record-stream build from the
current `crd`, then build the inner-active payload from that same current-source
record stream.

An intermediate probe that recomputed current cluster centers for active prune
was rejected: force `sorted_xq`, pair-shift metadata, and record-stream sources
are keyed to the layout cluster-center frame, so injecting a separate
current-center frame made the geometry source less consistent, not more.

Validation on DNA_COU basic+velocity with the current-mask peak-like env and
`write_mdout_interval=100`:

| variant | result |
|---|---|
| current-mask cached outer-source active refresh | NaN by mdout step 700-800 |
| stable-source active view, same 1200-step input | finite to step 1200, `Calculate_Force=0.549364 s` |
| current-mask current-source rebuild | finite to step 1200, `Calculate_Force=31.054245 s` |

This is a correctness fix and a diagnosis result, not a peak candidate. It
shows the NaN was caused by deriving current masks from stale cached outer-source
geometry. The current-mask gate must remain default-off and out of peak env until
a cheaper current-source update exists; stable-source active-view remains the
performance path.

## Current-Source Patch/Update Contract Probe

Follow-up implementation adds a default-off current-source patch gate:

```text
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_CURRENT_SOURCE_PATCH=1
SPONGE_CLUSTERED_GMXPACKED_ACTIVE_VIEW_CURRENT_SOURCE_PATCH_MAX_DIRTY_RATIO=<float>
```

The contract is separate from the stable-source active-view path:

- current source anchor generation and current source generation are tracked
  independently from inner-active anchor/source generation;
- source-cache patch is attempted even when the ordinary cached-coordinate
  rebuild check says the old layout is stale, because this path is governed by
  the current source anchor/window contract;
- dirty source candidates patch the source cache by candidate row offsets;
- zero-dirty source candidates reuse the existing source cache without copying
  the 2M source rows;
- active-source partial refresh and compact payload reuse are only used when
  the source row mapping is unchanged and the compact anchor/source generation
  still matches;
- pure source/compact reuse does not refresh source or active anchors. Anchors
  are only advanced when the payload they represent is actually updated.

The last point is essential. An intermediate version refreshed source/active
anchors on zero-dirty reuse and looked fast on 1000 steps, but failed 10000-step
validation with NaN because cumulative displacement was hidden from the dirty
contract.

Validation on wat160k force-only NVE, peak env plus current-mask/current-source
patch:

| variant | result |
|---|---|
| current-mask off, 1000 steps | finite, `Calculate_Force=0.687824 s`, `111.619247 ns/day` |
| current-source patch before partial/compact reuse | finite 1000 steps, `Calculate_Force=23.935474 s`, `3.599688 ns/day` |
| current-source patch with zero-dirty source reuse but wrong anchor refresh | 1000-step finite, `Calculate_Force=0.810725 s`, `98.795883 ns/day`; 10000-step NaN |
| current-source patch with corrected anchor contract | 1000-step finite, `Calculate_Force=5.269162 s`, `16.170336 ns/day`; 10000-step finite, `Calculate_Force=52.510219 s`, `16.216900 ns/day` |

20-step trace after the corrected contract showed 19 current-source patch hits,
17 zero-dirty source-reuse hits, and 2 full payload builds. The extra full build
is expected after cumulative displacement exceeds the active/source reuse
contract; unlike the rejected version, this preserves correctness.

Conclusion: the current-source patch/update contract now gives a finite
correctness path, but its rebuild cadence is too high for peak. It remains
default-off and must not be added to peak env. The next optimization target, if
this line is continued, is reducing full source/payload rebuild frequency under
the corrected anchor contract rather than relaxing the anchor semantics.

## Nsys Overall Split After Current-Source Commit

Artifacts:

```text
/tmp/sponge-nsys-overall-20260706/wat160k_peak/wat160k_peak_2000.nsys-rep
/tmp/sponge-nsys-overall-20260706/wat160k_peak/wat160k_peak_2000.sqlite
/tmp/sponge-nsys-overall-20260706/wat160k_peak/wat160k_peak_2000_kern_cuda_gpu_kern_sum.csv
/tmp/sponge-nsys-overall-20260706/wat160k_peak/wat160k_peak_2000_api_cuda_api_sum.csv
/tmp/sponge-nsys-overall-20260706/dna_peak_padding/dna_peak_padding_2000.nsys-rep
/tmp/sponge-nsys-overall-20260706/dna_peak_padding/dna_peak_padding_2000.sqlite
/tmp/sponge-nsys-overall-20260706/dna_peak_padding/dna_peak_padding_2000_kern_cuda_gpu_kern_sum.csv
/tmp/sponge-nsys-overall-20260706/dna_peak_padding/dna_peak_padding_2000_api_cuda_api_sum.csv
```

Profiler command shape:

```text
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-memory-usage=false
```

The `nsys` binary is packaged under the pixi `nsight-compute` installation:

```text
.pixi/envs/dev-cuda13/nsight-compute-2025.3.1/host/target-linux-x64/nsys
```

Historical note: this nsys split predates the later queue2/cooperative-count
plus cached-fill peak envelope. Keep it for kernel-family attribution, but do
not use its `Core Run Speed` rows as the current peak-speed target.

Runs:

| case | env | step count | status | Calculate_Force | nsys Core Run Speed |
|---|---|---:|---|---:|---:|
| wat160k force-only NVE | then-current stable peak env | 2000 | finite | `1.334775 s` | `117.582184 ns/day` |
| DNA_COU basic NVT | then-current stable peak env plus `SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING=1` | 2000 | finite | `0.869038 s` | `277.370850 ns/day` |

Kernel-time split, excluding one-time initialization kernels such as
`Total_C6_Get`, `get_atom_and_residues_single_domain`, `get_local_device`, and
`device_get_excluded`:

| case | repeated GPU kernel total | gmxpacked force | gmxpacked builder/update | PME/excluded correction | integration/constraint | bonded/nb14 |
|---|---:|---:|---:|---:|---:|---:|
| wat160k | `1226.612 ms` | `514.710 ms` (`41.96%`) | `529.551 ms` (`43.17%`) | `47.520 ms` (`3.87%`) | `52.707 ms` (`4.30%`) | `31.378 ms` (`2.56%`) |
| DNA_COU | `978.513 ms` | `351.164 ms` (`35.89%`) | `192.952 ms` (`19.72%`) | `130.690 ms` (`13.36%`) | `221.377 ms` (`22.62%`) | `49.507 ms` (`5.06%`) |

Top repeated kernels:

| case | kernel | total | instances | avg |
|---|---|---:|---:|---:|
| wat160k | force-only main gmxpacked force `<..., use_lj_comb=1, full_local_dense=1, ...>` | `487.096 ms` | 1999 | `243.670 us` |
| wat160k | candidate leaf collect onepass | `185.660 ms` | 7 | `26.523 ms` |
| wat160k | dedicated fixed-light count | `122.240 ms` | 7 | `17.463 ms` |
| wat160k | pair-shift bit refresh | `89.083 ms` | 2008 | `44.364 us` |
| DNA_COU | force-only main gmxpacked force `<..., use_lj_comb=0, full_local_dense=1, ...>` | `289.337 ms` | 2000 | `144.668 us` |
| DNA_COU | PME excluded correction | `130.690 ms` | 2001 | `65.312 us` |
| DNA_COU | SHAKE `Constrain_Force_Cycle` | `106.224 ms` | 50025 | `2.123 us` |
| DNA_COU | dedicated fixed-light count | `81.825 ms` | 7 | `11.689 ms` |
| DNA_COU | SHAKE coordinate refresh | `74.108 ms` | 50025 | `1.481 us` |

Interpretation:

- wat160k is no longer force-kernel-only. In steady repeated GPU time, builder
  and force are roughly equal. Candidate leaf collect plus fixed-light count are
  the largest builder items, and the per-step pair-shift bit refresh is also
  visible. Further force-only wins need to be large to move e2e.
- DNA_COU with full-dense padding reaches the intended AB-table full-local-dense
  force variant, but the overall bottleneck is more mixed: force is still the
  largest single family, while SHAKE/integration and PME excluded correction
  together exceed builder time.
- The next optimization pass should split by case: wat160k should return to
  candidate leaf/count/pair-shift update reduction, while DNA_COU should not
  judge e2e only from the LJ force kernel because SHAKE and PME excluded
  correction are already major repeated costs.

## Nsys Overall Split, 10000-Step Standard

The 2000-step run above is only a quick shape check. The following 10000-step
run was the standard comparison point for the 2026-07-06 force/builder split,
before the later queue2/cooperative-count plus cached-fill peak env update.

Artifacts:

```text
/tmp/sponge-nsys-overall-10000-20260706/wat160k_peak/wat160k_peak_10000.nsys-rep
/tmp/sponge-nsys-overall-10000-20260706/wat160k_peak/wat160k_peak_10000.sqlite
/tmp/sponge-nsys-overall-10000-20260706/wat160k_peak/wat160k_peak_10000_kern_cuda_gpu_kern_sum.csv
/tmp/sponge-nsys-overall-10000-20260706/wat160k_peak/wat160k_peak_10000_api_cuda_api_sum.csv
/tmp/sponge-nsys-overall-10000-20260706/dna_peak_padding/dna_peak_padding_10000.nsys-rep
/tmp/sponge-nsys-overall-10000-20260706/dna_peak_padding/dna_peak_padding_10000.sqlite
/tmp/sponge-nsys-overall-10000-20260706/dna_peak_padding/dna_peak_padding_10000_kern_cuda_gpu_kern_sum.csv
/tmp/sponge-nsys-overall-10000-20260706/dna_peak_padding/dna_peak_padding_10000_api_cuda_api_sum.csv
```

Runs:

| case | env | step count | status | Calculate_Force | nsys Core Run Speed |
|---|---|---:|---|---:|---:|
| wat160k force-only NVE | then-current stable peak env | 10000 | finite | `6.331339 s` | `123.122169 ns/day` |
| DNA_COU basic NVT | then-current stable peak env plus `SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING=1` | 10000 | finite | `4.448165 s` | `267.975922 ns/day` |

Kernel-time split, excluding one-time initialization kernels:

| case | repeated GPU kernel total | gmxpacked force | gmxpacked builder/update | PME/excluded correction | integration/constraint | bonded/nb14 |
|---|---:|---:|---:|---:|---:|---:|
| wat160k | `5924.541 ms` | `2654.033 ms` (`44.80%`) | `2400.025 ms` (`40.51%`) | `237.706 ms` (`4.01%`) | `286.330 ms` (`4.83%`) | `156.953 ms` (`2.65%`) |
| DNA_COU | `4999.474 ms` | `1724.235 ms` (`34.49%`) | `1113.790 ms` (`22.28%`) | `647.188 ms` (`12.95%`) | `1124.531 ms` (`22.49%`) | `244.782 ms` (`4.90%`) |

Top repeated kernels:

| case | kernel | total | instances | avg |
|---|---|---:|---:|---:|
| wat160k | force-only main gmxpacked force `<..., use_lj_comb=1, full_local_dense=1, ...>` | `2436.357 ms` | 9999 | `243.660 us` |
| wat160k | candidate leaf collect onepass | `790.942 ms` | 31 | `25.514 ms` |
| wat160k | dedicated fixed-light count | `555.767 ms` | 31 | `17.928 ms` |
| wat160k | pair-shift bit refresh | `456.392 ms` | 10032 | `45.494 us` |
| DNA_COU | force-only main gmxpacked force `<..., use_lj_comb=0, full_local_dense=1, ...>` | `1453.835 ms` | 10000 | `145.383 us` |
| DNA_COU | PME excluded correction | `647.188 ms` | 10001 | `64.712 us` |
| DNA_COU | SHAKE `Constrain_Force_Cycle` | `523.697 ms` | 250025 | `2.095 us` |
| DNA_COU | dedicated fixed-light count | `480.394 ms` | 41 | `11.717 ms` |
| DNA_COU | SHAKE coordinate refresh | `372.864 ms` | 250025 | `1.491 us` |

10000-step interpretation:

- wat160k stable peak is still mostly the gmxpacked direct path, but the split is
  not force-only: gmxpacked force is `44.80%` and builder/update is `40.51%` of
  repeated GPU kernel time. The next wat160k work should prioritize candidate
  leaf collect, fixed-light count, and pair-shift refresh along with any force
  kernel improvement.
- DNA_COU with padding reaches the AB-table full-local-dense force shape, but
  repeated GPU time is split across force (`34.49%`), SHAKE/integration
  (`22.49%`), builder/update (`22.28%`), and PME excluded correction
  (`12.95%`). A DNA e2e win from LJ force alone is therefore capped.
- Keep current-mask and current-source patch out of this standard env. They are
  correctness/probe paths, not peak paths.
