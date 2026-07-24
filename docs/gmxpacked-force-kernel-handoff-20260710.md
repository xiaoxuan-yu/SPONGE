# gmxpacked force-kernel tuning handoff - 2026-07-10

This note is for handing the current optimization line to a fresh session.
It supersedes the 2026-07-04 hotspot note for the current branch state, but it
does not delete the older traceable path. Read this together with:

- `docs/gmxpacked-candidate-leaf-collect-probe-20260704.md`
- `docs/gmxpacked-force-hotspot-followup-20260704.md`
- `docs/gmxpacked-phase-b-force-kernel-handoff-20260630.md`

## 2026-07-23 public-baseline correction

NPT-vs-NVT timing is diagnostic only. Acceptance is now a same-system,
same-ensemble comparison against the public SPONGE 2.0 CUDA package installed
by `/home/youmans/sidereus/P8_short_screening_2ps`.

The three-cycle 10000-step matrix covers DNA_COU and both water systems in NVT
and NPT. Current DNA is faster than public by 12.83% in NVT and 23.81% in NPT;
the water paths have larger margins. Thus DNA is still the primary target
because it has the smallest margin, not because it is currently slower than
the public baseline. Full protocol and results:

```text
docs/gmxpacked-public-baseline-ensemble-benchmark-20260723.md
```

The first DNA NPT-only follow-up partitions the virial specialization across
two CTAs per SCI. It improves the profiled kernel by 21.4% and paired 10000-step
DNA NPT speed by 0.90%, while the final public-baseline margin is 25.40%.
Keep this gate out of NVE/NVT and both water paths:

```text
SPONGE_CLUSTERED_GMXPACKED_VIRIAL_SCI_SPLIT2_PROBE=1
docs/gmxpacked-dna-npt-virial-sci-split2-20260723.md
```

## Repository State

Worktree:

```text
/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked
```

Branch:

```text
opt/gmxpacked-phase-b-force-kernel
```

Latest committed head at handoff:

```text
1e5561e Optimize gmxpacked DNA force and virial paths
```

The worktree is intentionally dirty. Do not assume the current force-kernel
microbench additions are committed.

Dirty tracked files:

```text
SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp
SPONGE/Lennard_Jones_force/clustered_lj.cpp
SPONGE/Lennard_Jones_force/clustered_lj_count_experiments.cpp
SPONGE/Lennard_Jones_force/clustered_lj_count_experiments.h
docs/gmxpacked-candidate-leaf-collect-probe-20260704.md
tools/nbnxm_microbench/nbnxm_microbench.cu
tools/nbnxm_microbench/nbnxm_microbench_snapshot.h
```

New untracked diagnostic source:

```text
tools/nbnxm_microbench/analyze_force_branch_paths.cpp
```

Other untracked artifacts:

```text
ncu_reports/
.tmp/
SPONGE/2026-06-28-174033-builderlj-kernelgmxp.txt
```

The new diagnostic source is useful and should probably be kept, but it is not
part of the normal build. Compile it manually:

```sh
g++ -O3 -std=c++17 tools/nbnxm_microbench/analyze_force_branch_paths.cpp -o /tmp/analyze_force_branch_paths
```

## Current Peak Path

The current production peak stack includes the accepted count/collect/refresh
work plus the new queue2 probe/emit fuse gate. The newly recorded three-system
10000-step result is in:

```text
/tmp/sponge-probe-emit-fuse-e2e10000-20260709
```

Extra gate over the prior peak env:

```text
SPONGE_CLUSTERED_FIXED_SHIFT_CANDIDATE_LEAF_QUEUE2_FUSED=1
```

DNA also used:

```text
SPONGE_CLUSTERED_GMXPACKED_FULL_DENSE_PADDING=1
```

10000-step e2e result:

| system | speed | wall | Calculate_Force | stderr |
|---|---:|---:|---:|---|
| wat160k | 149.064377 ns/day | 5.796733 s | 4.979652 s | empty |
| wat600k | 48.225800 ns/day | 17.917513 s | 15.661341 s | empty |
| dna_cou | 298.562500 ns/day | 5.788312 s | 3.873791 s | AB-table fallback warning only |

Compared with `/tmp/sponge-current-peak-recheck-20260709`, the fused queue2 gate
measured:

| system | previous peak speed | fused speed | delta |
|---|---:|---:|---:|
| wat160k | 144.195648 ns/day | 149.064377 ns/day | +3.38% |
| wat600k | 46.822659 ns/day | 48.225800 ns/day | +3.00% |
| dna_cou | 295.798553 ns/day | 298.562500 ns/day | +0.93% |

This result is already documented at the end of
`docs/gmxpacked-candidate-leaf-collect-probe-20260704.md`.

## 2026-07-10 DNA-first force update

Production dispatch was rechecked before the next experiment. The conclusion
is correct: dna_cou consumes the packed AB-table template instance, while
wat160k and wat600k consume the faster geometric LJ-combination instance. The
same entry symbol in NSYS had hidden this template difference.

The actual DNA production payload is:

```text
/tmp/sponge-force-table-microbench-20260709/dna_cou/table_payload.sponge_gmxpacked_forceonly.bin
```

The previously named DNA production comb snapshot is diagnostic only.

Fresh DNA AB NCU found only 0.56 CTA waves/SM, 15.92% achieved occupancy, and
0.37 eligible warps/scheduler. Partitioning each SCI record interval across
two CTAs reduced the production safe-force instance from 180.06 to 104.26 us
under the final full-NCU repeat (an earlier repeat measured 103.55 us). Three
and four CTAs were faster but exceeded the existing force tolerance and were
rejected.

Accepted default-off DNA gate:

```text
SPONGE_CLUSTERED_GMXPACKED_FORCE_SCI_SPLIT2_PROBE=1
SPONGE_CLUSTERED_GMXPACKED_FORCE_SCI_SPLIT3_CONTIGUOUS_PROBE=1
```

The contiguous split3 follow-up supersedes split2 when both flags are set. It
requires force-only AB-table, full-local-dense, SCI-shift-split dispatch and is
mutually exclusive with compact/float4 output, AB matrix, and writeback probes.
Water gate traces remain `lj_comb=1, sci_work_split2=0,
sci_work_split3_contiguous=0`.

AB/BA production results against the current mainline binary:

| steps | mainline speed | split2 speed | speed delta | force delta |
|---:|---:|---:|---:|---:|
| 10000, four runs each | 320.486397 ns/day | 361.680825 ns/day | +12.85% | -13.95% |
| 50000, two runs each | 324.400711 ns/day | 360.568177 ns/day | +11.15% | -12.91% |

Use the split2 flag together with full-dense padding for DNA peak checks. Keep
it out of the shared water peak environment. Full evidence and artifact paths:
`docs/gmxpacked-dna-force-sci-work-split-probe-20260710.md`.

The 2026-07-23 contiguous split3 refinement restores the force tolerance that
the earlier interleaved split3 failed. Production full NCU improves
104.22 -> 89.12 us, six-pair DNA NVT speed improves 2.61%, and three-pair DNA
NPT speed improves 2.08%. Enable it after split2 in the DNA peak environment.
Full evidence:
`docs/gmxpacked-dna-force-sci-split3-contiguous-20260723.md`.

## Recent Collect Decision

Probe/emit fusion is the accepted collect-side direction for now.

Wat600k 1000-step alternating 3-round test:

```text
/tmp/sponge-probe-emit-fuse-20260709/wat600k
```

| mode | speed mean | speed min-max | wall mean | Calculate_Force mean |
|---|---:|---:|---:|---:|
| queue2 baseline | 45.088894 ns/day | 44.720234-45.273972 | 1.918195 s | 1.710170 s |
| queue2 fused | 46.245899 ns/day | 45.837807-46.667191 | 1.870242 s | 1.655510 s |

Nsys 1000-step kernel split:

| mode | probe | emit | fused | fused scatter | group total |
|---|---:|---:|---:|---:|---:|
| queue2 baseline | 61.771 ms | 42.683 ms | n/a | n/a | 104.454 ms |
| queue2 fused | n/a | n/a | 44.298 ms | 0.707 ms | 45.005 ms |

Keep the other two collect directions as future work, but they are not the next
best path right now:

- finer task granularity beyond root-child/depth-2 tasks;
- real subgroup-parallel traversal rather than only endpoint-screen assistance.

## Current Force Microbench State

The next tuning target has moved to the table/gmxpacked force replay inner loop.
The microbench now has several force replay attribution modes in
`tools/nbnxm_microbench/nbnxm_microbench.cu`:

```text
production-gmxpacked-sorted-force
production-gmxpacked-sorted-force-local-i-mask8
production-gmxpacked-sorted-force-active-i-mask8
production-gmxpacked-sorted-force-dense-noexcl
production-gmxpacked-sorted-force-attr-all-i
production-gmxpacked-sorted-force-attr-no-cutoff
production-gmxpacked-sorted-force-attr-all-i-no-cutoff
production-gmxpacked-sorted-force-no-atomic
production-gmxpacked-sorted-force-sci-split2
production-gmxpacked-sorted-force-sci-split3
production-gmxpacked-sorted-force-sci-split4
```

The replay kernel template now also supports:

- `use_lj_comb=false` for AB-table replay;
- force writeback skipping;
- local 8-bit i-mask variants;
- dense no-exclusion fast path;
- automatic force validation for two-, three-, and four-part SCI replay;
- attribution modes that remove `active_pair`, `cutoff`, or both branches.

Production snapshot dumping was also adjusted so the microbench header records
whether the production run used LJ combination data or the AB matrix path:

```text
header.use_lj_comb
header.lj_type_matrix_stride
```

The main NCU artifacts are under:

```text
ncu_reports/force_table_replay_20260709
ncu_reports/force_inner_mask_20260709
ncu_reports/force_dense_noexcl_20260709
ncu_reports/force_reconv_attr_20260709
ncu_reports/force_dna_ab_20260710/dna_ab_production_sci_split2_dedicated_full.ncu-rep
ncu_reports/force_dna_ab_20260710/dna_ab_production_sci_split2_dedicated_diff.md
```

## Same-day Follow-up

The cutoff preclassification experiments requested by this handoff are recorded
in `docs/gmxpacked-force-cutoff-sidecar-probe-20260710.md`. The best realistic
microbench result is a GPU cutoff-mask producer plus a 64-bit sidecar consumer:
about 9.9% faster on wat600k and 5.7% on wat160k, neutral on the comb-format
DNA diagnostic snapshot, and 45% slower on the actual small DNA AB-table
production payload. Keep it default-off and payload-size gated for any
production experiment.

The water production payloads, DNA comb-format diagnostic payload, and actual
DNA AB-table production payload are, respectively:

```text
/tmp/sponge-collect-distribution-20260708/wat160k_peak/wat160k_builder_footer.sponge_gmxpacked_forceonly.bin
/tmp/sponge-collect-distribution-20260708/wat600k_peak/wat600k_builder_footer.sponge_gmxpacked_forceonly.bin
/tmp/sponge-collect-distribution-20260708/dna_peak_padding/dna_cou_builder_footer.sponge_gmxpacked_forceonly.bin
/tmp/sponge-force-table-microbench-20260709/dna_cou/table_payload.sponge_gmxpacked_forceonly.bin
```

Build the microbench with a local tmpdir because `/tmp` has been close to full:

```sh
TMPDIR=/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked/.tmp/nvcc \
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target NBNXM_MICROBENCH -j
```

## Force Reconvergence Attribution

Wat600k NCU attribution results:

```text
ncu_reports/force_reconv_attr_20260709
```

## DNA NPT virial register/shared follow-up - 2026-07-23

Starting from the default-off DNA virial split2 probe:

- specialize its AB-table `__launch_bounds__` from 64/13 to 64/10;
- retain 96 rather than 72 registers/thread;
- merge the two CTA warps' final per-atom virials through 3.07 KB additional
  shared memory before global atomic output.

Full NCU progression:

| metric | split2 | launch bounds 10 | + shared merge |
|---|---:|---:|---:|
| duration | 192.90 us | 126.91 us | 124.45 us |
| local spilling requests | 2,949,688 | 469,486 | 398,464 |
| registers/thread | 72 | 96 | 96 |
| achieved occupancy | 37.10% | 28.28% | 27.32% |
| eligible warps/scheduler | 0.35 | 0.57 | 0.60 |
| L1 hit rate | 51.16% | 59.18% | 66.78% |

The cumulative kernel improvement is 35.5% and spill requests fall 86.5%.
Three uncontaminated 10000-step DNA NPT pairs show about +1.32% end-to-end
speed from launch bounds. Two clean shared-merge pairs show another 0.75%
reduction in `Calculate_Force`; shared-GPU runs overlapping an external Python
compute job were discarded.

Keep the changes scoped to the existing DNA NPT virial split2 gate. NVE, NVT,
one-part virial, energy, and water LJ-combination specializations remain
unchanged. The final candidate retained a `+25.48%` speed advantage in one
uncontaminated 10000-step pair against public SPONGE 2.0, consistent with the
previous three-pair `+25.40%` result. wat160k and wat600k both completed final
10000-step NPT guardrails; their part-1 kernel resource usage is byte-for-byte
unchanged across the lb10 and shared-merge binaries. Full details and artifacts are in
`docs/gmxpacked-dna-npt-virial-register-shared-20260723.md`.

## DNA energy+virial split2 follow-up - 2026-07-23

The full per-atom energy+virial specialization was the slowest direct-force
variant in the new profile: 342.69 us, 17.83% achieved occupancy, and
6,022,572 local-store spill sectors. A separate default-off gate now
partitions this AB-table SCI work across two CTAs:

```text
SPONGE_CLUSTERED_GMXPACKED_ENERGY_VIRIAL_SCI_SPLIT2_PROBE=1
```

Final full NCU is 193.34 us (`-43.6%`), 29.12% achieved occupancy, 96
registers/thread, and 3,392,467 spill sectors (`-43.7%`). The force, energy,
and virial output layouts and consumers are unchanged.

Existing 2000-step nsys data records only one fast and one slow energy+virial
launch, so the normal sparse-output 10000-step difference is below reliable
end-to-end attribution. With output every 10 steps, three paired DNA NPT runs
improve by 2.33%. The water LJ-combination kernels do not satisfy this gate.
Full details:

```text
docs/gmxpacked-dna-energy-virial-sci-split2-20260723.md
```

Single-kernel raw metrics:

| mode | duration | regs/thread | total inst | active warps/sched | eligible warps/sched |
|---|---:|---:|---:|---:|---:|
| sorted | 3.366528 ms | 70 | 1.951B | 49.73% | 0.905 |
| attr-all-i | 3.971296 ms | 96 | 2.073B | 36.36% | 0.720 |
| attr-no-cutoff | 5.060544 ms | 71 | 4.129B | 49.11% | 2.209 |
| attr-all-i-no-cutoff | 6.214944 ms | 87 | 4.898B | 35.58% | 1.954 |

Source/SASS aggregate:

| mode | BSSY+BSYNC+BRA inst | reconv samples | ISETP+LOP3+PLOP3 inst | mask samples |
|---|---:|---:|---:|---:|
| sorted | 484,785,135 | 122,530 | 226,938,284 | 84,978 |
| attr-all-i | 410,716,949 | 132,984 | 121,460,864 | 94,609 |
| attr-no-cutoff | 414,442,861 | 99,093 | 230,051,778 | 87,457 |
| attr-all-i-no-cutoff | 106,996,350 | 47,003 | 118,347,370 | 104,270 |

Interpretation:

- Removing only `active_pair` reduces mask work but raises register pressure
  sharply and slows the kernel.
- Removing only cutoff removes the cutoff branch but executes force math for far
  too many pairs, so runtime explodes.
- Only removing both branches collapses BSSY/BSYNC strongly, which means the
  expensive reconvergence is caused by the nested active-pair plus cutoff scope.
- This is an attribution result, not an optimization. The branch cannot simply
  be removed.

## Real Payload Branch Mix

The new tool `tools/nbnxm_microbench/analyze_force_branch_paths.cpp` replays the
real force payload on host and classifies each warp-site by the two inner
branches:

- `A0`: `active_pair` all false;
- `A1_C1`: all active and cutoff all pass;
- `A1_C0`: all active and cutoff all fail;
- `A1_Cmix`: all active and cutoff mixed;
- `Amix_*`: `active_pair` itself is mixed.

Run:

```sh
g++ -O3 -std=c++17 tools/nbnxm_microbench/analyze_force_branch_paths.cpp -o /tmp/analyze_force_branch_paths

/tmp/analyze_force_branch_paths \
  wat160k /tmp/sponge-collect-distribution-20260708/wat160k_peak/wat160k_builder_footer.sponge_gmxpacked_forceonly.bin \
  wat600k /tmp/sponge-collect-distribution-20260708/wat600k_peak/wat600k_builder_footer.sponge_gmxpacked_forceonly.bin \
  dna_cou /tmp/sponge-collect-distribution-20260708/dna_peak_padding/dna_cou_builder_footer.sponge_gmxpacked_forceonly.bin
```

Warp-site result:

| system | A0 | A1_C0 | A1_Cmix | Amix_C1 | Amix_Cmix |
|---|---:|---:|---:|---:|---:|
| wat160k | 34.09% | 59.87% | 5.43% | 0.27% | 0.10% |
| wat600k | 34.72% | 59.28% | 5.47% | 0.23% | 0.12% |
| dna_cou | 33.26% | 60.57% | 5.57% | 0.26% | 0.12% |

Dynamic lane result:

| system | inactive | active + cutoff pass | active + cutoff fail |
|---|---:|---:|---:|
| wat160k | 34.21% | 2.29% | 63.50% |
| wat600k | 34.83% | 2.16% | 63.01% |
| dna_cou | 33.38% | 2.31% | 64.32% |

Conclusion:

- All branch combinations occur in real payloads.
- `active_pair` mixed exists, but is small and mostly exclusion-driven.
- The larger structural issue is cutoff: most active lanes fail cutoff, and
  about 5.5% of warp-sites are cutoff-mixed.
- A naive path split is risky because extra control flow or empty launches can
  erase the benefit. Any split should be driven by a payload redesign, not just
  by inserting more runtime branches.

The dna_cou run reports a tiny OOB skip because that snapshot has tail padding
shape differences; the skipped count is negligible for the above percentages.

## Current Read On Optimization Direction

The next useful force-kernel optimization should not start with reordering
atoms or records. That is likely to introduce a large memory cost and makes
correctness risk high.

Better candidates:

1. Build a payload-side preclassification for force replay.

   The real payload has many `A0` and `A1_C0` warp-sites. If builder/refresh can
   cheaply tag a `cjpacked split/JM/I` site as all-inactive or all-cutoff-fail
   using existing geometry, the force kernel can avoid entering nested
   reconvergence for those cases. This needs a combined payload design: the
   classification must be produced where candidate/inner-active information is
   already being built or refreshed.

2. Try a cutoff-first screening shape without per-lane force math.

   The attribution shows that removing cutoff is impossible because too much
   force math executes. A viable experiment would compute a compact predicate
   or all-fail/all-pass classification before the expensive math and before the
   nested `active_pair` branch region. It must be measured against register
   pressure.

3. Keep path-split experiments inside the microbench until the payload contract
   is clear.

   A separate fast path for dense/no-exclusion or cutoff-uniform records is only
   promising if the builder can route a large enough fraction of work without
   adding per-step host gaps or extra empty kernels.

4. Keep AB-table/per-atom energy/virial replay connected.

   The current microbench now supports table payload inspection and per-atom
   energy/virial replay variants. This is important because dna_cou commonly
   exercises AB-table fallback, so force-only wat600k is not enough coverage.

## Things Already Tried Or Rejected

- Dense no-exclusion fast path is useful for attribution, but the real payload
  still has exclusion and cutoff structure. Do not promote it directly.
- `active_pair` removal by force-all-i regresses from register pressure and
  extra executed work.
- `cutoff` removal greatly increases force math and is not viable.
- Atom or record reordering may create large memory movement cost and should
  not be the first branch-reduction strategy.
- Earlier sorted/float4 force support was a single-kernel win but an e2e
  regression before the current collect/refresh stack; do not default-enable
  older sorted-force envs without a fresh e2e gate.

## Suggested Next Session Plan

1. Rebuild and reproduce the DNA AB split2 result.

   Use the 17,951-record AB-table payload, not the comb-format diagnostic DNA
   snapshot. Confirm zero tolerance mismatches and about 0.080 ms normal replay
   time before editing.

2. Start the next full NCU iteration from split2.

   The production kernel remains latency-bound at 103.55 us, 30.6% achieved
   occupancy, and about 0.7 eligible warps/cycle. Analyze the two work parts
   separately for record-count and active-pair imbalance before changing the
   partition.

3. Prefer an AB lookup-latency experiment that preserves split2 arithmetic.

   DNA has only 17 atom types. Test cooperative staging of the 153-entry packed
   triangular AB table into shared memory in the replay microbench. Measure
   load overhead, long-scoreboard stalls, registers, and L1 behavior. The
   row-major global matrix has already been rejected.

4. Treat adaptive parts/SCI as a later side-path experiment.

   Parts 3 and 4 exceeded the current force tolerance. Do not relax that
   tolerance. If adaptive splitting is tested, keep two parts for the normal
   population and add extra parts only for proven heavy SCI, with an automatic
   force comparison on every replay.

5. Keep production acceptance DNA-first with water guardrails.

   Require full NCU, paired NSYS, AB/BA dna_cou 10000-step and 50000-step checks.
   wat160k and wat600k only need to prove `lj_comb=1, sci_work_split2=0` and
   finite 10000-step completion unless a change intentionally touches their
   LJ-combination path.

## Useful Commands

Build:

```sh
TMPDIR=/home/youmans/sidereus/SPONGE-mainline-nbnxm-gmxpacked/.tmp/nvcc \
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target NBNXM_MICROBENCH -j
```

Run branch mix analyzer:

```sh
g++ -O3 -std=c++17 tools/nbnxm_microbench/analyze_force_branch_paths.cpp -o /tmp/analyze_force_branch_paths
```

Use CUDA tools through pixi:

```sh
pixi run -e dev-cuda13 ncu ...
pixi run -e dev-cuda13 nsys ...
```

When compiling CUDA, keep `TMPDIR` under the repo `.tmp/nvcc` to avoid `/tmp`
space failures.

## Artifact Index

Peak and collect:

```text
/tmp/sponge-current-peak-recheck-20260709
/tmp/sponge-current-peak-alt5-20260709
/tmp/sponge-current-peak-nsys-20260709
/tmp/sponge-probe-emit-fuse-20260709
/tmp/sponge-probe-emit-fuse-e2e10000-20260709
/tmp/sponge-probe-emit-ncu-wat600k-20260709
```

Skin and refresh:

```text
/tmp/sponge-current-peak-skin-sweep-wat160k-20260709
/tmp/sponge-current-peak-skin-confirm-wat160k-20260709
/tmp/sponge-refresh-ncu-20260709
/tmp/sponge-next-ncu-wat600k-20260709
```

Force/table:

```text
/tmp/sponge-force-table-microbench-20260709
/tmp/sponge-mainline-compare-dna-cou-20260709
/tmp/sponge-mainline-compare-dna-cou-long-20260709
ncu_reports/force_table_replay_20260709
ncu_reports/force_inner_mask_20260709
ncu_reports/force_dense_noexcl_20260709
ncu_reports/force_reconv_attr_20260709
```
