# gmxpacked DNA force SCI split3 contiguous - 2026-07-23

This experiment continues force-only DNA optimization after the accepted
SCI split2 specialization. The water LJ-combination paths remain frozen.

## Constraints and diagnosis

The implementation remains a generic compile-time AB-table branch. It does
not inspect a system name, atom count, or LJ type count. The gate requires the
same force-only, atom-order, full-local-dense, SCI-shift-split layout as
split2. LJ-combination kernels, including both water systems, reject it.

No texture object or whole AB-table shared-memory cache was added. This follows
the current GROMACS CUDA LJ branch organization: combination-rule kernels keep
per-atom combination parameters, while the non-combination branch fetches the
pair table through the read-only path. The relevant upstream sources are:

- `src/gromacs/nbnxm/cuda/nbnxm_cuda_kernel.cuh`
- `src/gromacs/nbnxm/cuda/nbnxm_cuda_kernel_utils.cuh`

SASS PC sampling on the accepted split2 production kernel showed that the
largest late global dependency was `sorted_atom_ids` before atom-order force
writeback, not the packed AB lookup. Two default-off cache experiments were
therefore tried and rejected:

| experiment | full-NCU result | end-to-end result | decision |
|---|---:|---:|---|
| preload i atom ids into existing shared storage | 104.22 -> 104.00 us | NVT speed -0.32%, force +0.49% | reject |
| one j atom-id load plus subgroup shuffle | 104.22 -> 107.46 us | not run | reject |

The i preload reduced registers from 69 to 64 and long-scoreboard samples, but
moved the dependency to wait/short-scoreboard stalls. The j broadcast reduced
long scoreboard from 28.6% to 22.8%, but wait rose to 33.5% and total duration
regressed 3.1%. This confirms that software caching or shuffle broadcast does
not improve the current cached same-address ID loads.

## Contiguous split3

The earlier interleaved three-CTA replay was fast but failed the force
tolerance:

```text
duration=0.064082 ms
force_tolerance_mismatches=2
force_max_scaled=1.16576148e-5
```

The new variant divides each SCI's `cjpacked` interval into three balanced,
contiguous ranges:

```text
begin = sci_begin + packed_count * work_part / 3
end   = sci_begin + packed_count * (work_part + 1) / 3
```

Contiguous accumulation restored the existing tolerance while retaining more
parallel CTA work:

```text
force_tolerance_mismatches=0
force_max_scaled=7.80157006e-6
```

At 1000 replay iterations, accepted split2 measured 0.086121 ms and
contiguous split3 measured 0.072723 ms, a 15.6% reduction.

Default-off production gate:

```text
SPONGE_CLUSTERED_GMXPACKED_FORCE_SCI_SPLIT3_CONTIGUOUS_PROBE=1
```

When both split2 and split3-contiguous are requested, the force-only AB-table
call selects split3-contiguous. Virial-only and energy+virial calls retain
their independent split2 specializations.

## Full NCU verification

Replay full NCU:

| metric | split2 | contiguous split3 | delta |
|---|---:|---:|---:|
| duration | 93.44 us | 81.86 us | -12.4% |
| achieved occupancy | 30.2% | 35.9% | +5.8 pp |
| eligible warps/cycle | 0.8 | 1.0 | +28% |
| registers/thread | 69 | 68 | -1 |

Production atom-order full NCU:

| metric | split2 | contiguous split3 | delta |
|---|---:|---:|---:|
| duration | 104.22 us | 89.12 us | -14.5% |
| SM throughput | 28.2% | 34.0% | +5.8 pp |
| memory throughput | 24.9% | 30.8% | +5.8 pp |
| achieved occupancy | 30.6% | 37.0% | +6.4 pp |
| eligible warps/cycle | 0.67 | 0.85 | +26% |
| registers/thread | 69 | 69 | unchanged |
| shared memory/block | 2.3 KB | 2.3 KB | unchanged |

The kernel remains latency-bound, but the larger grid exposes enough
independent work to reduce the effect of wait and scoreboard stalls.

## End-to-end

Two independent three-pair, 10000-step DNA NVT runs, combined:

| path | Calculate_Force (s) | wall (s) | speed (ns/day) |
|---|---:|---:|---:|
| split2 | 2.946066 | 4.569813 | 378.226669 |
| contiguous split3 | 2.811207 | 4.453912 | 388.088384 |

Force time falls 4.58%, wall time falls 2.54%, and speed improves 2.61%.
All final temperatures were finite at 296.53-302.18 K.

Three paired 10000-step DNA NPT runs, with the accepted virial and
energy+virial split2 gates enabled on both sides:

| path | Calculate_Force (s) | wall (s) | speed (ns/day) |
|---|---:|---:|---:|
| force split2 | 2.968550 | 4.709244 | 366.976847 |
| force contiguous split3 | 2.875913 | 4.613320 | 374.615244 |

Force time falls 3.12% and speed improves 2.08%. Final temperatures and
densities were finite. Against the previously measured same-ensemble public
SPONGE 2.0 result of 287.456940 ns/day, the new candidate is 30.32% faster.

## Water guardrails

With the new flag requested, both 1000-step NPT water checks retained
`lj_comb=1` and `sci_work_split3_contiguous=0`:

| system | speed (ns/day) | final temperature (K) |
|---|---:|---:|
| wat160k | 142.896133 | 295.38 |
| wat600k | 44.407997 | 296.79 |

The water force kernel specialization is unchanged.

## Decision

Accept the default-off contiguous split3 gate for the DNA peak environment.
Keep the split2 implementation as the fallback and numerical reference. Do not
enable split3-contiguous globally: other AB-table systems must independently
pass the replay force tolerance and end-to-end gates.

Artifacts:

```text
.tmp/dna-split3-contiguous-20260723/payload.sponge_gmxpacked_forceonly.bin
.tmp/dna-split3-contiguous-correctness-20260723
.tmp/dna-split3-contiguous-pairs-20260723
.tmp/dna-split3-contiguous-pairs-repeat-20260723
.tmp/dna-split3-contiguous-npt-pairs-20260723
.tmp/water-split3-contiguous-guardrail1000-20260723
ncu_reports/dna_split3_contiguous_microbench_20260723
ncu_reports/dna_split3_contiguous_production_20260723
```
