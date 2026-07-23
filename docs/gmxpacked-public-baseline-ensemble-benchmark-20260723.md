# gmxpacked public SPONGE ensemble baseline - 2026-07-23

This note corrects the acceptance criterion for the NVT/NPT optimization line.
NPT is expected to cost more than NVT; that cross-ensemble difference is useful
for attribution only. Performance acceptance must compare the same system,
ensemble, step count, and physical input against public SPONGE.

## Public baseline

The reference executable is the installed public package used by
`/home/youmans/sidereus/P8_short_screening_2ps`:

```text
sponge-cuda12 == 2.0.0_alpha
SPONGE v2.0.0 2026-02-16 Spring Festive
SHA-256: 31fc094963eda85bd59e6e43ebf16b0372715f0e139b4fe4ad8fc083c99c57f9
```

Executable:

```text
/home/youmans/sidereus/P8_short_screening_2ps/.pixi/envs/default/bin/SPONGE
```

## Protocol

- RTX 4090, one process, no profiler wrapper.
- DNA_COU, water 160k, and water 600k.
- NVT and NPT, 10000 steps, three paired cycles.
- Public/current order and NVT/NPT order alternate by cycle.
- Force-only direct calculation: `[PM] MPI_size = 0`.
- Current DNA uses full-dense padding and the accepted AB-table `split2` gate.
- Current water uses the accepted LJ-combination fast path.
- NPT uses middle Langevin plus Andersen barostat, 1 bar target, update interval
  10, tau 1.0, and compressibility `4.5e-5`.

Public SPONGE cannot load the water benchmark's GROMACS `[ settles ]` topology.
The water systems were therefore expanded to native SPONGE files. Both
executables consumed the same native files. The conversion preserves GRO
coordinates and velocities, box, TIP3P mass/charge/LJ, residue boundaries,
exclusions, three rigid constraints, and the three zero-force-constant bonds
created by the current GROMACS loader. At 20 steps, public native, current
native, and current GROMACS input energies agree within floating-point
rounding.

## Results

Mean and sample standard deviation over three runs:

| system | ensemble | implementation | Calculate_Force (s) | wall (s) | speed (ns/day) |
|---|---|---|---:|---:|---:|
| DNA_COU | NVT | public | 3.488948 +/- 0.063344 | 5.299456 +/- 0.143776 | 326.265859 +/- 8.959632 |
| DNA_COU | NVT | current | 2.999026 +/- 0.041058 | 4.695727 +/- 0.086845 | 368.115509 +/- 6.859375 |
| DNA_COU | NPT | public | 3.536352 +/- 0.029074 | 6.122513 +/- 0.118635 | 282.336721 +/- 5.529778 |
| DNA_COU | NPT | current | 3.168330 +/- 0.034664 | 4.944806 +/- 0.079841 | 349.552999 +/- 5.612615 |
| water 160k | NVT | public | 9.100652 +/- 0.085945 | 10.448780 +/- 0.192889 | 82.716326 +/- 1.542231 |
| water 160k | NVT | current | 4.967418 +/- 0.077898 | 5.819734 +/- 0.134457 | 148.528737 +/- 3.474150 |
| water 160k | NPT | public | 9.371943 +/- 0.061138 | 11.746717 +/- 0.246970 | 73.581769 +/- 1.565775 |
| water 160k | NPT | current | 5.022816 +/- 0.067479 | 5.989349 +/- 0.141473 | 144.324870 +/- 3.452582 |
| water 600k | NVT | public | 35.872959 +/- 0.317010 | 40.097712 +/- 0.631600 | 21.553065 +/- 0.337598 |
| water 600k | NVT | current | 15.353690 +/- 0.151093 | 18.315006 +/- 0.303816 | 47.187855 +/- 0.787176 |
| water 600k | NPT | public | 36.669844 +/- 0.328100 | 43.813321 +/- 0.622063 | 19.724671 +/- 0.281805 |
| water 600k | NPT | current | 15.762254 +/- 0.177441 | 18.994284 +/- 0.333169 | 45.501160 +/- 0.790250 |

Current relative to public:

| system | ensemble | force time | wall time | speed |
|---|---|---:|---:|---:|
| DNA_COU | NVT | -14.04% | -11.39% | +12.83% |
| DNA_COU | NPT | -10.41% | -19.24% | +23.81% |
| water 160k | NVT | -45.42% | -44.30% | +79.56% |
| water 160k | NPT | -46.41% | -49.01% | +96.14% |
| water 600k | NVT | -57.20% | -54.32% | +118.94% |
| water 600k | NPT | -57.02% | -56.65% | +130.68% |

All 36 final temperatures were finite. NPT final pressure is a trajectory
sample, not an equilibrium acceptance metric for these short runs.

## Decision

- Keep the accepted NVE path unchanged.
- The claim that DNA force makes current end-to-end performance slower than the
  public baseline is false for this locked configuration.
- DNA remains the primary tuning system because its public-baseline margin is
  the smallest. The two water systems remain required regression guardrails.
- NPT-vs-NVT deltas may localize barostat/virial cost but must not be used as
  the optimization acceptance criterion.
- Continue DNA NPT work from the profiled virial force specialization. Its
  register/local-memory latency is the next single-kernel target, while the
  public same-ensemble comparison remains the final gate.

## Artifacts

```text
.tmp/public-comparison-ensemble-matrix-20260723/results.tsv
.tmp/run_public_comparison_ensemble_matrix_20260723.sh
.tmp/analyze_public_comparison_ensemble_20260723.py
.tmp/convert_water_gro_to_native_20260723.py
.tmp/water-native-20260723/
```
