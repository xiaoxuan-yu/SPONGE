# Fixed-Light Slim Count Experiment - 2026-07-03

## Summary

This round added a default-off count-kernel specialization:

```sh
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_SLIM=1
```

The gate only applies to the current fixed-shift/light specialized count path:
gmxpacked direct, outer lifecycle, active view, fixed-shift leaf screening,
one-pass candidate leaves, fill-prune-reuse-light, parallel count accumulation,
parallel fragment emission, and fixed-shift builder specialization.

It is intentionally conservative. It does not change candidate leaf order,
source fragment order, fill semantics, overflow handling, or legacy fallback
dispatch. It is mutually exclusive with the previous candidate-leaf count
metadata path and with the older fixed-light SASS opt gate.

## Implementation

The specialized subgroup count kernel now has an additional template parameter:

```cpp
kFixedLightSlim
```

For the slim variant the subgroup path skips `cluster_extents[cluster_j]` and
the three extent broadcasts for `j` clusters. These values are dead in the
fixed-shift leaf-screened light-fragment path because reach masks already encode
the i/j spatial screening and the count path only needs source fragments/counts.

The selected kernel variant is:

```text
Count_Nbnxm_Payload_From_Candidate_Leaves_Subgroup<true,true,true,true,true,false,true>
```

Trace name:

```text
subgroup-fixed-light-slim
```

## Correctness

Build:

```sh
pixi run -e dev-cuda13 compile
```

Result: passed.

20-step smoke:

```text
/tmp/sponge-fixed-light-slim-20260703/trace1_slim.out
/tmp/sponge-fixed-light-slim-20260703/trace1_slim.err
```

Trace confirmed dispatch:

```text
variant=subgroup-fixed-light-slim fixed_specialized=1 onepass=1 light=1 parallel_accum=1 parallel_fragment_emit=1 count_metadata=0
```

2000-step subgroup verify:

```text
/tmp/sponge-fixed-light-slim-20260703/verify2000_slim.out
/tmp/sponge-fixed-light-slim-20260703/verify2000_slim.err
```

All sampled verify records were zero-mismatch:

```text
flag_mismatch=0 cj_mismatch=0 excl_mismatch=0
row_count_mismatch=0 field_mismatch=0
```

The run stayed finite through step 2000 with final temperature `293.47 K`.

## NCU Comparison

Input:

```text
/tmp/sponge-fixed-light-slim-20260703/mdin_trace1_slim.spg.toml
```

Same binary, same case, first count launch, `ncu --launch-count 1`.

| metric | fixed-light SASS opt | fixed-light slim |
|---|---:|---:|
| duration | `60.948 ms` | `59.978 ms` |
| registers/thread | `72` | `72` |
| theoretical occupancy | `58.33%` | `58.33%` |
| achieved occupancy | `19.11%` | `19.04%` |
| eligible warps/scheduler | `0.12` | `0.12` |
| issued warp/scheduler | `0.11` | `0.11` |
| DRAM throughput | `0.93%` | `0.92%` |
| local memory spilling requests | `0` | `0` |

Artifacts:

```text
/tmp/sponge-fixed-light-slim-20260703/ncu_count_sassopt.log
/tmp/sponge-fixed-light-slim-20260703/ncu_count_slim.log
```

## 10000-Step Sanity

Input:

```text
/tmp/sponge-onepass-capacity-check/mdin_skin11_current_10000.spg.toml
```

Peak env plus:

```sh
SPONGE_CLUSTERED_GMXPACKED_COUNT_FIXED_LIGHT_SLIM=1
```

Result:

| case | speed | `Calculate_Force` | final temperature | stderr |
|---|---:|---:|---:|---:|
| fixed-light slim | `107.187294 ns/day` | `7.300425 s` | `294.78 K` | empty |

Artifact:

```text
/tmp/sponge-fixed-light-slim-20260703/slim_10000.out
/tmp/sponge-fixed-light-slim-20260703/slim_10000.err
```

This is finite, but below the currently recorded stable peak of
`111.908737 ns/day`, so it is not a peak-env improvement.

## Static SASS/PTX Delta

The slim variant removes only a small amount of code:

| metric | fixed-light SASS opt | fixed-light slim |
|---|---:|---:|
| PTX instructions | `10664` | `10655` |
| PTX loads | `1602` | `1597` |
| PTX shfl | `28` | `25` |
| SASS instructions | `16728` | `16680` |
| SASS loads | `1566` | `1563` |
| SASS control ops | `2044` | `2036` |
| SASS collective ops | `64` | `52` |

Artifacts:

```text
/tmp/sponge-fixed-light-slim-20260703/SPONGE.ptx
/tmp/sponge-fixed-light-slim-20260703/count_slim_sm80.sass
/tmp/sponge-fixed-light-slim-20260703/count_peak_sm80.sass
```

## Decision

Keep the gate default-off and do not add it to the peak env. The implementation
is correct in the tested verify path and the NCU single-launch comparison shows
a small improvement, but the 10000-step sanity run does not beat the current
stable peak.

The result also narrows the next target: remaining count cost is not dominated
by the dead `j` extent load/broadcast. Further count work should focus on the
large latency/control-flow region around dedup/source-fragment accumulation, or
on a larger contract change that removes repeated per-cluster work across
collect/count without adding metadata traffic that regresses collect.
