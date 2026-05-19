# FEP + REST2 Manager Fixture

This fixture is a four-schedule FEP test system used by the REST2 manager
benchmark. It was copied from `/mnt/data8t/Data/FEP_test_for_REMD`.

Only the SPONGE runtime text inputs are stored here:

- `0/TMP_*.txt`
- `1/TMP_*.txt`
- `2/TMP_*.txt`
- `3/TMP_*.txt`
- `step2_mdin.txt`

Large auxiliary `TMP.mol2` files and previous output files are intentionally
excluded because the benchmark reads the SPONGE `TMP_*.txt` files directly.

The benchmark uses four soft-core LJ lambdas:

- schedule 0: `lambda_lj = 0.0`
- schedule 1: `lambda_lj = 0.1`
- schedule 2: `lambda_lj = 0.2`
- schedule 3: `lambda_lj = 0.3`

REST2 is enabled with the first 55 atoms treated as the hot region.
