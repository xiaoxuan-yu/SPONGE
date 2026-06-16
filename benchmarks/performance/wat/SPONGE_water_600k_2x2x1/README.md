# SPONGE water large case (Water_600k_2x2x1, 658176 atoms)

This package is prepared for SPONGE testing.

Contents:
- water_2x2x1.top\n- water_npt_eq_2x2x1.gro
- mdin_pme_nve.spg.toml
- mdin_esp_nve.spg.toml

Usage:
- Run PME:
  `SPONGE -mdin mdin_pme_nve.spg.toml`
- Run ESP:
  `SPONGE -mdin mdin_esp_nve.spg.toml`

Notes:
- All paths in the mdin files are relative, so the package is self-contained.
- `step_limit` is set to 1000 for short benchmark-style testing. Adjust it if needed.
- The ESP mdin keeps the current validated settings used in recent SPONGE ESP work.
