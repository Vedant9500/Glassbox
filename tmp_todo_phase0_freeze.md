# Temp TODO — Phase 0 freeze (run later)

## Status
- [ ] Not started

## Why
Official multi-seed multi-tier noise-protocol baseline so later phases have a locked scoreboard.
Parallel cell sweep is already in `scripts/benchmark_noise.py` (`--jobs` / `--omp-num-threads`).

## Command (Ryzen 7 7840HS recommended)
```bash
python scripts/benchmark_noise.py \
  --output-dir results/noise_protocol_baseline \
  --problems Poly-x2,Poly-x3-x,Nguyen-1,Nguyen-5,Keijzer-4,Feynman-I.6.20a \
  --seeds 11,23,47,89,137 \
  --tiers all \
  --n-samples 300 \
  --generations 40 \
  --population-size 60 \
  --timeout 45 \
  --jobs 4 \
  --omp-num-threads 4 \
  --ablation full
```

## Matrix
- 6 problems × 7 tiers × 5 seeds = **210 cells**
- Expected wall time with 4×4: ~30–90 min (depends on early exits)
- Sequential worst case: ~2–3 h

## Artifacts to verify after run
Under `results/noise_protocol_baseline/`:
- `noise_protocol_rows.json`
- `noise_protocol_summary.json`
- `noise_protocol_report.md`
- stamped dated subdir `noise_protocol_YYYYMMDD_HHMMSS/`

## After success
- [ ] Confirm 210 rows (or expected cell count) and no mass failures
- [ ] Mark Phase 0 complete in `noise_handling_phases.md`
- [ ] Delete this temp file when done
