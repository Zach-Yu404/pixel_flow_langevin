# onestep_mse — PixelFlowICLR experiment 1 (+1b visuals)

- `all_mse.csv` — raw MSE, 1400 rows = 5 tasks x 7 images x 4 stages x 10 t
  (kept with the task column for completeness; the 5 tasks are bitwise identical —
  one-step prediction never touches the operator/y).
- `raw_mse.json` — one canonical copy of the per-row data (+ note); per-task kw in
  `task_configs.json`.
- `mse_vs_t_summary.png` — THE curve figure (4 stage panels, WLS vs Model,
  per-image thin lines + bold mean). Replaces the former 140 per-(image,stage)
  pngs + 5 overviews.
- `onestep_predictions.mp4` — per-t one-step predictions video: 40 frames =
  4 stages x 10 steps, each frame rows [GT x1 | x_t | WLS | Model] x 7 images,
  per-image MSE under the WLS/Model rows. Produced by onestep_visual.py
  (A100 job) + ffmpeg.

Regenerate: `sbatch ../run_onestep_mse.sbatch`, `sbatch ../run_onestep_visual.sbatch`,
then `python ../consolidate_results.py`; encode: see .research experiment record.
