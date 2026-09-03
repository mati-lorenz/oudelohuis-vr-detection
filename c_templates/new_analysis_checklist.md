# Checklist for a new analysis script

Every script in this project has converged on roughly this order -- use it
to avoid re-discovering the same bugs (silent bin-collapse, CV leakage,
double-filtering, etc.) that earlier scripts already hit.

- [ ] Load sessions (shallow first if the loader supports it)
- [ ] Apply the anatomical/projection-identity filter BEFORE any computation
- [ ] Apply the QC filter BEFORE any computation (and BEFORE the anatomical
      filter's index alignment breaks -- filter once, keep indices aligned)
- [ ] Group by (area, label) using `celldata_utils.get_area_label` /
      `ordered_groups`, not ad-hoc string matching
- [ ] Pick response/target binning deliberately per variable (equipopulated
      vs equal-width; see `docs/null_hypotheses_framework.md` and
      `src/infotheory/discretize.py`'s notes on tie-collapse)
- [ ] If this makes an "is this real" claim: pick the null that targets the
      SPECIFIC simple explanation you're worried about, not a generic shuffle
- [ ] If any per-fold statistic (scaler, PCA, etc.) is fit inside a
      cross-validation loop: fit it on the TRAIN fold only
- [ ] Save results as CSV (always) + a rendered table/figure (for humans)
- [ ] Save figures via `infotheory.plotting.save_figure` (not raw
      `fig.savefig`), so format/dpi stay consistent
- [ ] Update the step's `README.md` -- Findings if this answered something,
      Open questions if it raised something
