# Pattern for caching an expensive per-session computation using paths.store
# (this SAME script's own scratch space) -- reach for this only when a
# computation is genuinely slow to redo (minutes+).

from infotheory import cache_path, load_or_compute, get_pipeline_paths

paths = get_pipeline_paths(__file__)

def _compute(session):
    raise NotImplementedError  # TODO: the expensive thing

key_params = dict(session_id="TODO", protocol="DN", n_bins=4)  # every param that affects the result
path = cache_path(paths.store, "my_analysis_name", **key_params)
result = load_or_compute(path, lambda: _compute(session=None), force_recompute=False)
