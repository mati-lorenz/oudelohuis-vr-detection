# Nested-parallelism pattern used throughout this project: parallelize
# across SESSIONS as the primary level, and keep the inner (per-neuron,
# per-pair) parallelism at n_jobs=1 UNLESS you only have 1-2 sessions, to
# avoid oversubscribing CPUs with loky-inside-loky.

from joblib import Parallel, delayed

n_jobs_sessions = -1   # bump this
n_jobs_inner = 1        # keep this at 1 when n_jobs_sessions uses most cores


def process_session(session, n_jobs_inner):
    # ... pass n_jobs_inner down to whatever per-neuron/per-pair Parallel
    # call happens inside here ...
    raise NotImplementedError


results = Parallel(n_jobs=n_jobs_sessions, backend="loky", verbose=10)(
    delayed(process_session)(session, n_jobs_inner) for session in []  # TODO: sessions
)
