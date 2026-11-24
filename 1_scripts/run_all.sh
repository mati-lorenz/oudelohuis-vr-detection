#!/usr/bin/env bash
# run_all.sh -- all the 2x and 3x steps in order (2a ... 2j, then 3a ... 3l and 3a2), for an overnight run on few cores.
#
#   bash run_all.sh               2 sessions in parallel (default)
#   bash run_all.sh 1             one worker (lowest RAM)
#   bash run_all.sh 2 --aggregate mouse       extra arguments go to EVERY script
#   nohup bash run_all.sh 1 > run_all.log 2>&1 &         (survives closing the terminal)
#   STAGES="3a 3b 3c" bash run_all.sh 2     only these steps (the numbers/letters in the SCRIPTS list below)
#
# Run it from the project root (the folder with 1_scripts/ and 2_pipeline/). Also works if started from inside it.
# * Pairs: 3a-3l use 200 pairs per (session, area, label) (MAX_PAIRS_PER_GROUP at the top of each script). 2c (between-area MI) and
#   3a2 use PAIRS=200 pairs per PAIR OF GROUPS per session, passed as --max-pairs. A pair cap does not move the results to
#   out/selection/ (only --max-cells / --animals / --sessions do), so the next steps find them in out/.
# * Dependencies: a step is skipped when a step it needs failed (2a/2b -> everything after; 2c -> 2d -> 2e; 3a -> 3b, 3c).
#   A failing step never stops the others. One log per step in run_all_logs/. Failures are listed at the end (exit code 1).
# * Re-running skips finished sessions (per-session caches in each script's store/).
# * Memory: one session per worker; BLAS / OpenMP threads limited to 1, so n_jobs is the only parallelism.
# * Not included: 1x (behavior) and 2b's heavy parts can be switched off with extra args, e.g. --no-outcome --no-raster.
set -u
N_JOBS="${1:-2}"
[ "$#" -ge 1 ] && shift
EXTRA=("$@")
PAIRS=200

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")" 2>/dev/null
[ -d 1_scripts ] || cd ..
[ -d 1_scripts ] || { echo "run it from the project root (the folder with 1_scripts/)"; exit 1; }

# id | script | extra arguments of this step | steps it needs (ids, space separated)
STEPS=(
  "2a|2a_cell_distribution||"
  "2b|2b_activity_statistics||2a"
  "2c|2c_information|--max-pairs $PAIRS|2a 2b"
  "2d|2d_linear_encod||2c"
  "2e|2e_nonlinear_encod||2d"
  "2f|2f_lag_information||2a 2b"
  "2g|2g_behavior_correlation||2a 2b"
  "2h|2h_position_information||2a 2b"
  "2j|2j_pupil_activity_shape||2a 2b"
  "3a|3a_pair_activity_statistics||2a 2b"
  "3b|3b_pair_information_breakdown||3a"
  "3c|3c_pair_information_pid||3a"
  "3d|3d_pair_activity_statistics_temporal||2a 2b"
  "3e|3e_pair_information_breakdown_temporal||2a 2b"
  "3f|3f_pair_information_pid_temporal||2a 2b"
  "3g|3g_pair_hit_miss_correlations||2a 2b"
  "3h|3h_pair_correlations_strategies||2a 2b"
  "3i|3i_pair_behavior_correlations||2a 2b"
  "3j|3j_pair_behavior_information||2a 2b"
  "3k|3k_pair_behavior_breakdown||2a 2b"
  "3l|3l_pair_behavior_pid||2a 2b"
)
STAGES="${STAGES:-}"

LOGDIR=run_all_logs
mkdir -p "$LOGDIR"
declare -A BAD
SUMMARY=()
START=$(date +%s)
echo "== run_all: --n-jobs $N_JOBS, pair cap $PAIRS, extra args: ${EXTRA[*]:-none}  ($(date))"
for step in "${STEPS[@]}"; do
  IFS='|' read -r id script sargs needs <<< "$step"
  if [ -n "$STAGES" ] && [[ " $STAGES " != *" $id "* ]]; then continue; fi
  blocked=""
  for n in $needs; do [ "${BAD[$n]:-}" = 1 ] && blocked="$blocked $n"; done
  if [ -n "$blocked" ]; then
    echo "-- $id $script: skipped (needs$blocked, which failed)"; BAD[$id]=1; SUMMARY+=("$id (skipped)"); continue
  fi
  t0=$(date +%s)
  echo "-- $id $script  started $(date +%H:%M:%S)"
  # shellcheck disable=SC2086
  python "1_scripts/$script.py" --n-jobs "$N_JOBS" $sargs "${EXTRA[@]}" > "$LOGDIR/$script.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then
    echo "   ok   ($((dt / 60)) min)"
  else
    echo "   FAILED (exit $rc, $((dt / 60)) min) -- $LOGDIR/$script.log"; tail -n 3 "$LOGDIR/$script.log" | sed 's/^/   | /'
    BAD[$id]=1; SUMMARY+=("$id")
  fi
done
echo "== done in $(( ($(date +%s) - START) / 60 )) min"
if [ ${#SUMMARY[@]} -gt 0 ]; then echo "== failed / skipped: ${SUMMARY[*]}"; exit 1; fi
echo "== all steps finished"
