# 1d_performance_predictors

*Auto-generated from `2_pipeline/1d_performance_predictors/out/` on 2026-09-07 16:45. Re-run `make_progress_md.py` after the script's output changes -- don't hand-edit this file.*

## Tables

### Performance mi vs linear

| protocol | predictor | n_trials | r | r2 | p_linear | mi_bits_raw | mi_bits_shuffle_mean | mi_bits_corrected | mi_p_value | mi_bits_from_r2 | mi_reference_bits | frac_explained_linearly | linear_sufficient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DM | cur_stim_bin | 3045 | 0.1515 | 0.02295 | 4.319e-17 | 0.01416 | 0.0002497 | 0.01391 | 0 | 0.01674 | 0.01674 | 1 | True |
| DM | prev_stim_bin | 3027 | -0.0345 | 0.00119 | 0.05774 | 0.0009082 | 0.0001938 | 0.0007144 | 0.01 | 0.0008589 | 0.0008589 | 1 | True |
| DM | cur_runspeed | 3045 | 0.1607 | 0.02582 | 4.633e-19 | 0.02754 | 0.004411 | 0.02313 | 0 | 0.01887 | 0.02313 | 0.8157 | True |
| DM | cur_pupil_area | 988 | 0.05336 | 0.002848 | 0.09365 | 0.0235 | 0.01488 | 0.00862 | 0.06 | 0.002057 | 0.00862 | 0.2386 | False |
| DM | cur_motionenergy | 2014 | -0.0201 | 0.0004042 | 0.3672 | 0.01553 | 0.006853 | 0.008676 | 0 | 0.0002916 | 0.008676 | 0.03361 | False |
| DM | cur_lick | 3045 | -0.06611 | 0.004371 | 0.0002616 | 0.009481 | 0.0008751 | 0.008606 | 0 | 0.00316 | 0.008606 | 0.3672 | False |
| DM | prev_runspeed | 3027 | 0.1291 | 0.01666 | 1.022e-12 | 0.01842 | 0.004695 | 0.01372 | 0 | 0.01212 | 0.01372 | 0.8829 | True |
| DM | prev_pupil_area | 982 | 0.04882 | 0.002383 | 0.1263 | 0.02465 | 0.01502 | 0.009633 | 0.01 | 0.001721 | 0.009633 | 0.1787 | False |
| DM | prev_motionenergy | 2003 | -0.02849 | 0.0008115 | 0.2025 | 0.009303 | 0.007181 | 0.002122 | 0.18 | 0.0005856 | 0.002122 | 0.276 | False |
| DM | prev_lick | 3027 | -0.06437 | 0.004143 | 0.0003949 | 0.007184 | 0.0006149 | 0.00657 | 0 | 0.002995 | 0.00657 | 0.4558 | False |
| DM | prev_correct | 3027 | 0.2493 | 0.06214 | 4.202e-44 | 0.03057 | 0.0002507 | 0.03032 | 0 | 0.04628 | 0.04628 | 1 | True |
| DM | prev_lickResponse | 3027 | 0.1419 | 0.02012 | 4.475e-15 | 0.01266 | 0.0002945 | 0.01237 | 0 | 0.01466 | 0.01466 | 1 | True |
| DP | cur_stim_bin | 3194 | 0.1932 | 0.03732 | 3.155e-28 | 0.03667 | 0.0007427 | 0.03593 | 0 | 0.02743 | 0.03593 | 0.7636 | False |
| DP | prev_stim_bin | 3176 | -0.009245 | 8.547e-05 | 0.6025 | 0.0001967 | 0.0006715 | 0 | 0.8 | 6.165e-05 | 6.165e-05 | 1 | True |
| DP | cur_runspeed | 3194 | -0.02597 | 0.0006746 | 0.1422 | 0.009254 | 0.004578 | 0.004676 | 0 | 0.0004868 | 0.004676 | 0.1041 | False |
| DP | cur_pupil_area | 1137 | -0.1712 | 0.0293 | 6.305e-09 | 0.03533 | 0.01207 | 0.02326 | 0 | 0.02145 | 0.02326 | 0.9222 | True |
| DP | cur_motionenergy | 2085 | -0.1116 | 0.01245 | 3.252e-07 | 0.03094 | 0.006821 | 0.02412 | 0 | 0.009039 | 0.02412 | 0.3748 | False |
| DP | cur_lick | 3194 | -0.06012 | 0.003614 | 0.0006752 | 0.002063 | 0.0009257 | 0.001137 | 0.05 | 0.002612 | 0.002612 | 1 | True |
| DP | prev_runspeed | 3176 | -0.04404 | 0.00194 | 0.01306 | 0.007671 | 0.004175 | 0.003496 | 0.01 | 0.001401 | 0.003496 | 0.4006 | False |
| DP | prev_pupil_area | 1130 | -0.1661 | 0.02758 | 1.957e-08 | 0.03569 | 0.01251 | 0.02319 | 0 | 0.02018 | 0.02319 | 0.8701 | True |
| DP | prev_motionenergy | 2075 | -0.1077 | 0.01161 | 8.716e-07 | 0.03284 | 0.006475 | 0.02637 | 0 | 0.00842 | 0.02637 | 0.3193 | False |
| DP | prev_lick | 3176 | -0.007104 | 5.046e-05 | 0.689 | 0.0003961 | 0.0008908 | 0 | 0.69 | 3.64e-05 | 3.64e-05 | 1 | True |
| DP | prev_correct | 3176 | 0.07635 | 0.00583 | 1.65e-05 | 0.004097 | 0.0002382 | 0.003859 | 0 | 0.004218 | 0.004218 | 1 | True |
| DP | prev_lickResponse | 3176 | 0.1052 | 0.01106 | 2.814e-09 | 0.007849 | 0.000253 | 0.007596 | 0 | 0.008026 | 0.008026 | 1 | True |
| DN | cur_stim_bin | 6698 | 0.3275 | 0.1073 | 3.227e-167 | 0.1095 | 0.0003282 | 0.1092 | 0 | 0.08184 | 0.1092 | 0.7496 | False |
| DN | prev_stim_bin | 6672 | 0.01058 | 0.0001118 | 0.3878 | 0.003493 | 0.0003122 | 0.003181 | 0 | 8.068e-05 | 0.003181 | 0.02537 | False |
| DN | cur_runspeed | 6698 | 0.002196 | 4.821e-06 | 0.8574 | 0.00743 | 0.00211 | 0.00532 | 0 | 3.478e-06 | 0.00532 | 0.0006537 | False |
| DN | cur_pupil_area | 6212 | -0.03416 | 0.001167 | 0.007084 | 0.009661 | 0.002348 | 0.007313 | 0 | 0.0008424 | 0.007313 | 0.1152 | False |
| DN | cur_motionenergy | 6431 | -0.02451 | 0.0006008 | 0.04936 | 0.006698 | 0.002172 | 0.004525 | 0 | 0.0004335 | 0.004525 | 0.09579 | False |
| DN | cur_lick | 6698 | -0.01938 | 0.0003756 | 0.1127 | 0.0004039 | 0.0002697 | 0.0001343 | 0.21 | 0.000271 | 0.000271 | 1 | True |

*(showing first 30 of 36 rows -- see the linked CSV for the rest)*

[performance_mi_vs_linear.csv](../2_pipeline/1d_performance_predictors/out/performance_mi_vs_linear.csv)

### Performance multivariate mi vs linear

| protocol | predictors | n_predictors | n_trials | r2 | adj_r2 | p_linear | mi_bits_raw | mi_bits_shuffle_mean | mi_bits_corrected | mi_p_value | mi_bits_from_r2 | mi_reference_bits | frac_explained_linearly | linear_sufficient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DM | cur_stim_bin, prev_stim_bin, cur_runspeed, cur_pupil_area, cur_motionenergy, cur_lick, prev_runspeed, prev_pupil_area, prev_motionenergy, prev_lick, prev_correct, prev_lickResponse | 12 | 979 | 0.1683 | 0.158 | 7.089e-32 | 0.09513 | 0.003103 | 0.09203 | 0 | 0.1329 | 0.1329 | 1 | True |
| DP | cur_stim_bin, prev_stim_bin, cur_runspeed, cur_pupil_area, cur_motionenergy, cur_lick, prev_runspeed, prev_pupil_area, prev_motionenergy, prev_lick, prev_correct, prev_lickResponse | 12 | 1129 | 0.1016 | 0.09199 | 5.741e-20 | 0.1009 | 0.008539 | 0.09239 | 0 | 0.07733 | 0.09239 | 0.837 | True |
| DN | cur_stim_bin, prev_stim_bin, cur_runspeed, cur_pupil_area, cur_motionenergy, cur_lick, prev_runspeed, prev_pupil_area, prev_motionenergy, prev_lick, prev_correct, prev_lickResponse | 12 | 6188 | 0.1293 | 0.1276 | 2.085e-175 | 0.1305 | 0.003327 | 0.1272 | 0 | 0.09985 | 0.1272 | 0.7851 | False |

[performance_multivariate_mi_vs_linear.csv](../2_pipeline/1d_performance_predictors/out/performance_multivariate_mi_vs_linear.csv)

## Figures

### Performance overview

![1_performance_overview](../2_pipeline/1d_performance_predictors/out/figures/1_performance_overview.png)

### Tuning current trial

![2_tuning_current_trial](../2_pipeline/1d_performance_predictors/out/figures/2_tuning_current_trial.png)

### Tuning previous trial

![3_tuning_previous_trial](../2_pipeline/1d_performance_predictors/out/figures/3_tuning_previous_trial.png)

### History kernel

![4_history_kernel](../2_pipeline/1d_performance_predictors/out/figures/4_history_kernel.png)

### Mi vs linear summary

![5_mi_vs_linear_summary](../2_pipeline/1d_performance_predictors/out/figures/5_mi_vs_linear_summary.png)

### 5a frac explained linearly

![5a_frac_explained_linearly](../2_pipeline/1d_performance_predictors/out/figures/5a_frac_explained_linearly.png)

### Multivariate mi vs linear summary

![6_multivariate_mi_vs_linear_summary](../2_pipeline/1d_performance_predictors/out/figures/6_multivariate_mi_vs_linear_summary.png)

### 6a multivariate frac explained linearly

![6a_multivariate_frac_explained_linearly](../2_pipeline/1d_performance_predictors/out/figures/6a_multivariate_frac_explained_linearly.png)
