# 1b_psychometric

*Auto-generated from `2_pipeline/1b_psychometric/out/` on 2026-09-04 15:01. Re-run `make_progress_md.py` after the script's output changes -- don't hand-edit this file.*

## Tables

### Excluded sessions

| session_id          | protocol   | exclude_reasons                                                                                                         |
|:--------------------|:-----------|:------------------------------------------------------------------------------------------------------------------------|
| LPE10884_2024_01_18 | DN         | FA rate (engaged) = 57% > 50%                                                                                           |
| LPE11622_2024_02_21 | DN         | threshold (mu=22%) outside tested stimulus range (z-range [-1.93, -0.82] doesn't bracket 0)                             |
| LPE11622_2024_02_27 | DN         | threshold (mu=17%) outside tested stimulus range (z-range [-4.28, -0.59] doesn't bracket 0)                             |
| LPE11997_2024_04_12 | DN         | threshold (mu=16%) outside tested stimulus range (z-range [-5.32, -0.32] doesn't bracket 0)                             |
| LPE11998_2024_04_23 | DN         | threshold (mu=16%) outside tested stimulus range (z-range [-0.85, -0.08] doesn't bracket 0)                             |
| LPE11998_2024_04_26 | DN         | FA rate (engaged) = 64% > 50%; threshold (mu=9%) outside tested stimulus range (z-range [0.09, 1.76] doesn't bracket 0) |
| LPE11998_2024_04_30 | DN         | FA rate (engaged) = 56% > 50%                                                                                           |
| LPE12013_2024_04_23 | DN         | threshold (mu=28%) outside tested stimulus range (z-range [-1.23, -0.54] doesn't bracket 0)                             |
| LPE12013_2024_04_24 | DN         | threshold (mu=33%) outside tested stimulus range (z-range [-1.83, -1.04] doesn't bracket 0)                             |
| LPE12385_2024_06_26 | DN         | threshold (mu=6%) outside tested stimulus range (z-range [0.22, 2.20] doesn't bracket 0)                                |
| LPE11495_2024_02_21 | DP         | FA rate (engaged) = 64% > 50%                                                                                           |
| LPE11623_2024_04_01 | DP         | d' (engaged) = 0.71 < 1.0                                                                                               |
| LPE11997_2024_04_08 | DP         | FA rate (engaged) = 56% > 50%                                                                                           |

[excluded_sessions.csv](../2_pipeline/1b_psychometric/out/excluded_sessions.csv)

### Psychometric fits

| session_id          | protocol   |   n_trials_engaged |   dprime_engaged |   fa_rate_engaged |   frac_engaged |   n_signal_levels | fit_status             |        mu |     sigma |   lapse_rate |   guess_rate |         r2 | included   | exclude_reasons                                                                             |
|:--------------------|:-----------|-------------------:|-----------------:|------------------:|---------------:|------------------:|:-----------------------|----------:|----------:|-------------:|-------------:|-----------:|:-----------|:--------------------------------------------------------------------------------------------|
| LPE10884_2023_11_30 | DM         |                157 |          2.64401 |         0.421053  |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE10884_2023_12_10 | DM         |                122 |          1.08576 |         0.458333  |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11081_2023_12_11 | DM         |                104 |          2.36324 |         0.285714  |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11081_2023_12_12 | DM         |                 98 |          2.75917 |         0.210526  |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11495_2024_02_14 | DM         |                147 |          3.21094 |         0.137931  |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11495_2024_02_16 | DM         |                169 |          2.85279 |         0.0882353 |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11622_2024_02_16 | DM         |                353 |          2.94075 |         0.128571  |       0.983287 |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11623_2024_03_27 | DM         |                 87 |          1.28495 |         0.294118  |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11623_2024_03_29 | DM         |                118 |          1.62483 |         0.375     |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11997_2024_04_05 | DM         |                179 |          2.19681 |         0.194444  |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11997_2024_04_07 | DM         |                209 |          2.67331 |         0.214286  |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11998_2024_04_11 | DM         |                149 |          2.93618 |         0.0666667 |       0.818681 |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE11998_2024_04_12 | DM         |                194 |          2.65285 |         0.128205  |       0.885845 |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE12013_2024_04_11 | DM         |                235 |          2.59721 |         0.152174  |       0.983264 |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE12013_2024_04_12 | DM         |                174 |          2.44175 |         0.114286  |       0.824645 |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE12223_2024_05_29 | DM         |                176 |          2.06568 |         0.194444  |       0.93617  |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE12223_2024_05_30 | DM         |                210 |          3.44599 |         0.0238095 |       0.963303 |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE12385_2024_05_30 | DM         |                164 |          2.22232 |         0.1875    |       1        |                 2 | not_applicable         | nan       | nan       | nan          |   nan        | nan        | True       | nan                                                                                         |
| LPE10884_2024_01_12 | DN         |                209 |          2.0559  |         0.190476  |       0.72069  |                13 | ok                     |  10.9745  |   2.47972 |   0.121282   |     0.228571 |   0.802815 | True       | nan                                                                                         |
| LPE10884_2024_01_17 | DN         |                258 |          2.17454 |         0.458333  |       0.791411 |                13 | ok                     |  11.0009  |   5.07759 |   0.0153846  |     0.467547 |   0.732981 | True       | nan                                                                                         |
| LPE10884_2024_01_18 | DN         |                146 |          1.94803 |         0.571429  |       0.715686 |                13 | ok                     |  12.5306  |   4.9055  |   0.00537187 |     0.522981 |   0.494519 | False      | FA rate (engaged) = 57% > 50%                                                               |
| LPE11495_2024_02_22 | DN         |                204 |          2.48705 |         0.428571  |       1        |                13 | ok                     |   6.34809 |   7.03854 |   0.016989   |     0.332857 |   0.45185  | True       | nan                                                                                         |
| LPE11622_2024_02_20 | DN         |                396 |          1.90834 |         0.288889  |       0.609231 |                13 | ok                     |  11.8091  |   9.76983 |   0.0891792  |     0.221111 |   0.400684 | True       | nan                                                                                         |
| LPE11622_2024_02_21 | DN         |                438 |          2.23291 |         0.22449   |       0.551637 |                13 | threshold_out_of_range |  22.4264  |   9.03388 |   0.0700056  |     0.200812 |   0.715746 | False      | threshold (mu=22%) outside tested stimulus range (z-range [-1.93, -0.82] doesn't bracket 0) |
| LPE11622_2024_02_22 | DN         |                438 |          2.08545 |         0.235294  |       0.659639 |                13 | ok                     |  13.4272  |   2       |   0.0854569  |     0.192467 |   0.939553 | True       | nan                                                                                         |
| LPE11622_2024_02_23 | DN         |                246 |          2.38457 |         0.222222  |       0.44086  |                13 | ok                     |   9.68743 |   3.79936 |   0.0507043  |     0.222968 |   0.81953  | True       | nan                                                                                         |
| LPE11622_2024_02_26 | DN         |                340 |          2.12937 |         0.289474  |       0.585198 |                13 | ok                     |  14.3509  |   6.48357 |   0.0573438  |     0.281247 |   0.721397 | True       | nan                                                                                         |
| LPE11622_2024_02_27 | DN         |                336 |          2.30039 |         0.111111  |       0.518519 |                13 | threshold_out_of_range |  16.6132  |   2.71384 |   0.140224   |     0.133333 |   0.872378 | False      | threshold (mu=17%) outside tested stimulus range (z-range [-4.28, -0.59] doesn't bracket 0) |
| LPE11623_2024_04_02 | DN         |                160 |          2.07786 |         0.1875    |       1        |                13 | ok                     |  15.726   |   5.14357 |   0.115628   |     0.182711 |   0.639253 | True       | nan                                                                                         |
| LPE11997_2024_04_09 | DN         |                277 |          1.26714 |         0.4       |       0.6925   |                13 | ok                     |   8.59412 |   9.3155  |   0.155938   |     0.31     |   0.624643 | True       | nan                                                                                         |
*(showing first 30 of 75 rows -- see the linked CSV for the rest)*


[psychometric_fits.csv](../2_pipeline/1b_psychometric/out/psychometric_fits.csv)

## Figures

### Psychometric fits DN page1

![1_psychometric_fits_DN_page1](../2_pipeline/1b_psychometric/out/figures/1_psychometric_fits_DN_page1.png)

### Psychometric fits DN page2

![1_psychometric_fits_DN_page2](../2_pipeline/1b_psychometric/out/figures/1_psychometric_fits_DN_page2.png)

### Psychometric fits DP

![1_psychometric_fits_DP](../2_pipeline/1b_psychometric/out/figures/1_psychometric_fits_DP.png)

### Psychometric population DN

![2_psychometric_population_DN](../2_pipeline/1b_psychometric/out/figures/2_psychometric_population_DN.png)

### Dataset overview clean

![3_dataset_overview_clean](../2_pipeline/1b_psychometric/out/figures/3_dataset_overview_clean.png)

### Trial outcomes clean

![4_trial_outcomes_clean](../2_pipeline/1b_psychometric/out/figures/4_trial_outcomes_clean.png)

### Dprime criterion clean

![5_dprime_criterion_clean](../2_pipeline/1b_psychometric/out/figures/5_dprime_criterion_clean.png)

### Signal level balance clean

![6_signal_level_balance_clean](../2_pipeline/1b_psychometric/out/figures/6_signal_level_balance_clean.png)
