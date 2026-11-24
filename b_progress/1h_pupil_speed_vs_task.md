# 1h_pupil_speed_vs_task

*Auto-generated from `2_pipeline/1h_pupil_speed_vs_task/out/` on 2026-11-17 13:25. Re-run `make_progress_md.py` after the script's output changes -- don't hand-edit this file.*

## Figures

### Pupil speed vs signal quantile

![1_pupil_speed_vs_signal_quantile](../2_pipeline/1h_pupil_speed_vs_task/out/figures/1_pupil_speed_vs_signal_quantile.png)

### Pupil speed vs stim strategy

![2_pupil_speed_vs_stim_strategy](../2_pipeline/1h_pupil_speed_vs_task/out/figures/2_pupil_speed_vs_stim_strategy.png)

### Pupil speed vs performance

![3_pupil_speed_vs_performance](../2_pipeline/1h_pupil_speed_vs_task/out/figures/3_pupil_speed_vs_performance.png)

### Pupil speed vs signal by outcome

![4_pupil_speed_vs_signal_by_outcome](../2_pipeline/1h_pupil_speed_vs_task/out/figures/4_pupil_speed_vs_signal_by_outcome.png)

## Tables

### Binning used

| protocol | stim_strategies | plotted_strategy | strategy_definitions |
| --- | --- | --- | --- |
| DM | zero_vs_nonzero | zero_vs_nonzero | {"zero_vs_nonzero": {"edges": [0.0], "kind": "contrast", "names": ["0", ">0"], "zero_separate": false}} |
| DP | zero_vs_nonzero | zero_vs_nonzero | {"zero_vs_nonzero": {"edges": [0.0], "kind": "contrast", "names": ["0", ">0"], "zero_separate": false}} |
| DN | zero_to80_above80, pct_0-20_20-80_80-100, zero_vs_nonzero | pct_0-20_20-80_80-100 | {"pct_0-20_20-80_80-100": {"edges": [0.2, 0.8], "kind": "curve_pct", "names": ["0-20%", "20-80%", "80-100%"], "zero_separate": false}, "zero_to80_above80": {"edges": [0.8], "kind": "curve_pct", "names": ["0", "<=80%", ">80%"], "zero_separate": true}, "zero_vs_nonzero": {"edges": [0.0], "kind": "contrast", "names": ["0", ">0"], "zero_separate": false}} |

[binning_used.csv](../2_pipeline/1h_pupil_speed_vs_task/out/binning_used.csv)

### Performance comparison group tests  session

| protocol | window | y | n_units | mean_diff | p | p_holm |
| --- | --- | --- | --- | --- | --- | --- |
| DM | pre_stim | pupil_area | 6 | 0.1658 | 0.4375 | 1 |
| DM | pre_stim | runspeed | 10 | 4.035 | 0.001953 | 0.005859 |
| DM | reward | pupil_area | 6 | -0.08271 | 1 | 1 |
| DM | reward | runspeed | 10 | -3.524 | 0.003906 | 0.007812 |
| DM | stim | pupil_area | 6 | -0.05072 | 1 | 1 |
| DM | stim | runspeed | 10 | 0.3735 | 0.625 | 0.625 |
| DN | pre_stim | pupil_area | 25 | -0.1189 | 0.001625 | 0.003251 |
| DN | pre_stim | runspeed | 26 | 0.7046 | 0.05585 | 0.05585 |
| DN | reward | pupil_area | 25 | -0.2764 | 4.172e-07 | 1.252e-06 |
| DN | reward | runspeed | 26 | -13.67 | 2.98e-08 | 8.941e-08 |
| DN | stim | pupil_area | 25 | -0.1211 | 0.003419 | 0.003419 |
| DN | stim | runspeed | 26 | -2.072 | 0.02197 | 0.04394 |
| DP | pre_stim | pupil_area | 7 | -0.1082 | 0.2188 | 0.2188 |
| DP | pre_stim | runspeed | 10 | 0.8477 | 0.08398 | 0.168 |
| DP | reward | pupil_area | 7 | -0.2184 | 0.03125 | 0.0625 |
| DP | reward | runspeed | 10 | -6.396 | 0.001953 | 0.005859 |
| DP | stim | pupil_area | 7 | -0.1587 | 0.01562 | 0.04688 |
| DP | stim | runspeed | 10 | -0.8737 | 0.5566 | 0.5566 |

[performance_comparison_group_tests__session.csv](../2_pipeline/1h_pupil_speed_vs_task/out/performance_comparison_group_tests__session.csv)

### Performance comparison per unit  session

| protocol | window | y | unit | animal_id | n_correct | n_error | mean_correct | mean_error | diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DM | pre_stim | pupil_area | LPE11623_2024_03_27 | LPE11623 | 66 | 21 | 0.5731 | 0.4125 | 0.1606 |
| DM | pre_stim | pupil_area | LPE11623_2024_03_29 | LPE11623 | 99 | 18 | 0.6244 | 0.07547 | 0.5489 |
| DM | pre_stim | pupil_area | LPE11997_2024_04_07 | LPE11997 | 194 | 13 | 0.3264 | 0.5993 | -0.2728 |
| DM | pre_stim | pupil_area | LPE11998_2024_04_11 | LPE11998 | 138 | 11 | 0.5864 | 0.02792 | 0.5584 |
| DM | pre_stim | pupil_area | LPE11998_2024_04_12 | LPE11998 | 179 | 15 | 0.2213 | 0.1998 | 0.02155 |
| DM | pre_stim | pupil_area | LPE12013_2024_04_11 | LPE12013 | 216 | 18 | 0.2949 | 0.3166 | -0.02165 |
| DM | pre_stim | runspeed | LPE11495_2024_02_16 | LPE11495 | 157 | 12 | 19.64 | 16.21 | 3.438 |
| DM | pre_stim | runspeed | LPE11622_2024_02_16 | LPE11622 | 334 | 19 | 26.73 | 24.18 | 2.549 |
| DM | pre_stim | runspeed | LPE11623_2024_03_27 | LPE11623 | 66 | 21 | 11.43 | 11.16 | 0.269 |
| DM | pre_stim | runspeed | LPE11623_2024_03_29 | LPE11623 | 100 | 18 | 13.2 | 11.65 | 1.555 |
| DM | pre_stim | runspeed | LPE11997_2024_04_05 | LPE11997 | 159 | 20 | 19.3 | 13.14 | 6.157 |
| DM | pre_stim | runspeed | LPE11997_2024_04_07 | LPE11997 | 195 | 14 | 17.65 | 12.64 | 5.012 |
| DM | pre_stim | runspeed | LPE11998_2024_04_11 | LPE11998 | 138 | 11 | 16.69 | 10.31 | 6.38 |
| DM | pre_stim | runspeed | LPE11998_2024_04_12 | LPE11998 | 179 | 15 | 19.73 | 12.62 | 7.116 |
| DM | pre_stim | runspeed | LPE12013_2024_04_11 | LPE12013 | 217 | 18 | 21.56 | 16.94 | 4.614 |
| DM | pre_stim | runspeed | LPE12013_2024_04_12 | LPE12013 | 155 | 19 | 14.11 | 10.85 | 3.257 |
| DM | stim | pupil_area | LPE11623_2024_03_27 | LPE11623 | 66 | 21 | 0.42 | 0.4747 | -0.05472 |
| DM | stim | pupil_area | LPE11623_2024_03_29 | LPE11623 | 100 | 18 | 0.3356 | 0.1923 | 0.1432 |
| DM | stim | pupil_area | LPE11997_2024_04_07 | LPE11997 | 195 | 13 | 0.2267 | 0.8337 | -0.6069 |
| DM | stim | pupil_area | LPE11998_2024_04_11 | LPE11998 | 138 | 11 | 0.4724 | -0.2549 | 0.7273 |
| DM | stim | pupil_area | LPE11998_2024_04_12 | LPE11998 | 179 | 15 | 0.1745 | 0.06939 | 0.1051 |
| DM | stim | pupil_area | LPE12013_2024_04_11 | LPE12013 | 216 | 18 | -0.1263 | 0.4919 | -0.6183 |
| DM | stim | runspeed | LPE11495_2024_02_16 | LPE11495 | 157 | 12 | 9.661 | 12.22 | -2.559 |
| DM | stim | runspeed | LPE11622_2024_02_16 | LPE11622 | 334 | 19 | 16.76 | 19.87 | -3.113 |
| DM | stim | runspeed | LPE11623_2024_03_27 | LPE11623 | 66 | 21 | 7.479 | 10.07 | -2.595 |
| DM | stim | runspeed | LPE11623_2024_03_29 | LPE11623 | 100 | 18 | 8.974 | 11.79 | -2.813 |
| DM | stim | runspeed | LPE11997_2024_04_05 | LPE11997 | 159 | 20 | 12.56 | 11.44 | 1.121 |
| DM | stim | runspeed | LPE11997_2024_04_07 | LPE11997 | 195 | 14 | 18.42 | 15.01 | 3.405 |
| DM | stim | runspeed | LPE11998_2024_04_11 | LPE11998 | 138 | 11 | 10.13 | 8.385 | 1.746 |
| DM | stim | runspeed | LPE11998_2024_04_12 | LPE11998 | 179 | 15 | 14.69 | 10.28 | 4.409 |

*(showing first 30 of 252 rows -- see the linked CSV for the rest)*

[performance_comparison_per_unit__session.csv](../2_pipeline/1h_pupil_speed_vs_task/out/performance_comparison_per_unit__session.csv)

### Signal quantile group tests  session

| protocol | window | y | n_units | mean_rho | p | p_holm |
| --- | --- | --- | --- | --- | --- | --- |
| DM | pre_stim | pupil_area | 6 | -0.01858 | 1 | 1 |
| DM | pre_stim | runspeed | 11 | 0.102 | 0.08301 | 0.08301 |
| DM | stim | pupil_area | 6 | -0.2165 | 0.3125 | 0.625 |
| DM | stim | runspeed | 11 | -0.2092 | 0.02441 | 0.04883 |
| DM | reward | pupil_area | 6 | -0.3032 | 0.0625 | 0.1875 |
| DM | reward | runspeed | 11 | -0.4778 | 0.0009766 | 0.00293 |
| DP | pre_stim | pupil_area | 7 | 0.002788 | 0.6875 | 1 |
| DP | pre_stim | runspeed | 10 | -0.05961 | 0.375 | 0.375 |
| DP | stim | pupil_area | 7 | -0.04327 | 0.4688 | 1 |
| DP | stim | runspeed | 10 | -0.2069 | 0.03711 | 0.07422 |
| DP | reward | pupil_area | 7 | -0.1053 | 0.4688 | 1 |
| DP | reward | runspeed | 10 | -0.4433 | 0.003906 | 0.01172 |
| DN | pre_stim | pupil_area | 25 | 0.09412 | 2.211e-05 | 6.634e-05 |
| DN | pre_stim | runspeed | 26 | 0.02442 | 0.4082 | 0.4082 |
| DN | stim | pupil_area | 25 | 0.04097 | 0.01145 | 0.02291 |
| DN | stim | runspeed | 26 | -0.1327 | 0.002182 | 0.004363 |
| DN | reward | pupil_area | 25 | -0.02944 | 0.5424 | 0.5424 |
| DN | reward | runspeed | 26 | -0.531 | 2.98e-08 | 8.941e-08 |

[signal_quantile_group_tests__session.csv](../2_pipeline/1h_pupil_speed_vs_task/out/signal_quantile_group_tests__session.csv)

### Signal quantile tuning  session

| protocol | window | y | signal_bin | mean | sem |
| --- | --- | --- | --- | --- | --- |
| DM | pre_stim | pupil_area | 1 | 0.4113 | 0.1368 |
| DM | pre_stim | pupil_area | 2 | 0.2109 | 0.1631 |
| DM | pre_stim | pupil_area | 3 | 0.6033 | 0.2341 |
| DM | pre_stim | pupil_area | 4 | 0.3873 | 0.1228 |
| DM | pre_stim | pupil_area | 5 | 0.5047 | 0.2232 |
| DM | pre_stim | pupil_area | 6 | 0.4224 | 0.1739 |
| DM | pre_stim | pupil_area | 7 | 0.3601 | 0.1528 |
| DM | pre_stim | pupil_area | 8 | 0.4182 | 0.2665 |
| DM | pre_stim | runspeed | 1 | 17.41 | 1.446 |
| DM | pre_stim | runspeed | 2 | 15.17 | 1.313 |
| DM | pre_stim | runspeed | 3 | 16.11 | 2.07 |
| DM | pre_stim | runspeed | 4 | 17.77 | 1.372 |
| DM | pre_stim | runspeed | 5 | 18.89 | 1.483 |
| DM | pre_stim | runspeed | 6 | 18.26 | 1.437 |
| DM | pre_stim | runspeed | 7 | 19.06 | 1.308 |
| DM | pre_stim | runspeed | 8 | 16.57 | 1.187 |
| DM | stim | pupil_area | 1 | 0.4763 | 0.1144 |
| DM | stim | pupil_area | 2 | 0.265 | 0.1773 |
| DM | stim | pupil_area | 3 | 0.2679 | 0.2069 |
| DM | stim | pupil_area | 4 | 0.3018 | 0.1708 |
| DM | stim | pupil_area | 5 | 0.2379 | 0.1959 |
| DM | stim | pupil_area | 6 | 0.2007 | 0.1912 |
| DM | stim | pupil_area | 7 | 0.0756 | 0.1659 |
| DM | stim | pupil_area | 8 | 0.1995 | 0.2527 |
| DM | stim | runspeed | 1 | 17.51 | 1.065 |
| DM | stim | runspeed | 2 | 14.64 | 1.01 |
| DM | stim | runspeed | 3 | 9.383 | 1.247 |
| DM | stim | runspeed | 4 | 10.11 | 1.06 |
| DM | stim | runspeed | 5 | 10.47 | 1.263 |
| DM | stim | runspeed | 6 | 11.48 | 1.435 |

*(showing first 30 of 144 rows -- see the linked CSV for the rest)*

[signal_quantile_tuning__session.csv](../2_pipeline/1h_pupil_speed_vs_task/out/signal_quantile_tuning__session.csv)

### Stim strategy group tests  session

| protocol | strategy | window | y | n_units | mean_rho | p | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DM | zero_vs_nonzero | pre_stim | pupil_area | 6 | 0.03451 | 0.3125 | 0.3125 |
| DM | zero_vs_nonzero | pre_stim | runspeed | 11 | -0.008732 | 0.7646 | 0.7646 |
| DM | zero_vs_nonzero | stim | pupil_area | 6 | -0.1901 | 0.03125 | 0.09375 |
| DM | zero_vs_nonzero | stim | runspeed | 11 | -0.4529 | 0.001953 | 0.003906 |
| DM | zero_vs_nonzero | reward | pupil_area | 6 | -0.2431 | 0.03125 | 0.09375 |
| DM | zero_vs_nonzero | reward | runspeed | 11 | -0.5208 | 0.0009766 | 0.00293 |
| DP | zero_vs_nonzero | pre_stim | pupil_area | 7 | 0.01086 | 0.9375 | 1 |
| DP | zero_vs_nonzero | pre_stim | runspeed | 10 | -0.03693 | 0.2324 | 0.2324 |
| DP | zero_vs_nonzero | stim | pupil_area | 7 | 0.009691 | 0.9375 | 1 |
| DP | zero_vs_nonzero | stim | runspeed | 10 | -0.1585 | 0.06445 | 0.1289 |
| DP | zero_vs_nonzero | reward | pupil_area | 7 | -0.03302 | 0.5781 | 1 |
| DP | zero_vs_nonzero | reward | runspeed | 10 | -0.341 | 0.001953 | 0.005859 |
| DN | zero_to80_above80 | pre_stim | pupil_area | 25 | 0.03435 | 0.1073 | 0.2146 |
| DN | zero_to80_above80 | pre_stim | runspeed | 26 | 0.02749 | 0.1574 | 0.1574 |
| DN | zero_to80_above80 | stim | pupil_area | 25 | -0.009333 | 0.8325 | 0.8325 |
| DN | zero_to80_above80 | stim | runspeed | 26 | -0.1195 | 0.006131 | 0.01226 |
| DN | zero_to80_above80 | reward | pupil_area | 25 | -0.08324 | 0.02027 | 0.06082 |
| DN | zero_to80_above80 | reward | runspeed | 26 | -0.5181 | 2.98e-08 | 8.941e-08 |
| DN | pct_0-20_20-80_80-100 | pre_stim | pupil_area | 25 | 0.03127 | 0.2002 | 0.4004 |
| DN | pct_0-20_20-80_80-100 | pre_stim | runspeed | 26 | 0.03226 | 0.07957 | 0.07957 |
| DN | pct_0-20_20-80_80-100 | stim | pupil_area | 25 | -0.011 | 0.9368 | 0.9368 |
| DN | pct_0-20_20-80_80-100 | stim | runspeed | 26 | -0.1228 | 0.006131 | 0.01226 |
| DN | pct_0-20_20-80_80-100 | reward | pupil_area | 25 | -0.08853 | 0.01051 | 0.03153 |
| DN | pct_0-20_20-80_80-100 | reward | runspeed | 26 | -0.5318 | 2.98e-08 | 8.941e-08 |
| DN | zero_vs_nonzero | pre_stim | pupil_area | 25 | 0.02776 | 0.03668 | 0.07336 |
| DN | zero_vs_nonzero | pre_stim | runspeed | 26 | 0.02798 | 0.03955 | 0.03955 |
| DN | zero_vs_nonzero | stim | pupil_area | 25 | 0.002714 | 0.4578 | 0.4578 |
| DN | zero_vs_nonzero | stim | runspeed | 26 | -0.06225 | 0.007938 | 0.01588 |
| DN | zero_vs_nonzero | reward | pupil_area | 25 | -0.06716 | 0.009635 | 0.02891 |
| DN | zero_vs_nonzero | reward | runspeed | 26 | -0.3591 | 2.98e-08 | 8.941e-08 |

[stim_strategy_group_tests__session.csv](../2_pipeline/1h_pupil_speed_vs_task/out/stim_strategy_group_tests__session.csv)

### Stim strategy tuning  session

| protocol | strategy | window | y | bin | mean | sem |
| --- | --- | --- | --- | --- | --- | --- |
| DM | zero_vs_nonzero | pre_stim | pupil_area | 0 | 0.3925 | 0.07068 |
| DM | zero_vs_nonzero | pre_stim | pupil_area | 1 | 0.4186 | 0.06404 |
| DM | zero_vs_nonzero | pre_stim | runspeed | 0 | 17.44 | 1.421 |
| DM | zero_vs_nonzero | pre_stim | runspeed | 1 | 17.4 | 1.274 |
| DM | zero_vs_nonzero | stim | pupil_area | 0 | 0.4639 | 0.08039 |
| DM | zero_vs_nonzero | stim | pupil_area | 1 | 0.2005 | 0.09396 |
| DM | zero_vs_nonzero | stim | runspeed | 0 | 17.77 | 1.063 |
| DM | zero_vs_nonzero | stim | runspeed | 1 | 10.63 | 1.096 |
| DM | zero_vs_nonzero | reward | pupil_area | 0 | 0.4835 | 0.07918 |
| DM | zero_vs_nonzero | reward | pupil_area | 1 | 0.1215 | 0.05495 |
| DM | zero_vs_nonzero | reward | runspeed | 0 | 16.91 | 1.143 |
| DM | zero_vs_nonzero | reward | runspeed | 1 | 6.819 | 0.6653 |
| DP | zero_vs_nonzero | pre_stim | pupil_area | 0 | 0.1249 | 0.1283 |
| DP | zero_vs_nonzero | pre_stim | pupil_area | 1 | 0.197 | 0.1351 |
| DP | zero_vs_nonzero | pre_stim | runspeed | 0 | 22.34 | 2.976 |
| DP | zero_vs_nonzero | pre_stim | runspeed | 1 | 21.64 | 2.998 |
| DP | zero_vs_nonzero | stim | pupil_area | 0 | 0.1443 | 0.1337 |
| DP | zero_vs_nonzero | stim | pupil_area | 1 | 0.1969 | 0.1584 |
| DP | zero_vs_nonzero | stim | runspeed | 0 | 20.91 | 2.885 |
| DP | zero_vs_nonzero | stim | runspeed | 1 | 18.2 | 2.905 |
| DP | zero_vs_nonzero | reward | pupil_area | 0 | 0.1386 | 0.1334 |
| DP | zero_vs_nonzero | reward | pupil_area | 1 | 0.1184 | 0.1296 |
| DP | zero_vs_nonzero | reward | runspeed | 0 | 20.71 | 2.875 |
| DP | zero_vs_nonzero | reward | runspeed | 1 | 13.22 | 2.019 |
| DN | zero_to80_above80 | pre_stim | pupil_area | 0 | -0.006146 | 0.08979 |
| DN | zero_to80_above80 | pre_stim | pupil_area | 1 | 0.01811 | 0.08613 |
| DN | zero_to80_above80 | pre_stim | pupil_area | 2 | 0.06276 | 0.08132 |
| DN | zero_to80_above80 | pre_stim | runspeed | 0 | 28.68 | 2.435 |
| DN | zero_to80_above80 | pre_stim | runspeed | 1 | 29.29 | 2.429 |
| DN | zero_to80_above80 | pre_stim | runspeed | 2 | 29.34 | 2.375 |

*(showing first 30 of 72 rows -- see the linked CSV for the rest)*

[stim_strategy_tuning__session.csv](../2_pipeline/1h_pupil_speed_vs_task/out/stim_strategy_tuning__session.csv)
