# Exploratory HR Outcome Regressions

## Sample

The sample is the Nathan-score window |score| <= 1 ft, restricted to launch angles at or below the first quartile of observed HR launch angle (26 degrees). It contains 630 batted-ball plate appearances: 86 HRs and 544 controls (HR share 13.7%).

This is an exploratory unadjusted comparison: each regression is `outcome ~ home_run` with HC3 heteroskedasticity-robust standard errors. Robust SEs address heteroskedasticity, not confounding or misspecification.

## Outcome Availability

The stored data are one row per batted-ball plate appearance. They do not contain subsequent pitches, pitcher substitutions after the event, next-half-inning outcomes, next at-bat outcomes, post-event scores, or a complete pitch-by-pitch sequence. Therefore, the only available numeric change outcomes are `delta_run_exp` and `delta_home_win_exp`; both measure the same play/plate appearance that defines the treatment, not the response after it. Their regressions below are descriptive/mechanical and do not estimate downstream effects of a home run.

The requested next-pitch, rest-of-inning, next-half-inning, and next-PA effects require a complete pitch-level game stream joined to the treated batted-ball rows. The current downloader filters to fly-ball batted events, so those follow-up observations are absent.

## Regression Results

| outcome | treated_n | control_n | treated_mean | control_mean | hr_coefficient | hc3_robust_se | ci_95_lower | ci_95_upper | p_value | r_squared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| delta_run_exp | 86.000 | 544.000 | 1.521 | 0.154 | 1.367 | 0.060 | 1.250 | 1.484 | 7.77e-116 | 0.453 |
| delta_home_win_exp | 86.000 | 544.000 | 0.008 | -0.005 | 0.013 | 0.021 | -0.028 | 0.054 | 0.531 | 0.002 |

The coefficient is the HR-minus-control mean difference in the same-play outcome. Interpret neither coefficient as an effect on subsequent pitching or scoring.

## Sample Summary

| Variable | HR mean | HR SD | Control mean | Control SD |
|---|---:|---:|---:|---:|
| launch_speed | 102.230 | 3.018 | 99.405 | 4.080 |
| launch_angle | 23.756 | 1.982 | 24.121 | 1.875 |
| running_var_nathan | -0.029 | 0.567 | 0.014 | 0.576 |
| z_at_fence_ft | 8.039 | 1.765 | 8.598 | 3.251 |
| model_landing_distance_ft | 411.089 | 12.899 | 399.807 | 17.850 |
| delta_run_exp | 1.521 | 0.512 | 0.154 | 0.518 |
| delta_home_win_exp | 0.008 | 0.191 | -0.005 | 0.072 |

Launch speed remains substantially imbalanced: HR mean 102.23 mph versus control mean 99.40 mph; SMD = 0.79. This is not ‘pretty good’ balance and the unadjusted regressions should not be interpreted causally.

## Full Descriptive Statistics

The table below summarizes all 69 columns in the selected sample. Numeric rows show overall and treatment/control means, SDs, quartiles, and range; categorical and identifier rows show missingness, unique counts, and the ten most frequent values in each group. A machine-readable version is in [nathan_q1_pm1_descriptive_stats.csv](nathan_q1_pm1_descriptive_stats.csv).

| variable | type | missing (all) | unique (all) | all mean | all SD | all min | all Q1 | all median | all Q3 | all max | HR mean | HR SD | control mean | control SD | most common levels by group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| game_pk | categorical/id | 0.000 | 625.000 |  |  |  |  |  |  |  |  |  |  |  | all: 530052: 2; 565968: 2; 565005: 2; 565770: 2; 492156: 2; HR: 565770: 2; 565572: 1; 567199: 1; 567499: 1; 565842: 1; control: 565968: 2; 530052: 2; 492156: 2; 416060: 1; 661218: 1 |
| game_date | categorical/id | 0.000 | 544.000 |  |  |  |  |  |  |  |  |  |  |  | all: 2015-10-03: 4; 2022-09-25: 4; 2017-09-04: 3; 2019-05-12: 3; 2018-05-16: 3; HR: 2017-09-14: 2; 2019-09-24: 2; 2019-08-31: 2; 2015-06-28: 1; 2019-08-18: 1; control: 2015-10-03: 4; 2022-09-25: 4; 2023-07-24: 3; 2018-05-16: 3; 2022-07-02: 3 |
| game_year | numeric | 0.000 | 10.000 | 2019.487 | 2.900 | 2015.000 | 2017.000 | 2019.000 | 2022.000 | 2024.000 | 2018.453 | 2.268 | 2019.651 | 2.956 |  |
| game_type | categorical/id | 0.000 | 4.000 |  |  |  |  |  |  |  |  |  |  |  | all: R: 621; S: 7; D: 1; F: 1; HR: R: 85; S: 1; control: R: 536; S: 6; D: 1; F: 1 |
| home_team | categorical/id | 0.000 | 28.000 |  |  |  |  |  |  |  |  |  |  |  | all: BOS: 34; MIN: 33; BAL: 31; ATL: 30; KC: 30; HR: LAA: 8; NYY: 8; BAL: 7; KC: 6; TB: 5; control: BOS: 32; MIN: 30; ATL: 26; PIT: 26; SF: 25 |
| away_team | categorical/id | 0.000 | 30.000 |  |  |  |  |  |  |  |  |  |  |  | all: TOR: 31; MIA: 29; ATH: 27; DET: 27; MIL: 27; HR: CLE: 7; DET: 6; BOS: 6; PHI: 5; MIA: 5; control: TOR: 29; MIL: 26; MIA: 24; ATH: 23; BAL: 22 |
| batter | categorical/id | 0.000 | 408.000 |  |  |  |  |  |  |  |  |  |  |  | all: 666182: 6; 527038: 6; 596019: 5; 543305: 5; 446334: 5; HR: 673548: 2; 475174: 2; 641531: 2; 660271: 2; 435559: 2; control: 666182: 5; 656775: 5; 527038: 5; 543305: 5; 650333: 4 |
| pitcher | categorical/id | 0.000 | 450.000 |  |  |  |  |  |  |  |  |  |  |  | all: 527048: 6; 519144: 5; 669432: 4; 592662: 4; 592332: 4; HR: 519144: 3; 601713: 2; 571666: 2; 407845: 1; 453265: 1; control: 527048: 5; 592332: 4; 571945: 4; 542881: 4; 592662: 4 |
| at_bat_number | numeric | 0.000 | 90.000 | 39.284 | 22.664 | 1.000 | 19.000 | 40.000 | 57.000 | 113.000 | 36.477 | 21.673 | 39.728 | 22.805 |  |
| pitch_number | numeric | 0.000 | 10.000 | 3.473 | 1.896 | 1.000 | 2.000 | 3.000 | 5.000 | 11.000 | 3.628 | 1.729 | 3.449 | 1.921 |  |
| inning | numeric | 0.000 | 13.000 | 5.041 | 2.635 | 1.000 | 3.000 | 5.000 | 7.000 | 13.000 | 4.744 | 2.622 | 5.088 | 2.636 |  |
| inning_topbot | categorical/id | 0.000 | 2.000 |  |  |  |  |  |  |  |  |  |  |  | all: Top: 336; Bot: 294; HR: Top: 46; Bot: 40; control: Top: 290; Bot: 254 |
| events | categorical/id | 0.000 | 6.000 |  |  |  |  |  |  |  |  |  |  |  | all: field_out: 319; double: 181; home_run: 86; triple: 33; sac_fly: 8; HR: home_run: 86; control: field_out: 319; double: 181; triple: 33; sac_fly: 8; single: 3 |
| description | categorical/id | 0.000 | 1.000 |  |  |  |  |  |  |  |  |  |  |  | all: hit_into_play: 630; HR: hit_into_play: 86; control: hit_into_play: 544 |
| type | categorical/id | 0.000 | 1.000 |  |  |  |  |  |  |  |  |  |  |  | all: X: 630; HR: X: 86; control: X: 544 |
| bb_type | categorical/id | 0.000 | 1.000 |  |  |  |  |  |  |  |  |  |  |  | all: fly_ball: 630; HR: fly_ball: 86; control: fly_ball: 544 |
| launch_speed | numeric | 0.000 | 171.000 | 99.790 | 4.067 | 87.100 | 97.400 | 100.300 | 102.400 | 110.600 | 102.230 | 3.018 | 99.405 | 4.080 |  |
| launch_angle | numeric | 0.000 | 12.000 | 24.071 | 1.892 | 15.000 | 23.000 | 25.000 | 26.000 | 26.000 | 23.756 | 1.982 | 24.121 | 1.875 |  |
| hc_x | numeric | 0.000 | 612.000 | 128.037 | 39.612 | 24.390 | 98.252 | 129.145 | 158.840 | 208.440 | 117.578 | 32.128 | 129.690 | 40.447 |  |
| hc_y | numeric | 0.000 | 588.000 | 49.942 | 20.929 | 11.510 | 36.110 | 45.160 | 58.657 | 124.940 | 29.162 | 10.596 | 53.227 | 20.264 |  |
| hit_distance_sc | numeric | 0.000 | 138.000 | 378.130 | 29.575 | 272.000 | 364.000 | 383.000 | 399.000 | 446.000 | 410.895 | 14.452 | 372.950 | 27.993 |  |
| release_spin_rate | numeric | 8.000 | 466.000 | 2189.371 | 321.747 | 958.000 | 2037.250 | 2228.500 | 2386.000 | 3213.000 | 2134.554 | 312.221 | 2197.813 | 322.643 |  |
| outs_when_up | numeric | 0.000 | 3.000 | 1.002 | 0.780 | 0.000 | 0.000 | 1.000 | 2.000 | 2.000 | 0.953 | 0.825 | 1.009 | 0.774 |  |
| on_1b | categorical/id | 439.000 | 165.000 |  |  |  |  |  |  |  |  |  |  |  | all: 677951: 3; 455976: 3; 434778: 2; 665489: 2; 516416: 2; HR: 444843: 1; 593934: 1; 641355: 1; 425877: 1; 608385: 1; control: 455976: 3; 677951: 3; 541645: 2; 621043: 2; 641553: 2 |
| on_2b | categorical/id | 509.000 | 112.000 |  |  |  |  |  |  |  |  |  |  |  | all: 673357: 2; 622761: 2; 607680: 2; 592192: 2; 542303: 2; HR: 592518: 1; 608700: 1; 664058: 1; 606157: 1; 605421: 1; control: 645277: 2; 673357: 2; 607680: 2; 592192: 2; 622761: 2 |
| on_3b | categorical/id | 581.000 | 47.000 |  |  |  |  |  |  |  |  |  |  |  | all: 542340: 2; 621035: 2; 453064: 1; 644374: 1; 664670: 1; HR: 542340: 2; 648717: 1; 461858: 1; 664670: 1; 657557: 1; control: 621035: 2; 453064: 1; 592669: 1; 606992: 1; 642211: 1 |
| home_score | numeric | 0.000 | 14.000 | 2.206 | 2.543 | 0.000 | 0.000 | 1.000 | 4.000 | 14.000 | 2.302 | 2.493 | 2.191 | 2.553 |  |
| away_score | numeric | 0.000 | 15.000 | 2.468 | 2.563 | 0.000 | 0.000 | 2.000 | 4.000 | 14.000 | 2.198 | 2.157 | 2.511 | 2.621 |  |
| bat_score | numeric | 0.000 | 13.000 | 2.446 | 2.436 | 0.000 | 0.000 | 2.000 | 4.000 | 12.000 | 2.326 | 2.204 | 2.465 | 2.472 |  |
| fld_score | numeric | 0.000 | 15.000 | 2.229 | 2.668 | 0.000 | 0.000 | 1.000 | 3.000 | 14.000 | 2.174 | 2.450 | 2.237 | 2.703 |  |
| bat_win_exp | numeric | 0.000 | 428.000 | 0.543 | 0.292 | 0.001 | 0.347 | 0.542 | 0.791 | 0.999 | 0.514 | 0.272 | 0.548 | 0.295 |  |
| home_win_exp | numeric | 0.000 | 423.000 | 0.496 | 0.295 | 0.001 | 0.251 | 0.520 | 0.704 | 0.999 | 0.524 | 0.271 | 0.492 | 0.299 |  |
| delta_home_win_exp | numeric | 0.000 | 258.000 | -0.003 | 0.097 | -0.418 | -0.032 | 0.000 | 0.025 | 0.614 | 0.008 | 0.191 | -0.005 | 0.072 |  |
| delta_run_exp | numeric | 0.000 | 386.000 | 0.341 | 0.698 | -0.458 | -0.251 | -0.163 | 0.796 | 2.639 | 1.521 | 0.512 | 0.154 | 0.518 |  |
| stand | categorical/id | 0.000 | 2.000 |  |  |  |  |  |  |  |  |  |  |  | all: R: 375; L: 255; HR: R: 56; L: 30; control: R: 319; L: 225 |
| p_throws | categorical/id | 0.000 | 2.000 |  |  |  |  |  |  |  |  |  |  |  | all: R: 442; L: 188; HR: R: 64; L: 22; control: R: 378; L: 166 |
| n_thruorder_pitcher | numeric | 0.000 | 4.000 | 1.562 | 0.778 | 1.000 | 1.000 | 1.000 | 2.000 | 4.000 | 1.535 | 0.762 | 1.566 | 0.781 |  |
| pitch_type | categorical/id | 0.000 | 13.000 |  |  |  |  |  |  |  |  |  |  |  | all: FF: 242; SI: 109; SL: 81; CH: 72; FC: 44; HR: FF: 30; SI: 18; CH: 14; SL: 8; FC: 6; control: FF: 212; SI: 91; SL: 73; CH: 58; FC: 38 |
| pitch_name | categorical/id | 0.000 | 13.000 |  |  |  |  |  |  |  |  |  |  |  | all: 4-Seam Fastball: 242; Sinker: 109; Slider: 81; Changeup: 72; Cutter: 44; HR: 4-Seam Fastball: 30; Sinker: 18; Changeup: 14; Slider: 8; Cutter: 6; control: 4-Seam Fastball: 212; Sinker: 91; Slider: 73; Changeup: 58; Cutter: 38 |
| release_speed | numeric | 0.000 | 213.000 | 89.147 | 5.838 | 53.700 | 85.125 | 90.250 | 93.800 | 100.700 | 89.197 | 5.390 | 89.140 | 5.910 |  |
| effective_speed | numeric | 1.000 | 213.000 | 89.008 | 5.993 | 51.700 | 85.000 | 90.200 | 93.700 | 102.000 | 89.078 | 5.395 | 88.997 | 6.086 |  |
| plate_x | numeric | 0.000 | 630.000 | 0.001 | 0.452 | -1.297 | -0.317 | 0.003 | 0.334 | 1.319 | -0.075 | 0.465 | 0.014 | 0.450 |  |
| plate_z | numeric | 0.000 | 630.000 | 2.390 | 0.506 | 0.700 | 2.034 | 2.379 | 2.755 | 3.709 | 2.415 | 0.556 | 2.386 | 0.498 |  |
| pfx_x | numeric | 0.000 | 280.000 | -0.112 | 0.876 | -1.820 | -0.840 | -0.250 | 0.637 | 2.020 | -0.175 | 0.950 | -0.102 | 0.865 |  |
| pfx_z | numeric | 0.000 | 230.000 | 0.724 | 0.706 | -1.450 | 0.340 | 0.850 | 1.290 | 1.930 | 0.777 | 0.609 | 0.715 | 0.720 |  |
| zone | numeric | 0.000 | 13.000 | 6.032 | 3.089 | 1.000 | 4.000 | 5.000 | 8.000 | 14.000 | 6.186 | 3.127 | 6.007 | 3.085 |  |
| if_fielding_alignment | categorical/id | 4.000 | 4.000 |  |  |  |  |  |  |  |  |  |  |  | all: Standard: 430; Infield shift: 113; Infield shade: 43; Strategic: 40; HR: Standard: 60; Infield shift: 16; Infield shade: 5; Strategic: 4; control: Standard: 370; Infield shift: 97; Infield shade: 38; Strategic: 36 |
| of_fielding_alignment | categorical/id | 4.000 | 2.000 |  |  |  |  |  |  |  |  |  |  |  | all: Standard: 586; Strategic: 40; HR: Standard: 80; Strategic: 5; control: Standard: 506; Strategic: 35 |
| hit_location | categorical/id | 105.000 | 3.000 |  |  |  |  |  |  |  |  |  |  |  | all: 8: 325; 9: 108; 7: 92; HR: ; control: 8: 325; 9: 108; 7: 92 |
| fielder_2 | categorical/id | 0.000 | 161.000 |  |  |  |  |  |  |  |  |  |  |  | all: 592663: 17; 425877: 14; 521692: 14; 518735: 13; 543877: 13; HR: 456078: 4; 506702: 4; 592663: 4; 596142: 4; 572287: 3; control: 425877: 14; 521692: 13; 518735: 13; 592663: 13; 543877: 11 |
| fielder_3 | categorical/id | 0.000 | 193.000 |  |  |  |  |  |  |  |  |  |  |  | all: 518692: 21; 547989: 17; 605137: 15; 474832: 14; 502671: 14; HR: 571506: 5; 621566: 3; 493329: 3; 467793: 2; 641820: 2; control: 518692: 21; 547989: 15; 605137: 14; 543333: 13; 502671: 13 |
| fielder_4 | categorical/id | 0.000 | 210.000 |  |  |  |  |  |  |  |  |  |  |  | all: 593160: 19; 570731: 16; 429664: 15; 596059: 15; 624428: 15; HR: 516770: 4; 593160: 4; 543401: 3; 596059: 3; 476704: 3; control: 593160: 15; 624428: 14; 645277: 14; 570731: 13; 429664: 12 |
| fielder_5 | categorical/id | 0.000 | 209.000 |  |  |  |  |  |  |  |  |  |  |  | all: 572122: 19; 592518: 19; 571448: 19; 656305: 15; 543685: 15; HR: 543768: 4; 446334: 3; 445988: 3; 600869: 3; 572122: 3; control: 571448: 18; 592518: 17; 572122: 16; 543685: 14; 656305: 13 |
| fielder_6 | categorical/id | 0.000 | 168.000 |  |  |  |  |  |  |  |  |  |  |  | all: 593428: 27; 607208: 19; 596019: 17; 444876: 16; 621043: 16; HR: 593428: 7; 596019: 4; 544369: 4; 588751: 3; 592743: 3; control: 593428: 20; 607208: 17; 641487: 14; 444876: 14; 596019: 13 |
| fielder_7 | categorical/id | 0.000 | 284.000 |  |  |  |  |  |  |  |  |  |  |  | all: 643217: 14; 460086: 13; 542303: 13; 665742: 10; 572816: 10; HR: 458731: 3; 643217: 3; 572816: 3; 460086: 3; 518618: 2; control: 542303: 12; 643217: 11; 460086: 10; 665742: 9; 457708: 8 |
| fielder_8 | categorical/id | 0.000 | 221.000 |  |  |  |  |  |  |  |  |  |  |  | all: 598265: 21; 572191: 17; 607680: 14; 456715: 13; 545361: 13; HR: 598265: 5; 543305: 4; 621446: 3; 543807: 3; 516782: 2; control: 598265: 16; 572191: 16; 607680: 13; 545361: 12; 456715: 12 |
| fielder_9 | categorical/id | 0.000 | 232.000 |  |  |  |  |  |  |  |  |  |  |  | all: 605141: 18; 455976: 16; 596146: 16; 592206: 13; 571745: 11; HR: 605141: 6; 608577: 3; 592450: 3; 502210: 3; 605119: 2; control: 455976: 15; 596146: 14; 605141: 12; 592206: 11; 543807: 11 |
| spray_angle_deg | numeric | 0.000 | 630.000 | 1.309 | 16.653 | -44.037 | -9.655 | 1.293 | 12.220 | 42.459 | -2.810 | 11.230 | 1.961 | 17.274 |  |
| fence_dist_ft | numeric | 0.000 | 629.000 | 391.086 | 18.290 | 320.894 | 380.637 | 393.623 | 403.932 | 433.317 | 401.499 | 13.255 | 389.440 | 18.443 |  |
| fence_height_ft | numeric | 0.000 | 13.000 | 8.514 | 3.035 | 4.000 | 8.000 | 8.000 | 8.000 | 37.000 | 8.068 | 1.715 | 8.585 | 3.189 |  |
| h_at_fence_ft | numeric | 0.000 | 630.000 | -6.502 | 8.786 | -40.133 | -11.741 | -5.988 | -0.386 | 22.391 | 4.473 | 4.167 | -8.236 | 8.038 |  |
| running_var | numeric | 0.000 | 630.000 | -15.016 | 8.798 | -48.133 | -20.128 | -14.420 | -9.068 | 6.019 | -3.594 | 4.046 | -16.821 | 7.948 |  |
| home_run | numeric | 0.000 | 2.000 | 0.137 | 0.344 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |  |
| predicted_hr | numeric | 0.000 | 2.000 | 0.030 | 0.171 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.209 | 0.409 | 0.002 | 0.043 |  |
| running_var_ft | numeric | 0.000 | 630.000 | 0.008 | 0.574 | -1.000 | -0.508 | -0.001 | 0.482 | 0.996 | -0.029 | 0.567 | 0.014 | 0.576 |  |
| z_at_fence_ft | numeric | 0.000 | 630.000 | 8.522 | 3.096 | 3.268 | 7.426 | 8.031 | 8.707 | 37.613 | 8.039 | 1.765 | 8.598 | 3.251 |  |
| model_landing_distance_ft | numeric | 0.000 | 630.000 | 401.347 | 17.680 | 343.750 | 390.561 | 403.137 | 414.229 | 442.435 | 411.089 | 12.899 | 399.807 | 17.850 |  |
| trajectory_status | categorical/id | 0.000 | 1.000 |  |  |  |  |  |  |  |  |  |  |  | all: 1: 630; HR: 1: 86; control: 1: 544 |
| running_var_nathan | numeric | 0.000 | 630.000 | 0.008 | 0.574 | -1.000 | -0.508 | -0.001 | 0.482 | 0.996 | -0.029 | 0.567 | 0.014 | 0.576 |  |

## Reproducibility

Run `python src/07_nathan_q1_post_hr_analysis.py` to regenerate this report and the CSV outputs.
