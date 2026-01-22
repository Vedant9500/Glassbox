# before applying fixes:

======================================================================
   EVOLUTIONARY ONN TRAINING TEST
======================================================================
Device: cuda

======================================================================
TASK: y = x²
======================================================================
============================================================
EVOLUTIONARY ONN TRAINING
============================================================
Device: cuda
Population: 15
Generations: 30
------------------------------------------------------------
Initialized population of 15 individuals
Refining initial constants...
Gen   0 | Best: 2.0467 | Mean: 6.4624 | Diversity: 15 | Best Ever: 2.0467
Gen   5 | Best: 0.1431 | Mean: 4.3207 | Diversity: 6 | Best Ever: 0.1431
Gen  10 | Best: 0.1431 | Mean: 0.8514 | Diversity: 3 | Best Ever: 0.1431
Gen  15 | Best: 0.1431 | Mean: 1.1776 | Diversity: 3 | Best Ever: 0.1431
Gen  20 | Best: 0.1212 | Mean: 0.9519 | Diversity: 5 | Best Ever: 0.1212
Gen  25 | Best: 0.1212 | Mean: 1.2371 | Diversity: 3 | Best Ever: 0.1212
Gen  29 | Best: 0.1212 | Mean: 0.3577 | Diversity: 3 | Best Ever: 0.1212
------------------------------------------------------------
Training complete in 260.5s
Final MSE: 0.0412
Correlation: 0.9970
Formula: (x0 + x0) + (-0.24*(x0 + x0)) + 0.24*(x0 * x0) + 1.62

Validation MSE: 4.7730
Validation Corr: 0.9993

--- MLP Baseline (same epochs) ---
MLP MSE: 1.4519, Corr: 0.9980

======================================================================
TASK: y = sin(x)
======================================================================
============================================================
EVOLUTIONARY ONN TRAINING
============================================================
Device: cuda
Population: 15
Generations: 30
------------------------------------------------------------
Initialized population of 15 individuals
Refining initial constants...
Gen   0 | Best: 0.2389 | Mean: 1.2891 | Diversity: 15 | Best Ever: 0.2389
Gen   5 | Best: 0.1718 | Mean: 0.3512 | Diversity: 5 | Best Ever: 0.1718
Gen  10 | Best: 0.1310 | Mean: 0.3645 | Diversity: 4 | Best Ever: 0.1310
Gen  15 | Best: 0.1310 | Mean: 0.2832 | Diversity: 4 | Best Ever: 0.1310
Gen  20 | Best: 0.1310 | Mean: 0.3294 | Diversity: 3 | Best Ever: 0.1310
Gen  25 | Best: 0.1310 | Mean: 0.3352 | Diversity: 2 | Best Ever: 0.1310
Gen  29 | Best: 0.1208 | Mean: 0.3469 | Diversity: 3 | Best Ever: 0.1208
------------------------------------------------------------
Training complete in 260.9s
Final MSE: 0.0208
Correlation: 0.9835
Formula: 0.31*cos(x0) + 0.30*(x0 + x0) + 0.06*(x0 + x0) + 0.06*(x0 * x0) + (-0.07)

Validation MSE: 1.7977
Validation Corr: -0.9698

--- MLP Baseline (same epochs) ---
MLP MSE: 0.2466, Corr: -0.9928

======================================================================
TASK: y = x² + x
======================================================================
============================================================
EVOLUTIONARY ONN TRAINING
============================================================
Device: cuda
Population: 15
Generations: 30
------------------------------------------------------------
Initialized population of 15 individuals
Refining initial constants...
Gen   0 | Best: 1.2520 | Mean: 3.8379 | Diversity: 15 | Best Ever: 1.2520
Gen   5 | Best: 0.2653 | Mean: 1.8709 | Diversity: 6 | Best Ever: 0.2653
Gen  10 | Best: 0.2171 | Mean: 1.5208 | Diversity: 5 | Best Ever: 0.2171
Gen  15 | Best: 0.0955 | Mean: 1.0242 | Diversity: 4 | Best Ever: 0.0955
Gen  20 | Best: 0.0883 | Mean: 5.5801 | Diversity: 10 | Best Ever: 0.0883
Gen  25 | Best: 0.0883 | Mean: 2.2883 | Diversity: 4 | Best Ever: 0.0883
Gen  29 | Best: 0.0883 | Mean: 1.6489 | Diversity: 3 | Best Ever: 0.0883
------------------------------------------------------------
Training complete in 317.0s
Final MSE: 0.0083
Correlation: 0.9991
Formula: 1.15*(x0 * x0) + 0.10*(x0 * x0) + 2.27

Validation MSE: 0.0062
Validation Corr: 1.0000

--- MLP Baseline (same epochs) ---
MLP MSE: 2.0810, Corr: 0.9985

======================================================================
   SUMMARY
======================================================================

Task            ONN MSE      ONN Corr   MLP MSE      MLP Corr
------------------------------------------------------------
y = x²          4.7730       0.9993     1.4519       0.9980
y = sin(x)      1.7977       -0.9698    0.2466       -0.9928
y = x² + x      0.0062       1.0000     2.0810       0.9985

Discovered Formulas:
  y = x²: (x0 + x0) + (-0.24*(x0 + x0)) + 0.24*(x0 * x0) + 1.62
  y = sin(x): 0.31*cos(x0) + 0.30*(x0 + x0) + 0.06*(x0 + x0) + 0.06*(x0 *
  y = x² + x: 1.15*(x0 * x0) + 0.10*(x0 * x0) + 2.27

======================================================================
   TEST COMPLETE
======================================================================


# After applying fixes :



======================================================================
   EVOLUTIONARY ONN TRAINING TEST
======================================================================
Device: cuda

======================================================================
TASK: y = x²
======================================================================
============================================================
EVOLUTIONARY ONN TRAINING
============================================================
Device: cuda
Population: 15
Generations: 30
------------------------------------------------------------
Initialized population of 15 individuals
Refining initial constants...
Gen   0 | Best: 4.5344 | Mean: 9.9919 | Diversity: 15 | Best Ever: 4.5344
Gen   5 | Best: 0.5945 | Mean: 2.7027 | Diversity: 4 | Best Ever: 0.5945
Gen  10 | Best: 0.4755 | Mean: 1.2660 | Diversity: 2 | Best Ever: 0.4755
Gen  15 | Best: 0.3563 | Mean: 2.2858 | Diversity: 3 | Best Ever: 0.3563
Gen  20 | Best: 0.3563 | Mean: 1.3846 | Diversity: 5 | Best Ever: 0.3563
Gen  25 | Best: 0.3563 | Mean: 2.4590 | Diversity: 5 | Best Ever: 0.3563
Gen  29 | Best: 0.2658 | Mean: 1.4412 | Diversity: 7 | Best Ever: 0.2658
------------------------------------------------------------
Training complete in 168.3s
Final MSE: 0.1558
Correlation: 0.9943
Formula: 0.81*(x0 * x0) + (-0.77*(x0)²) + 0.19*agg(x0) + 0.13*ln(x0) + 1.61

Validation MSE: 0.3460
Validation Corr: 0.9941

--- MLP Baseline (same epochs) ---
MLP MSE: 1.1941, Corr: 0.9979

======================================================================
TASK: y = sin(x)
======================================================================
============================================================
EVOLUTIONARY ONN TRAINING
============================================================
Device: cuda
Population: 15
Generations: 30
------------------------------------------------------------
Initialized population of 15 individuals
Refining initial constants...
Gen   0 | Best: 0.1685 | Mean: 1.5570 | Diversity: 15 | Best Ever: 0.1685
Gen   5 | Best: 0.1384 | Mean: 0.1928 | Diversity: 3 | Best Ever: 0.1384
Gen  10 | Best: 0.1145 | Mean: 0.2561 | Diversity: 3 | Best Ever: 0.1145
Gen  15 | Best: 0.1145 | Mean: 0.2252 | Diversity: 3 | Best Ever: 0.1145
Gen  20 | Best: 0.1145 | Mean: 0.2053 | Diversity: 3 | Best Ever: 0.1145
Gen  25 | Best: 0.1145 | Mean: 0.2141 | Diversity: 4 | Best Ever: 0.1145
Gen  29 | Best: 0.0969 | Mean: 0.1935 | Diversity: 4 | Best Ever: 0.0969
------------------------------------------------------------
Training complete in 194.1s
Final MSE: 0.0169
Correlation: 0.9963
Formula: 0.02*x0 - 0.29*agg(x0) + 0.19


Validation MSE: 0.0571

Validation MSE: 0.0571
Validation Corr: 0.6791

Validation MSE: 0.0571
Validation Corr: 0.6791

--- MLP Baseline (same epochs) ---
MLP MSE: 0.2499, Corr: -0.9941

======================================================================
TASK: y = x² + x
======================================================================
============================================================
EVOLUTIONARY ONN TRAINING
============================================================
Device: cuda
Population: 15
Generations: 30
------------------------------------------------------------
Initialized population of 15 individuals
Refining initial constants...
Gen   0 | Best: 1.2543 | Mean: 3.9960 | Diversity: 15 | Best Ever: 1.2543
Gen   5 | Best: 0.4276 | Mean: 2.1645 | Diversity: 5 | Best Ever: 0.4276
Gen  10 | Best: 0.1994 | Mean: 1.2789 | Diversity: 3 | Best Ever: 0.1994
Gen  15 | Best: 0.1073 | Mean: 0.4717 | Diversity: 4 | Best Ever: 0.1073
Gen  20 | Best: 0.0895 | Mean: 0.7548 | Diversity: 3 | Best Ever: 0.0895
Gen  25 | Best: 0.0860 | Mean: 0.5895 | Diversity: 2 | Best Ever: 0.0860
Gen  29 | Best: 0.0860 | Mean: 0.3180 | Diversity: 3 | Best Ever: 0.0860
------------------------------------------------------------
Training complete in 162.3s
Final MSE: 0.0060
Correlation: 0.9998
Formula: x0^2 + 0.7*agg(x0) + 1.84

Validation MSE: 1.4136
Validation Corr: 0.9999

--- MLP Baseline (same epochs) ---
MLP MSE: 0.9413, Corr: 0.9990

======================================================================
   SUMMARY
======================================================================

Task            ONN MSE      ONN Corr   MLP MSE      MLP Corr
------------------------------------------------------------
y = x²          0.3460       0.9941     1.1941       0.9979
y = sin(x)      0.0571       0.6791     0.2499       -0.9941
y = x² + x      1.4136       0.9999     0.9413       0.9990

Discovered Formulas:
  y = x²: 0.81*(x0 * x0) + (-0.77*(x0)²) + 0.19*agg(x0) + 0.13*ln(x0)
  y = sin(x): 0.02*x0 - 0.29*agg(x0) + 0.19
  y = x² + x: x0^2 + 0.7*agg(x0) + 1.84

======================================================================
   TEST COMPLETE
======================================================================
