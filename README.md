# Pairwise Mutual Information for Mixed-Type Data

A Python utility for estimating pairwise mutual information (MI) between continuous and categorical variables in pandas DataFrames. Built on NumPy, pandas, and scikit-learn, it produces a symmetric MI matrix for exploring statistical dependencies, including nonlinear relationships that linear correlation may miss.

The implementation supports continuous–continuous, categorical–categorical, and categorical–continuous comparisons. All scores are expressed in **nats**, providing consistent units across variable types.

### Features

- Automatic variable-type detection, with explicit overrides for numeric category codes.
- Empirical MI estimation for categorical pairs.
- Nearest-neighbor MI estimation for pairs involving continuous variables.
- Averaging of both estimation directions for continuous pairs.
- Configurable missing-value handling and diagonal values.
- Reproducible estimates through a configurable random seed.
- Metadata documenting sample sizes, singleton categories, and skipped comparisons.

### Handling Rare Categories

For categorical–continuous comparisons, observations belonging to categories represented only once are excluded from the affected pair. A warning reports these exclusions. The resulting estimate describes the retained categories; if fewer than two categories remain, the comparison returns `NaN`.

Categorical–categorical and continuous–continuous comparisons retain all complete observations.

### Interpretation and Limitations

MI measures statistical dependence and does not indicate its direction or establish causality. Scores are not normalized and may exceed one. Estimates depend on sample size, category frequencies, and the number of neighbors, so comparisons require care—particularly when different pairs use different observations.

The diagonal is `NaN` by default. Optional self-information values are available, but continuous self-MI should not be interpreted as a finite reference score.