import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.metrics import mutual_info_score


def pairwise_mutual_information(
    df: pd.DataFrame,
    *,
    discrete_columns: Sequence[str] | None = None,
    n_neighbors: int = 3,
    random_state: int | None = 42,
    missing: Literal["raise", "drop"] = "raise",
    diagonal: Literal["nan", "self"] = "nan",
) -> pd.DataFrame:
    """Estimate pairwise mutual information for mixed-type data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with unique string column names.
        Numeric columns are continuous unless listed in discrete_columns.
        Boolean and nonnumeric columns are discrete.
        Datetime, timedelta, and complex columns are unsupported.

    discrete_columns : sequence of str, optional
        Additional columns to treat as discrete, including numeric
        category codes.

    n_neighbors : int, default=3
        Number of neighbors for continuous MI estimation. For mixed
        pairs, fewer neighbors may be used within small categories.

    random_state : int or None, default=42
        Seed for reproducibility within the same software environment.

    missing : {"raise", "drop"}, default="raise"
        Reject missing values or drop rows containing missing values
        in any input column.

    diagonal : {"nan", "self"}, default="nan"
        "nan" leaves the diagonal undefined.
        "self" returns empirical entropy for discrete variables,
        infinity for nonconstant continuous variables, and zero
        for constants.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix of MI estimates in nats.

        Metadata in matrix.attrs includes:
        - n_samples_used: complete observations available.
        - n_rows_dropped: observations removed for missing values.
        - discrete_columns: columns treated as discrete.
        - singleton_categories: categories observed only once.
        - pair_sample_sizes: observations used for each comparison.
        - skipped_pairs: pairs that could not be estimated.

    Notes
    -----
    Discrete-discrete pairs use empirical mutual information.
    Continuous-discrete pairs use mutual_info_classif.
    Continuous-continuous pairs average estimates in both directions.

    For nonconstant mixed pairs, singleton observations are excluded
    explicitly. Estimates therefore describe the retained categories.
    If fewer than two categories remain, the result is NaN.

    Constant-variable comparisons return zero without estimation.
    Scores are not normalized and may exceed one.
    Categories follow pandas.factorize equality semantics.
    """
    # Validate input.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.shape[1] == 0:
        raise ValueError("df must contain at least one column.")

    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")

    if not all(isinstance(column, str) for column in df.columns):
        raise TypeError("Column names must be strings.")

    if (
        isinstance(n_neighbors, bool)
        or not isinstance(n_neighbors, Integral)
        or n_neighbors < 1
    ):
        raise ValueError("n_neighbors must be a positive integer.")

    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, Integral)
        or not 0 <= random_state < 2**32
    ):
        raise ValueError(
            "random_state must be an integer in [0, 2**32), or None."
        )

    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be 'raise' or 'drop'.")

    if diagonal not in {"nan", "self"}:
        raise ValueError("diagonal must be 'nan' or 'self'.")

    if isinstance(discrete_columns, str):
        raise TypeError(
            "discrete_columns must be a sequence of names, "
            "not a single string."
        )

    requested_discrete = (
        [] if discrete_columns is None else list(discrete_columns)
    )

    if not all(isinstance(column, str) for column in requested_discrete):
        raise TypeError("discrete_columns must contain string names.")

    explicit_discrete = set(requested_discrete)
    unknown_columns = explicit_discrete.difference(df.columns)

    if unknown_columns:
        raise ValueError(
            f"Unknown discrete columns: {sorted(unknown_columns)!r}"
        )

    # Handle missing values without modifying the input.
    if df.isna().to_numpy().any() and missing == "raise":
        raise ValueError(
            "df contains missing values. Use missing='drop' to remove "
            "rows containing missing values."
        )

    data = df.dropna() if missing == "drop" else df
    n_samples = len(data)

    if n_samples < 2:
        raise ValueError(
            "At least two complete observations are required."
        )

    columns = data.columns
    n_columns = len(columns)
    neighbors = int(n_neighbors)
    seed = None if random_state is None else int(random_state)

    arrays: dict[str, np.ndarray] = {}
    is_discrete: dict[str, bool] = {}
    is_constant: dict[str, bool] = {}
    singleton_categories: dict[str, list] = {}
    mixed_masks: dict[str, np.ndarray] = {}

    # Encode discrete columns and validate continuous columns.
    for column in columns:
        series = data[column]
        dtype = series.dtype

        if (
            pd.api.types.is_datetime64_any_dtype(dtype)
            or pd.api.types.is_timedelta64_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            raise TypeError(
                f"Column {column!r} has unsupported dtype {dtype!r}. "
                "Transform it explicitly before calculating MI."
            )

        numeric = pd.api.types.is_numeric_dtype(dtype)
        discrete = (
            column in explicit_discrete
            or pd.api.types.is_bool_dtype(dtype)
            or not numeric
        )
        is_discrete[column] = discrete

        numeric_values = None
        if numeric:
            numeric_values = series.to_numpy(dtype=np.float64)

            if not np.isfinite(numeric_values).all():
                raise ValueError(
                    f"Column {column!r} contains nonfinite numeric values."
                )

        if discrete:
            try:
                values, categories = pd.factorize(series, sort=False)
            except TypeError as exc:
                raise TypeError(
                    f"Column {column!r} must contain hashable categories."
                ) from exc

            counts = np.bincount(values, minlength=len(categories))
            singleton_categories[column] = categories[
                counts == 1
            ].tolist()

            # Shared mask for mixed comparisons involving this column.
            mixed_masks[column] = counts[values] >= 2
        else:
            values = numeric_values

        arrays[column] = values
        is_constant[column] = np.unique(values).size == 1

    result = np.full((n_columns, n_columns), np.nan, dtype=float)
    pair_sample_sizes = np.zeros((n_columns, n_columns), dtype=int)
    skipped_pairs: list[tuple[str, str]] = []
    warned_columns: set[str] = set()

    for i, column_a in enumerate(columns):
        values_a = arrays[column_a]

        if diagonal == "self":
            pair_sample_sizes[i, i] = n_samples

            if is_constant[column_a]:
                result[i, i] = 0.0
            elif is_discrete[column_a]:
                result[i, i] = mutual_info_score(values_a, values_a)
            else:
                result[i, i] = np.inf

        for j in range(i + 1, n_columns):
            column_b = columns[j]
            values_b = arrays[column_b]

            discrete_a = is_discrete[column_a]
            discrete_b = is_discrete[column_b]
            samples_for_pair = n_samples

            # Constant-variable comparisons.
            if is_constant[column_a] or is_constant[column_b]:
                score = 0.0

            # Discrete-discrete comparisons use all complete rows.
            elif discrete_a and discrete_b:
                score = mutual_info_score(values_a, values_b)

            # Continuous-discrete comparisons.
            elif discrete_a != discrete_b:
                categorical_column = (
                    column_a if discrete_a else column_b
                )
                categorical = values_a if discrete_a else values_b
                continuous = values_b if discrete_a else values_a

                keep = mixed_masks[categorical_column]
                samples_for_pair = int(keep.sum())
                n_removed = n_samples - samples_for_pair

                if n_removed and categorical_column not in warned_columns:
                    warnings.warn(
                        f"Column {categorical_column!r}: excluding "
                        f"{n_removed} singleton observations from its "
                        "nonconstant mixed-type comparisons. These "
                        "estimates describe the retained categories.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_columns.add(categorical_column)

                categorical = categorical[keep]
                continuous = continuous[keep]

                if np.unique(categorical).size < 2:
                    warnings.warn(
                        f"Cannot estimate MI between {column_a!r} and "
                        f"{column_b!r}: fewer than two categories remain "
                        "after excluding singleton observations. "
                        "Returning NaN.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    skipped_pairs.append((column_a, column_b))
                    # Zero indicates that no estimate was produced.
                    continue

                if np.unique(continuous).size == 1:
                    score = 0.0
                else:
                    score = mutual_info_classif(
                        continuous.reshape(-1, 1),
                        categorical,
                        discrete_features=False,
                        n_neighbors=neighbors,
                        random_state=seed,
                    )[0]

            # Continuous-continuous comparisons.
            else:
                if neighbors >= n_samples:
                    raise ValueError(
                        "For nonconstant continuous pairs, n_neighbors "
                        "must be smaller than the number of observations "
                        f"({n_samples})."
                    )

                score_ab = mutual_info_regression(
                    values_a.reshape(-1, 1),
                    values_b,
                    discrete_features=False,
                    n_neighbors=neighbors,
                    random_state=seed,
                )[0]

                score_ba = mutual_info_regression(
                    values_b.reshape(-1, 1),
                    values_a,
                    discrete_features=False,
                    n_neighbors=neighbors,
                    random_state=seed,
                )[0]

                score = (score_ab + score_ba) / 2.0

            result[i, j] = result[j, i] = float(score)
            pair_sample_sizes[i, j] = pair_sample_sizes[j, i] = (
                samples_for_pair
            )

    matrix = pd.DataFrame(
        result,
        index=columns.copy(),
        columns=columns.copy(),
    )

    # Store plain lists rather than nested DataFrames in attrs.
    matrix.attrs.update(
        units="nats",
        n_samples_used=n_samples,
        n_rows_dropped=len(df) - n_samples,
        discrete_columns=[
            column for column in columns if is_discrete[column]
        ],
        singleton_categories={
            column: categories
            for column, categories in singleton_categories.items()
            if categories
        },
        pair_sample_sizes=pair_sample_sizes.tolist(),
        skipped_pairs=skipped_pairs,
    )

    return matrix