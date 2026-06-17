"""Statistical comparison helpers for batch report analysis."""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StatResult:
    metric: str
    comparison: str
    group_a: str
    group_b: Optional[str]
    test: str
    statistic: float
    p_value: float
    significant: bool
    stratum: Optional[str] = None


STAT_FACTOR_COLUMNS = ("genotype", "condition", "timepoint")

TEST_LABELS = {
    "auto": "Automatic (recommended)",
    "mannwhitney": "Mann-Whitney U (2 groups)",
    "kruskal": "Kruskal-Wallis (3+ groups)",
    "anova": "One-way ANOVA (3+ groups, assumes normality)",
}

TEST_TOOLTIPS = {
    "auto": (
        "Picks Mann-Whitney U for 2 groups, Kruskal-Wallis for 3 or more. "
        "Pairwise follow-ups always use Mann-Whitney U with Bonferroni correction."
    ),
    "mannwhitney": (
        "Non-parametric test for exactly 2 independent groups. "
        "Use when comparing two genotypes, conditions, or timepoints."
    ),
    "kruskal": (
        "Non-parametric test for 3 or more groups. "
        "With only 2 groups, Mann-Whitney U is used instead."
    ),
    "anova": (
        "Parametric test for 3 or more groups; assumes roughly normal distributions. "
        "With 2 groups, Welch t-test is used instead."
    ),
}


def test_display_name(test_key: str) -> str:
    return TEST_LABELS.get(test_key, test_key)


def describe_test_choice(test: str, n_groups: int) -> str:
    if n_groups < 2:
        return "Need at least 2 groups with data in each comparison block."
    if test == "auto":
        if n_groups == 2:
            return "Automatic: Mann-Whitney U for 2 groups."
        return "Automatic: Kruskal-Wallis omnibus + Mann-Whitney pairwise comparisons."
    if test == "mannwhitney":
        if n_groups == 2:
            return "Mann-Whitney U compares the two groups directly."
        return "Mann-Whitney U is for 2 groups only; omnibus uses Kruskal-Wallis here."
    if test == "kruskal":
        if n_groups == 2:
            return "2 groups: Mann-Whitney U is used (equivalent non-parametric pair test)."
        return "Kruskal-Wallis tests whether any group differs (non-parametric)."
    if test == "anova":
        if n_groups == 2:
            return "2 groups: Welch t-test is used (parametric pair test)."
        return "One-way ANOVA tests whether any group differs (assumes normality)."
    return ""


def format_stat_config(
    compare_col: str,
    within_col: Optional[str],
    and_within_col: Optional[str],
    test: str,
    alpha: float,
) -> str:
    parts = [f"Compare: {compare_col}"]
    if within_col:
        parts.append(f"within each {within_col}")
    if and_within_col:
        parts.append(f"and within each {and_within_col}")
    parts.append(f"Test: {test_display_name(test)} (α={alpha})")
    return " | ".join(parts)


def _normalize_col(value: Optional[str]) -> Optional[str]:
    if not value or value == "None" or value == "(none)":
        return None
    return value


def _iter_strata(df: pd.DataFrame, within_col: Optional[str], and_within_col: Optional[str]):
    within_col = _normalize_col(within_col)
    and_within_col = _normalize_col(and_within_col)
    if not within_col or within_col not in df.columns:
        yield None, df
        return
    for w_val, sub in df.groupby(within_col, observed=False):
        if and_within_col and and_within_col in sub.columns:
            for a_val, sub2 in sub.groupby(and_within_col, observed=False):
                yield f"{within_col}={w_val}, {and_within_col}={a_val}", sub2
        else:
            yield f"{within_col}={w_val}", sub


def p_value_stars(p_value: float, alpha: float = 0.05) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < alpha:
        return "*"
    return "ns"


def _clean_values(series) -> np.ndarray:
    return pd.Series(series).dropna().astype(float).values


def _resolve_omnibus_test(test: str, n_groups: int) -> str:
    if test == "auto":
        return "mannwhitney" if n_groups == 2 else "kruskal"
    if test == "mannwhitney" and n_groups > 2:
        return "kruskal"
    if test in ("kruskal", "anova") and n_groups == 2:
        return test
    return test


def _run_two_group_test(a, b, test: str):
    a_vals = _clean_values(a)
    b_vals = _clean_values(b)
    if len(a_vals) < 1 or len(b_vals) < 1:
        return None

    if test in ("auto", "mannwhitney", "kruskal"):
        stat, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        label = "Mann-Whitney U"
        if test == "kruskal":
            label = "Mann-Whitney U (2-group equivalent of Kruskal-Wallis)"
        return label, float(stat), float(p)

    if test in ("anova", "ttest"):
        if len(a_vals) < 2 or len(b_vals) < 2:
            return None
        stat, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        label = "Welch t-test"
        if test == "anova":
            label = "Welch t-test (2-group equivalent of ANOVA)"
        return label, float(stat), float(p)

    return None


def _run_omnibus(groups: dict, test: str):
    arrays = [v for v in groups.values() if len(v) >= 1]
    n_groups = len(arrays)
    if n_groups < 2:
        return None

    resolved = _resolve_omnibus_test(test, n_groups)

    if resolved == "kruskal":
        stat, p = stats.kruskal(*arrays)
        return "Kruskal-Wallis", float(stat), float(p)
    if resolved == "anova":
        if any(len(a) < 2 for a in arrays):
            return None
        stat, p = stats.f_oneway(*arrays)
        return "One-way ANOVA", float(stat), float(p)
    if resolved == "mannwhitney" and n_groups == 2:
        stat, p = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        return "Mann-Whitney U", float(stat), float(p)
    return None


def _pairwise_comparisons(
    groups: dict,
    metric: str,
    comparison: str,
    alpha: float,
    stratum: Optional[str] = None,
) -> List[StatResult]:
    results = []
    names = sorted(groups.keys(), key=str)
    n_pairs = max(len(list(combinations(names, 2))), 1)

    for ga, gb in combinations(names, 2):
        outcome = _run_two_group_test(groups[ga], groups[gb], "mannwhitney")
        if outcome is None:
            continue
        test_name, stat, p = outcome
        adj_p = min(p * n_pairs, 1.0)
        results.append(
            StatResult(
                metric=metric,
                comparison=comparison,
                group_a=str(ga),
                group_b=str(gb),
                test=f"{test_name} (Bonferroni)",
                statistic=stat,
                p_value=adj_p,
                significant=adj_p < alpha,
                stratum=stratum,
            )
        )
    return results


def _compare_on_factor(
    df: pd.DataFrame,
    metric: str,
    factor_col: str,
    test: str,
    alpha: float,
    stratum: Optional[str] = None,
) -> List[StatResult]:
    if factor_col not in df.columns or metric not in df.columns:
        return []

    subset = df[[factor_col, metric]].dropna()
    if subset.empty:
        return []

    groups = {
        str(name): _clean_values(grp[metric])
        for name, grp in subset.groupby(factor_col, observed=False)
        if len(_clean_values(grp[metric])) > 0
    }
    if len(groups) < 2:
        return []

    comparison = f"by {factor_col}"
    if len(groups) == 2:
        names = sorted(groups.keys(), key=str)
        outcome = _run_two_group_test(groups[names[0]], groups[names[1]], test)
        if outcome is None:
            return []
        test_name, stat, p = outcome
        return [
            StatResult(
                metric=metric,
                comparison=comparison,
                group_a=names[0],
                group_b=names[1],
                test=test_name,
                statistic=stat,
                p_value=p,
                significant=p < alpha,
                stratum=stratum,
            )
        ]

    omnibus = _run_omnibus(groups, test)
    results = []
    if omnibus:
        test_name, stat, p = omnibus
        results.append(
            StatResult(
                metric=metric,
                comparison=comparison,
                group_a="(omnibus)",
                group_b=None,
                test=test_name,
                statistic=stat,
                p_value=p,
                significant=p < alpha,
                stratum=stratum,
            )
        )
    elif test == "mannwhitney":
        results.append(
            StatResult(
                metric=metric,
                comparison=comparison,
                group_a="(note)",
                group_b=None,
                test="Mann-Whitney U applies to 2 groups; Kruskal-Wallis could not be computed",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                stratum=stratum,
            )
        )
    results.extend(
        _pairwise_comparisons(groups, metric, comparison, alpha, stratum)
    )
    return results


def run_comparisons_structured(
    df: pd.DataFrame,
    metric: str,
    compare_col: str,
    within_col: Optional[str] = None,
    and_within_col: Optional[str] = None,
    test: str = "auto",
    alpha: float = 0.05,
) -> List[StatResult]:
    if metric not in df.columns or compare_col not in df.columns:
        return []

    compare_col = _normalize_col(compare_col)
    within_col = _normalize_col(within_col)
    and_within_col = _normalize_col(and_within_col)

    if not compare_col:
        return []

    if and_within_col and not within_col:
        and_within_col = None

    results = []
    for stratum, sub_df in _iter_strata(df, within_col, and_within_col):
        results.extend(
            _compare_on_factor(
                sub_df,
                metric,
                compare_col,
                test,
                alpha,
                stratum=stratum,
            )
        )
    return results


def results_to_text(results: List[StatResult], alpha: float = 0.05) -> str:
    if not results:
        return (
            "No statistical comparisons could be computed.\n"
            "Common reasons: only one group in a stratum, too few plants per group, "
            "or missing values for the selected metric."
        )

    lines = [
        "Statistical Analysis Summary",
        "=" * 40,
        f"Significance threshold: α = {alpha}",
        "Legend: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant",
        "",
    ]

    current_stratum = object()
    for r in results:
        if r.stratum != current_stratum:
            current_stratum = r.stratum
            if r.stratum:
                lines.append(f"\n--- {r.stratum} ---")
            else:
                lines.append(f"\n--- {r.comparison} ---")

        stars = p_value_stars(r.p_value, alpha)
        if r.group_b is None:
            lines.append(
                f"  {r.test}: statistic={r.statistic:.4f}, p={r.p_value:.4g} {stars}"
            )
        else:
            sig = "significant" if r.significant else "not significant"
            lines.append(
                f"  {r.group_a} vs {r.group_b}: {r.test}, "
                f"statistic={r.statistic:.4f}, p={r.p_value:.4g} {stars} ({sig})"
            )

    return "\n".join(lines)


def results_to_dataframe(results: List[StatResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "metric": r.metric,
                "comparison": r.comparison,
                "stratum": r.stratum or "",
                "group_a": r.group_a,
                "group_b": r.group_b or "",
                "test": r.test,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "significant": r.significant,
                "stars": p_value_stars(r.p_value),
            }
            for r in results
        ]
    )
