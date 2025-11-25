"""Shared utilities for AFib federated and local analysis.

This module contains common functions and classes used in both federated (FLAME-based)
and local AFib (Vorhofflimmern/Atrial Fibrillation) analysis implementations.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import statsmodels.api as sm

__author__ = "Jules Kreuer, jules.kreuer@uni-tuebingen.de"
__version__ = "0.0.0"

# Analysis parameters
MAX_ITERATIONS = 30
CONVERGENCE_THRESHOLD = 1e-6


class AFibAnalyzerMixin:
    """Mixin providing shared analysis methods for AFib analyzers."""

    def _prepare_analysis_data(self):
        """Prepare analysis dataframe with relevant columns and diagnosis flags."""
        if self.cohort_df is None or self.diagnoses_df is None:
            raise RuntimeError("Cohort or diagnoses DataFrame not loaded")

        # Create pivot of diagnoses to get one row per subject-encounter
        diagnosis_pivot = self.diagnoses_df.pivot_table(
            index=["subject", "encounter.id"],
            columns="code",
            aggfunc="size",
            fill_value=0,
        ).reset_index()

        # Merge with cohort data
        self.analysis_df = self.cohort_df.merge(
            diagnosis_pivot, on=["subject", "encounter.id"], how="left"
        )

        # Fill NaN diagnosis flags with 0
        diagnosis_cols = [
            col for col in self.analysis_df.columns if col.startswith("I") or col.startswith("E")
        ]
        self.analysis_df[diagnosis_cols] = self.analysis_df[diagnosis_cols].fillna(0)

        # Create binary diagnosis flags
        def has_diagnosis_pattern(pattern):
            """Check if any diagnosis column matches the pattern."""
            df = self.analysis_df
            if df is None:
                raise RuntimeError("Analysis DataFrame not prepared")
            matching_cols = [col for col in df.columns if any(p in col for p in pattern)]
            if matching_cols:
                return (df[matching_cols].sum(axis=1) > 0).astype(int)
            return pd.Series([0] * len(df), index=df.index)

        # Diagnosis flags
        self.analysis_df["IdiopathicHypotension"] = has_diagnosis_pattern(["I95.0"])
        self.analysis_df["AtrialFibrillation"] = has_diagnosis_pattern(
            [
                "I48.0",
                "I48.1",
                "I48.2",
                "I48.9",
            ]
        )
        self.analysis_df["MyocardialInfarction"] = has_diagnosis_pattern(
            [
                "I21",
                "I22",
                "I25.2",
            ]
        )
        self.analysis_df["HeartFailure"] = has_diagnosis_pattern(["I50"])
        self.analysis_df["Stroke"] = has_diagnosis_pattern(
            [
                "I60",
                "I61",
                "I62",
                "I63",
                "I64",
                "I69",
            ]
        )

        # Convert NT-proBNP value to numeric and apply unit conversion
        self.analysis_df["nt_pro_bnp_value"] = pd.to_numeric(
            self.analysis_df["NTproBNP.valueQuantity.value"],
            errors="coerce",
        )

        # Apply unit conversion
        self._convert_nt_pro_bnp_units()

    def _convert_nt_pro_bnp_units(self):
        """Convert NT-proBNP values to uniform unit pg/mL."""
        if self.analysis_df is None:
            raise RuntimeError("Analysis DataFrame not prepared")

        # Ensure required columns exist
        if "NTproBNP.unit" not in self.analysis_df.columns:
            self.analysis_df["NTproBNP.unit"] = np.nan
        if "NTproBNP.unitLabel" not in self.analysis_df.columns:
            self.analysis_df["NTproBNP.unitLabel"] = np.nan

        unit_col = self.analysis_df["NTproBNP.unit"]
        unit_label_col = self.analysis_df["NTproBNP.unitLabel"]

        # R-like behavior: all NA -> copy full unitLabel; else fill NA with unitLabel
        if unit_col.isna().all():
            self.analysis_df["NTproBNP.unit"] = unit_label_col
        else:
            self.analysis_df["NTproBNP.unit"] = unit_col.fillna(unit_label_col)

        self.analysis_df.dropna(subset=["NTproBNP.unit"], inplace=True)

        unit_factors = {
            "pg/ml": 1.0,  # default
            "ng/l": 1.0,
            "pg/dl": 0.01,
            "pg/100ml": 0.01,
            "pg%": 0.01,
            "pg/l": 0.001,
            "pmol/l": 8.457,  # See: https://journals.sagepub.com/doi/full/10.1258/acb.2007.007069 "NT-proBNP concentrations are expressed in picomoles/litre (for conversion to picograms/millilitre they are multiplied by 8.457)."
        }

        unit_lower = self.analysis_df["NTproBNP.unit"].str.lower()
        valid_mask = unit_lower.isin(unit_factors.keys())
        self.analysis_df = self.analysis_df[valid_mask].copy()

        conversion_factors = self.analysis_df["NTproBNP.unit"].str.lower().map(unit_factors)
        self.analysis_df["nt_pro_bnp_value"] *= conversion_factors

        self.analysis_df["NTproBNP.unit"] = "pg/mL"
        self.analysis_df["NTproBNP.unitLabel"] = "picogram per milliliter"

    def _filter_data(self):
        """Filter out missing values."""
        if self.analysis_df is None:
            raise RuntimeError("Analysis DataFrame not prepared")
        self.analysis_df = self.analysis_df[
            (self.analysis_df["nt_pro_bnp_value"].notna())
            & (self.analysis_df["nt_pro_bnp_value"] > 0)
            & (self.analysis_df["gender"].notna())
            & (self.analysis_df["age"].notna())
        ].copy()

    def _prepare_subcohorts(self):
        """Create and cache all subcohort dataframes for analysis."""
        if self.analysis_df is None:
            raise RuntimeError("Analysis DataFrame not prepared")

        # Recode gender once
        gender_map = {"male": 0, "female": 1}
        self.analysis_df["gender_numeric"] = (
            self.analysis_df["gender"].map(gender_map).fillna(2).astype(int)
        )

        # Subcohorts should NOT be filtered by outcome variable
        # They represent different populations where we model the outcome
        self.subcohorts = {
            "CONDITIONTEST": self.analysis_df.copy(),
            "CONDITION_AFIB": self.analysis_df.copy(),
            "CONDITION_HIS": self.analysis_df.copy(),
            "CONDITION_AFIB2": self.analysis_df[
                (self.analysis_df["MyocardialInfarction"] == 0) & (self.analysis_df["Stroke"] == 0)
            ].copy(),
            "CONDITION_HIS2": self.analysis_df[
                (self.analysis_df["MyocardialInfarction"] == 0) & (self.analysis_df["Stroke"] == 0)
            ].copy(),
            "CONDITION_AFIB3": self.analysis_df[
                (self.analysis_df["MyocardialInfarction"] == 0)
                & (self.analysis_df["Stroke"] == 0)
                & (self.analysis_df["HeartFailure"] == 0)
            ].copy(),
            # TODO: Demographic subcohorts, is existing but never used R analysis
            # "SUBSET_MALES": self.analysis_df[self.analysis_df["gender"] == "male"].copy(),
            # "SUBSET_FEMALES": self.analysis_df[self.analysis_df["gender"] == "female"].copy(),
            # "SUBSET_AGE_ABOVE_80": self.analysis_df[self.analysis_df["age"] > 80].copy(),
        }

    def _calculate_glm_iteration(
        self, df: pd.DataFrame, outcome: str, predictors: List[str], beta: np.ndarray
    ) -> Dict[str, Any] | None:
        """Calculate score vector and info matrix for one IRLS iteration."""
        df_model = df.copy()
        df_model["gender"] = df_model["gender_numeric"]
        df_model = df_model[[outcome] + predictors].dropna()

        if len(df_model) < 10 or df_model[outcome].nunique() < 2:
            return None

        y = df_model[outcome].values
        X = sm.add_constant(df_model[predictors], prepend=True).values

        # Logit function (inverse of expit)
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))

        # Calculate weights for info matrix
        W_diag = mu * (1 - mu)
        W = np.diag(W_diag)

        # Score vector and information matrix
        score_vector = X.T @ (y - mu)
        info_matrix = X.T @ W @ X

        # Deviance calculation
        # Ensure mu is not exactly 0 or 1 to avoid log(0)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        deviance = -2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))

        return {
            "score_vector": score_vector.tolist(),
            "info_matrix": info_matrix.tolist(),
            "n_samples": len(df_model),
            "deviance": deviance,
        }

    def _get_summary_stats(self):
        """Get summary statistics for the node (run once)."""
        if self.analysis_df is None:
            raise RuntimeError("Analysis DataFrame not prepared")
        return {
            "n_total": len(self.analysis_df),
            "nt_pro_bnp_mean": float(self.analysis_df["nt_pro_bnp_value"].mean()),
            "age_mean": float(self.analysis_df["age"].mean()),
            "atrial_fibrillation_count": int(self.analysis_df["AtrialFibrillation"].sum()),
            "heart_failure_count": int(self.analysis_df["HeartFailure"].sum()),
        }


class AFibAggregatorMixin:
    """Mixin providing shared aggregation methods for AFib aggregators."""

    def _initialize_model_states(self, analysis_results: List[Dict[str, Any]]):
        """Initialize state for all models based on the first results."""
        if self.model_states:
            return

        first_result = next((r for r in analysis_results if r["status"] == "success"), None)
        if not first_result:
            return

        for name, (outcome, predictors) in MODEL_DEFINITIONS.items():
            self.model_states[name] = {
                "beta": [0.0] * (len(predictors) + 1),
                "deviance": float("inf"),
                "converged": False,
                "iteration": 0,
                "predictors": ["intercept"] + predictors,
            }

    def _aggregate_summary_stats(self, analysis_results: List[Dict[str, Any]]):
        """Aggregate one-time summary statistics."""
        if self.summary_stats:
            return

        successful_results = [
            r
            for r in analysis_results
            if r["status"] == "success" and "summary_stats" in r and r["summary_stats"]
        ]
        if not successful_results:
            return

        # Calculate summary stats from available data
        total_n = sum(r["summary_stats"]["n_total"] for r in successful_results)
        if total_n == 0:
            return

        self.summary_stats = {
            "total_records": total_n,
            "nt_pro_bnp_weighted_mean": sum(
                r["summary_stats"]["n_total"] * r["summary_stats"]["nt_pro_bnp_mean"]
                for r in successful_results
            )
            / total_n,
            "age_weighted_mean": sum(
                r["summary_stats"]["n_total"] * r["summary_stats"]["age_mean"]
                for r in successful_results
            )
            / total_n,
            "total_atrial_fibrillation": sum(
                r["summary_stats"]["atrial_fibrillation_count"] for r in successful_results
            ),
            "total_heart_failure": sum(
                r["summary_stats"]["heart_failure_count"] for r in successful_results
            ),
            "n_nodes": len(successful_results),
        }

    def _perform_irls_update(self, model_name: str, all_iterations: List[Dict[str, Any]]):
        """Perform one IRLS update step for a given model.

        Args:
            model_name: Name of the model
            all_iterations: List of iteration results from all nodes

        Returns:
            True if model converged, False otherwise
        """
        if not all_iterations:
            return False

        state = self.model_states[model_name]
        total_info_matrix = np.sum([np.array(it["info_matrix"]) for it in all_iterations], axis=0)
        total_score_vector = np.sum([np.array(it["score_vector"]) for it in all_iterations], axis=0)
        total_deviance = np.sum([it["deviance"] for it in all_iterations])

        # IRLS update step
        try:
            inv_info_matrix = np.linalg.inv(total_info_matrix)
            beta_update = inv_info_matrix @ total_score_vector
            new_beta = np.array(state["beta"]) + beta_update

            # Check for convergence
            delta_deviance = abs(total_deviance - state["deviance"])
            converged = delta_deviance < CONVERGENCE_THRESHOLD

            state["beta"] = new_beta.tolist()
            state["deviance"] = total_deviance
            # Always calculate stderr for current state
            state["stderr"] = np.sqrt(np.diag(inv_info_matrix)).tolist()

            if converged:
                state["converged"] = True

            return converged

        except np.linalg.LinAlgError:
            state["error"] = "Singular matrix error during inversion."
            state["converged"] = True  # Stop iteration
            return True

    def get_result(self) -> str:
        """Format the final results into a JSON string."""
        final_models = {}
        for name, state in self.model_states.items():
            if "error" in state:
                final_models[name] = {
                    "status": "failed",
                    "error": state["error"],
                    "iterations": state["iteration"],
                }
            elif "stderr" in state and state["stderr"]:
                final_models[name] = {
                    "status": "success",
                    "coefficients": dict(zip(state["predictors"], state["beta"])),
                    "stderr": dict(zip(state["predictors"], state["stderr"])),
                    "iterations": state["iteration"],
                    "converged": bool(state.get("converged", False)),
                }
            else:
                final_models[name] = {
                    "status": "failed",
                    "error": "Did not converge",
                    "iterations": state["iteration"],
                }

        result = {
            "overall_status": "completed",
            "aggregated_summary": self.summary_stats,
            "aggregated_models": final_models,
        }
        return json.dumps(result, indent=2)


# Model definitions used in analysis
MODEL_DEFINITIONS = {
    # name, (outcome, [predictor_1, predictor_2, ...])
    "CONDITIONTEST": ("IdiopathicHypotension", ["nt_pro_bnp_value", "age", "gender"]),
    "CONDITION_AFIB": ("AtrialFibrillation", ["nt_pro_bnp_value", "age", "gender"]),
    "CONDITION_HIS": ("HeartFailure", ["nt_pro_bnp_value", "age", "gender"]),
    "CONDITION_AFIB2": ("AtrialFibrillation", ["nt_pro_bnp_value", "age", "gender"]),
    "CONDITION_HIS2": ("HeartFailure", ["nt_pro_bnp_value", "age", "gender"]),
    "CONDITION_AFIB3": ("AtrialFibrillation", ["nt_pro_bnp_value", "age", "gender"]),
    # For gender subsets, we exclude gender as a predictor since it's constant within the subset
    # "SUBSET_MALES": ("AtrialFibrillation", ["nt_pro_bnp_value", "age"]),
    # "SUBSET_FEMALES": ("AtrialFibrillation", ["nt_pro_bnp_value", "age"]),
    # For age >80 subset, we keep all predictors
    # "SUBSET_AGE_ABOVE_80": ("AtrialFibrillation", ["nt_pro_bnp_value", "age", "gender"]),
}


def run_analysis_iteration(
    analyzer: Any,
    aggregator_results: Dict[str, Any] | None,
    node_id: str | None = None,
) -> Dict[str, Any]:
    """Execute a single analysis iteration for a node.

    This is a shared method used by both local and federated analyses.

    NOTE: The caller must ensure that analyzer.subcohorts has been populated
    before calling this function (via _load_and_prepare_data or similar).

    Args:
        analyzer: The analyzer instance with prepared data and subcohorts
        aggregator_results: Results from the previous aggregation (or None on first iteration)
        node_id: Optional node identifier (for local analysis, comes from analyzer.node_id)

    Returns:
        Dictionary with iteration results including status, iteration_results, and summary_stats
    """
    import traceback

    if node_id is None:
        node_id = getattr(analyzer, "node_id", "unknown")

    # Handle aggregator_results safely - could be None, empty dict, or have missing keys
    iteration_num = 0
    if aggregator_results and "iteration" in aggregator_results:
        iteration_num = aggregator_results["iteration"]

    try:
        if not analyzer.subcohorts:
            return {
                "node_id": node_id,
                "status": "error",
                "error": "Data preparation failed",
            }

        model_states = {}
        if aggregator_results and "model_states" in aggregator_results:
            model_states = aggregator_results["model_states"]

        iteration_results = {}
        for name, (outcome, predictors) in MODEL_DEFINITIONS.items():
            subcohort_df = analyzer.subcohorts[name] if name in analyzer.subcohorts else None
            if subcohort_df is None or subcohort_df.empty:
                continue

            # Get current beta for this model, or initialize with zeros
            if name in model_states and "beta" in model_states[name]:
                current_beta = np.array(model_states[name]["beta"])
            else:
                current_beta = np.array([0] * (len(predictors) + 1))

            result = analyzer._calculate_glm_iteration(
                subcohort_df, outcome, predictors, current_beta
            )
            if result:
                print(f"[ANALYZER {node_id}] {name}: Computed score vector and info matrix")
                iteration_results[name] = result
            else:
                print(
                    f"[ANALYZER {node_id}] {name}: Calculation returned None (likely insufficient data)"
                )

        # Collect summary stats only on first iteration
        collect_summary = True
        if aggregator_results and "iteration" in aggregator_results:
            collect_summary = aggregator_results["iteration"] <= 1

        return {
            "node_id": node_id,
            "status": "success",
            "iteration_results": iteration_results,
            "summary_stats": analyzer._get_summary_stats() if collect_summary else None,
        }

    except Exception as e:
        return {
            "node_id": node_id,
            "status": "error",
            "error": f"Analysis failed: {type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def run_aggregation_iteration(
    aggregator: Any,
    analysis_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute a single aggregation iteration.

    This is a shared method used by both local and federated analyses.

    Args:
        aggregator: The aggregator instance with state and methods
        analysis_results: Results from all analyzers in this iteration

    Returns:
        Dictionary with aggregation results including model_states and summary
    """
    aggregator.iteration += 1
    print(f"\n[Iteration {aggregator.iteration}]")

    successful_nodes = [r for r in analysis_results if r["status"] == "success"]
    if not successful_nodes:
        return {"error": "No successful nodes in this iteration."}

    if aggregator.iteration == 1:
        aggregator._initialize_model_states(analysis_results)
        aggregator._aggregate_summary_stats(analysis_results)

    # Aggregate score vectors and info matrices for each model
    for name, state in aggregator.model_states.items():
        if state["converged"]:
            continue

        all_iterations = [
            r["iteration_results"][name]
            for r in successful_nodes
            if "iteration_results" in r and name in r["iteration_results"]
        ]

        if not all_iterations:
            continue

        converged = aggregator._perform_irls_update(name, all_iterations)
        state["iteration"] = aggregator.iteration
        if converged:
            print(f"  {name}: CONVERGED")

    # Check if all models have converged
    all_converged = all(state.get("converged", False) for state in aggregator.model_states.values())

    if all_converged:
        # Parse the JSON string result back to dict for consistency
        result_json = aggregator.get_result()
        return json.loads(result_json)

    # Continue iterations
    return {
        "iteration": aggregator.iteration,
        "model_states": aggregator.model_states,
        "summary": aggregator.summary_stats,
    }
