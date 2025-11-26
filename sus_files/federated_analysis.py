"""Federated AFib analysis using FLAME.

This analyzer performs logistic regression analysis on AFib (Vorhofflimmern/Atrial Fibrillation)
data across multiple nodes. Each node analyzes its local cohort and diagnosis data stored in
S3/MinIO, and results are aggregated centrally.

The analysis implements an iterative GLM fitter (Iteratively Reweighted Least Squares),
mimicking the approach of DataSHIELD's `ds.glm` function. It is mathematically
equivalent to pooling the data and fitting a single model.

The analysis:
1. Loads Cohort_extended.csv and Diagnoses_extended.csv from each node's S3 bucket.
2. Prepares an analysis dataset with diagnosis flags and unified NT-proBNP values.
3. Iteratively fits logistic regression models for different subcohorts by passing
   score vectors and information matrices between nodes and an aggregator.
4. The aggregator combines these to update model coefficients until convergence.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List
import io

from flame.star import StarModel, StarAnalyzer, StarAggregator

import pandas as pd
import numpy as np
import warnings

from afib_analysis_utils import (
    AFibAnalyzerMixin,
    AFibAggregatorMixin,
    MAX_ITERATIONS,
    run_analysis_iteration,
    run_aggregation_iteration,
)

warnings.filterwarnings("ignore")

__author__ = "Jules Kreuer, jules.kreuer@uni-tuebingen.de"
__version__ = "0.0.0"

# S3 object keys to analyze (must be present in each node's bucket)
# Set to None to analyze all CSV files
COHORT_KEY = "Cohort_extended.csv"
DIAGNOSES_KEY = "Diagnoses_extended.csv"

CSV_S3_KEYS: List[str] | None = [COHORT_KEY, DIAGNOSES_KEY]


class AFibAnalyzer(StarAnalyzer, AFibAnalyzerMixin):
    """Analyzer that performs one round of iterative GLM fitting on FLAME."""

    def __init__(self, flame):  # type: ignore[no-untyped-def]
        super().__init__(flame)
        self.cohort_df = None
        self.diagnoses_df = None
        self.analysis_df = None
        self.subcohorts: Dict[str, pd.DataFrame] = {}
        self.node_id = None

    def _load_and_prepare_data(self, data: List[Dict[str, Any]]):
        """Load, prepare, and cache the analysis dataframe and subcohorts."""
        # This method is designed to run only once per job
        if self.analysis_df is not None:
            return

        # Load CSV files from S3 data
        cohort_content, diagnoses_content = self._extract_s3_content(data)
        if cohort_content is None or diagnoses_content is None:
            raise RuntimeError(
                "Missing required CSV files (Cohort_extended.csv or Diagnoses_extended.csv)"
            )

        self.cohort_df = self._load_csv_from_bytes(cohort_content, COHORT_KEY)
        self.diagnoses_df = self._load_csv_from_bytes(diagnoses_content, DIAGNOSES_KEY)

        self._prepare_analysis_data()
        self._filter_data()
        self._prepare_subcohorts()

    def _extract_s3_content(self, data: List[Dict[str, Any]]):
        """Extract cohort and diagnoses file content from S3 data payload."""
        cohort_content = None
        diagnoses_content = None
        for objects in data:
            for fname, content in objects.items():
                if COHORT_KEY in fname:
                    cohort_content = content
                elif DIAGNOSES_KEY in fname:
                    diagnoses_content = content
        return cohort_content, diagnoses_content

    def _load_csv_from_bytes(self, content: bytes | str, filename: str) -> pd.DataFrame:
        """Load CSV from bytes content."""
        if isinstance(content, str):
            content = content.encode("utf-8")

        return pd.read_csv(io.BytesIO(content), sep=";", encoding="utf-8")

    def analysis_method(
        self,
        data: List[Dict[str, Any]],
        aggregator_results: Dict[str, Any] | List[Dict[str, Any]] | None,
    ) -> Dict[str, Any]:
        """Perform one round of federated GLM fitting."""
        self.node_id = self.flame.get_id()

        # Load and prepare data (cached after first call)
        self._load_and_prepare_data(data)

        # STAR model wraps aggregator results in a list - unwrap it
        # aggregator_results comes as a list from await_intermediate_data().values()
        unwrapped_results: Dict[str, Any] | None = None
        if isinstance(aggregator_results, list) and len(aggregator_results) > 0:
            unwrapped_results = aggregator_results[0]
        elif isinstance(aggregator_results, dict):
            unwrapped_results = aggregator_results

        return run_analysis_iteration(self, unwrapped_results, self.node_id)


class AFibAggregator(StarAggregator, AFibAggregatorMixin):
    """Aggregator that manages iterative GLM fitting."""

    def __init__(self, flame):  # type: ignore[no-untyped-def]
        super().__init__(flame)
        self.model_states = {}  # Stores beta, deviance, convergence status for each model
        self.summary_stats = {}
        self.iteration = 0

    def aggregation_method(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results and perform one IRLS update."""
        print(f"[Aggregator] aggregation_method called with {len(analysis_results)} results")
        return run_aggregation_iteration(self, analysis_results)

    def has_converged(self, result, last_result, num_iterations):  # type: ignore[no-untyped-def]
        """Check if all models have converged or max iterations reached."""
        if num_iterations >= MAX_ITERATIONS:
            return True

        model_states = result.get("model_states", {})
        if not model_states:
            return True

        all_converged = all(state.get("converged", False) for state in model_states.values())
        return all_converged


def main():
    """Configure and run the federated AFib analysis."""
    StarModel(
        analyzer=AFibAnalyzer,
        aggregator=AFibAggregator,
        data_type="s3",
        query=CSV_S3_KEYS,
        simple_analysis=False,  # Enable iterative analysis
        output_type="str",  # Use get_result() for final output
    )


if __name__ == "__main__":  # pragma: no cover
    main()
