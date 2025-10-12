import wilds
import torch
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds import get_dataset
from wilds.common.metrics.metric import Metric
from wilds.common.utils import maximum, minimum


class CalibrationError(Metric):
    """
    Computes calibration error for probabilistic predictions.
    Supports Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """

    def __init__(self, n_bins=15, mode="ece"):
        """
        Args:
            - n_bins (int): Number of bins to discretize [0, 1] interval.
            - mode (str | float):
                'ece' for Expected Calibration Error,
                'mce' for Maximum Calibration Error,
        """
        self.n_bins = n_bins
        self.mode = mode
        name = f"calibration_error-{mode}"
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        """
        Computes element-wise calibration error.
        Args:
            - y_pred (Tensor): Predicted logits (shape: [batch_size, n_classes]).
            - y_true (Tensor): Ground truth labels (shape: [batch_size]).
        Returns:
            - calibration_error (Tensor): Scalar calibration error.
        """
        y_pred = y_pred.float()
        y_true = y_true.float()
        y_prob = torch.softmax(y_pred, dim=1)
        confidences, predictions = torch.max(y_prob, dim=1)
        correctness = (predictions == y_true).float()

        # Split predictions into half-interval bins [bin_lower, bin_upper)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=y_true.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        # each prediction is mapped to corresponding confidence score's lower bin boundary
        bin_indices = torch.searchsorted(bin_boundaries, confidences, right=True).clamp(
            0, self.n_bins - 1
        )

        # Compute per-bin accuracy and average confidence
        print(f"{correctness.dtype=}")
        bin_corr_sum = torch.zeros(self.n_bins, device=y_true.device).scatter_add_(
            0, bin_indices, correctness
        )
        bin_conf_sum = torch.zeros(self.n_bins, device=y_true.device).scatter_add_(
            0, bin_indices, confidences
        )
        bin_counts = torch.zeros(self.n_bins, device=y_true.device).scatter_add_(
            0, bin_indices, torch.ones_like(confidences)
        )

        # Average while avoiding division by zero
        valid_bins = bin_counts > 0
        bin_acc = torch.where(
            valid_bins,
            bin_corr_sum / bin_counts,
            torch.tensor(0.0, device=y_true.device),
        )
        bin_conf = torch.where(
            valid_bins,
            bin_conf_sum / bin_counts,
            torch.tensor(0.0, device=y_true.device),
        )

        # Compute calibration error per bin
        bin_errors = torch.abs(bin_acc - bin_conf)

        # Aggregate errors (ECE or MCE)
        if self.mode == "ece":
            calibration_error = (
                torch.sum(bin_errors * (bin_counts / bin_counts.sum()))
                if bin_counts.sum() > 0
                else torch.tensor(0.0, device=y_true.device)
            )
        elif self.mode == "mce":
            calibration_error = (
                bin_errors.max()
                if valid_bins.any()
                else torch.tensor(0.0, device=y_true.device)
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'ece' or 'mce'.")

        return calibration_error

    def worst(self, metrics):
        """
        Returns the worst-case calibration error (maximum).
        Args:
            - metrics (Tensor): Tensor of calibration errors.
        Returns:
            - worst_metric (Tensor): Maximum calibration error.
        """
        return maximum(metrics)


def get_wrapped_dataset(*args, **kwargs):
    print("\n\nWarning: wrapping dataset into ClassificationUncertaintyDataset!\n\n")
    dataset = get_dataset(*args, **kwargs)
    dataset._eval = dataset.eval
    dataset.uncertainty_metrics = [
        CalibrationError(n_bins=15, mode="ece"),
        CalibrationError(n_bins=15, mode="mce"),
    ]

    def eval_unc(y_pred, y_true, metadata=None, prediction_fn=None):
        results = {}
        result_str = ""
        for metric in dataset.uncertainty_metrics:
            results.update(metric.compute(y_pred, y_true))
            result_str += f"{metric.name}: {results[metric.agg_metric_field]:.3f}\n"

        # if prediction_fn:
        #     y_pred = prediction_fn(y_pred)

        perf_res = dataset._eval(y_pred, y_true, metadata, prediction_fn=prediction_fn)
        results.update(perf_res[0])
        result_str += perf_res[1]
        return results, result_str

    dataset.eval = eval_unc

    return dataset


def main():
    import sys

    sys.path.insert(0, "wilds_git/examples")  # path to wilds examples repo
    import run_expt

    wilds.get_dataset = get_wrapped_dataset

    run_expt.main()


if __name__ == "__main__":
    main()
