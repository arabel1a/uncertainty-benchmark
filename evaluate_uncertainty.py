import wilds
import torch
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds import get_dataset
from wilds.common.metrics.metric import Metric, ElementwiseMetric
from wilds.common.utils import maximum, minimum

class CalibrationError(ElementwiseMetric):
    """
    Computes calibration error for probabilistic predictions.
    Supports Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """

    def __init__(self, n_bins=15, norm='l1', mode='ece', name=None):
        """
        Args:
            - n_bins (int): Number of bins to discretize [0, 1] interval.
            - norm (str): Norm used to compare accuracy and confidence ('l1' or 'l2').
            - mode (str): 'ece' for Expected Calibration Error, 'mce' for Maximum Calibration Error.
            - name (str): Name of the metric.
        """
        self.n_bins = n_bins
        self.norm = norm
        self.mode = mode
        if name is None:
            name = f'calibration_error-{mode}-{norm}'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        """
        Computes element-wise calibration error.
        Args:
            - y_pred (Tensor): Predicted probabilities or logits (shape: [batch_size, n_classes] or [batch_size]).
            - y_true (Tensor): Ground truth labels (shape: [batch_size]).
        Returns:
            - calibration_error (Tensor): Scalar calibration error.
        """
        # Convert logits to probabilities if needed
        if y_pred.dim() == 2:  # Multi-class logits
            y_prob = torch.softmax(y_pred, dim=1)
            confidences, predictions = torch.max(y_prob, dim=1)
        else:  # Binary logits or already probabilities
            y_prob = torch.sigmoid(y_pred) if y_pred.dim() == 1 else y_pred
            confidences = y_prob
            predictions = (y_prob > 0.5).long()

        # Compute accuracy and bin contributions
        accuracies = (predictions == y_true).float()
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=y_true.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # Assign each prediction to a bin
        bin_indices = torch.searchsorted(bin_boundaries, confidences, right=False) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)

        # Compute per-bin accuracy and average confidence
        bin_acc_sum = torch.zeros(self.n_bins, device=y_true.device)
        bin_conf_sum = torch.zeros(self.n_bins, device=y_true.device)
        bin_counts = torch.zeros(self.n_bins, device=y_true.device)

        bin_acc_sum.scatter_add_(0, bin_indices, accuracies)
        bin_conf_sum.scatter_add_(0, bin_indices, confidences)
        bin_counts.scatter_add_(0, bin_indices, torch.ones_like(confidences))

        # Avoid division by zero
        valid_bins = bin_counts > 0
        bin_acc = torch.where(valid_bins, bin_acc_sum / bin_counts, torch.tensor(0., device=y_true.device))
        bin_conf = torch.where(valid_bins, bin_conf_sum / bin_counts, torch.tensor(0., device=y_true.device))

        # Compute calibration error per bin
        if self.norm == 'l1':
            bin_errors = torch.abs(bin_acc - bin_conf)
        elif self.norm == 'l2':
            bin_errors = (bin_acc - bin_conf) ** 2
        else:
            raise ValueError(f"Unsupported norm: {self.norm}. Use 'l1' or 'l2'.")

        # Aggregate errors (ECE or MCE)
        if self.mode == 'ece':
            calibration_error = torch.sum(bin_errors * (bin_counts / bin_counts.sum())) if bin_counts.sum() > 0 else torch.tensor(0., device=y_true.device)
        elif self.mode == 'mce':
            calibration_error = bin_errors.max() if valid_bins.any() else torch.tensor(0., device=y_true.device)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'ece' or 'mce'.")

        return calibration_error.expand(y_true.size(0))  # Return per-element for consistency with ElementwiseMetric

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
        CalibrationError(n_bins=15, norm='l1', mode='ece'),
        CalibrationError(n_bins=15, norm='l1', mode='mce'),
    ]

    def eval_unc(self, y_pred, y_true, metadata=None):
        results, result_str = dataset._eval(y_pred, y_true, metadata)
        for metric in self,uncertainty_metrics:
            results.update(metric.compute(y_pred, y_true))
            result_str += f"{metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        return results, result_str

    dataset.eval = eval_unc

    return dataset


if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'wilds_git/examples')  # path to wilds examples repo
    import run_expt

    wilds.get_dataset = get_wrapped_dataset

    run_expt.main()
