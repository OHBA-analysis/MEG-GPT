"""
Layers for the objective functions and optimizations.

Mathematical Notation:
    - B : batch size
    - L : sequence length
    - C : number of channels
    - N_t : number of tokens
"""

# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class CrossEntropyLoss(nn.Module):
    """
    Module for calculating cross-entropy loss and top-k accuracies.

    Parameters
    ----------
    loss_sequence_length : int
        Length of the input sequence (i.e., number of tokens) for
        which to compute the loss.
    top_k : List[int], optional
        List of top-k values for accuracy calculation.
    """
    def __init__(
        self,
        loss_sequence_length: int,
        top_k: Optional[List[int]] = None,
    ):
        super().__init__()
        self.loss_sequence_length = loss_sequence_length
        self.top_k = top_k or [1]

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates the cross-entropy loss and top-k metrics for the batch.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted logits of shape (B, L_out, C, N_t).
        y_true : torch.Tensor
            Ground truth labels of shape (B, L, C).

        Returns
        -------
        loss : torch.Tensor
            The computed cross-entropy loss, summed over channels and averaged
            over batch and time. Shape is (1,).
        y_pred_loss : torch.Tensor
            The sliced predicted logits for the loss computation.
            Shape is (B, loss_sequence_length, C, N_t).
        metrics : Dict[str, torch.Tensor]
            Dictionary containing the top-k accuracy metrics.
        """
        # Slice the last `loss_sequence_length` tokens
        y_pred_loss = y_pred[:, -self.loss_sequence_length:, :, :]
        y_true_loss = y_true[:, -self.loss_sequence_length:, :]

        # Compute cross-entropy loss
        y_pred_loss_tp = y_pred_loss.permute(0, 3, 1, 2)  # shape: (B, N_t, loss_sequence_length, C)
        ce_loss = F.cross_entropy(y_pred_loss_tp, y_true_loss)  # mean over batch, time, and channels
        # shape: (B, loss_sequence_length, C)

        # Compute top-k accuracies
        metrics = {}
        for k in self.top_k:
            metrics[f"top_{k}_acc"] = self._calculate_top_k_accuracy(
                y_pred_loss, y_true_loss, k
            )

        return loss.unsqueeze(0), y_pred_loss, metrics

    @staticmethod
    def _calculate_top_k_accuracy(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Calculates the batch-level top-k accuracy.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted logits for loss calculation.
        y_true : torch.Tensor
            Ground truth labels for loss calculation.
        k : int
            The top-k value for accuracy calculation.

        Returns
        -------
        top_k_acc : torch.Tensor
            The computed top-k accuracy.
        """
        # Get indices of top-k predictions along the vocabulary dimension
        _, top_k_indices = torch.topk(y_pred, k, dim=-1)
        # shape: (B, loss_sequence_length, C, k)

        # Expand y_true to match top_k_indices shape
        y_true_expanded = y_true.unsqueeze(-1)
        # shape: (B, loss_sequence_length, C, 1)

        # Check if the ground truth index is present in the top-k predictions
        correct = torch.eq(top_k_indices, y_true_expanded).any(dim=-1)
        # shape: (B, loss_sequence_length, C)

        # Calculate the mean accuracy
        top_k_acc = correct.float().mean()

        return top_k_acc


class LyapunovLoss(nn.Module):
    """
    Module for calculating the Lyapunov regularization loss
    using a learned Lyapunov function V(z) in the embedding space.

    Parameters
    ----------
    loss_sequence_length : int
        Number of tokens/time steps to calculate the loss for.
    input_features : int
        Number of input features for the MLP layer calculating Lyapunov values.
    beta : float
        Weighting for the Lyapunov loss.
    mu : float
        Weighting for the V(0) anchoring term.
    collapse_weight : float
        Weight for anti-collapse regularization on V statistics.
    collapse_target_mean : float
        Minimum target mean of V over batch/time.
    collapse_target_var : float
        Minimum target variance of V over batch/time.
    dim : int
        Dimension of the Lyapunov function output.
    """
    def __init__(
        self,
        loss_sequence_length: int,
        input_features: int,
        beta: float = 1.0,
        mu: float = 10.0,
        collapse_weight: float = 0.0,
        collapse_target_mean: float = 1e-4,
        collapse_target_var: float = 1e-4,
        dim: int = 16,
    ):
        super().__init__()
        self.loss_sequence_length = loss_sequence_length
        self.dim = dim

        # Register non-trainable scalar tensors as buffers (device agnostic)
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("collapse_weight", torch.tensor(collapse_weight, dtype=torch.float32))
        self.register_buffer("collapse_target_mean", torch.tensor(collapse_target_mean, dtype=torch.float32))
        self.register_buffer("collapse_target_var", torch.tensor(collapse_target_var, dtype=torch.float32))

        # Initialize MLP to learn the Lyapunov energy landscape V(z)
        self.mlp = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.dim),
        )

    def _compute_v(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the Lyapunov energy values V(z_t).

        Parameters
        ----------
        z : torch.Tensor
            Latent/embedding tensor for which to compute the Lyapunov values.

        Returns
        -------
        v_t : torch.Tensor
            The computed Lyapunov values.
        """
        # Get input dimensions
        B, L_loss, C, E = z.shape

        # Calculate Lyapunov values
        z_flat = z.reshape(B, L_loss, C * E)
        v_t = self.mlp(z_flat)  # shape: (B, loss_sequence_length, dim)

        # Calculate squared norm over Lyapunov dimension to ensure V(z_t) >= 0
        v_t = torch.sum(v_t ** 2, dim=-1)  # shape: (B, loss_sequence_length)

        return v_t

    def forward(
        self,
        z_pred_t_plus_one: torch.Tensor,
        z_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates the Lyapunov loss to enforce dynamical stability.

        Parameters
        ----------
        z_pred_t_plus_one : torch.Tensor
            The predicted latent embedding for the next time step (t + 1).
        z_t : torch.Tensor
            The known input latent embedding for the current time step (t).

        Returns
        -------
        loss : torch.Tensor
            The computed Lyapunov loss of shape (1,).
        lyapunov_violation : torch.Tensor
            The computed Lyapunov violation of shape (B, loss_sequence_length).
        metrics : Dict[str, torch.Tensor]
            A dictionary containing various metrics related to the loss computation.
        """
        # Truncate to the specified loss sequence length
        z_pred_t_plus_one = z_pred_t_plus_one[:, -self.loss_sequence_length:, :, :]
        z_t = z_t[:, -self.loss_sequence_length:, :, :]
        # shape: (B, loss_sequence_length, C, E)

        # Compute Lyapunov energy for current and predicted next state
        v_t_plus_one = self._compute_v(z_pred_t_plus_one)
        v_t = self._compute_v(z_t)

        # Calculate Lyapunov violation: V(t + 1) - V(t) <= 0
        # If energy increases, the ReLU triggers a penalty.
        lyapunov_violation = F.relu(v_t_plus_one - v_t)  # shape: (B, loss_sequence_length)
        # NOTE: Inputs are already temporally aligned before this layer.
        stability_loss = torch.mean(lyapunov_violation)  # mean over batch and sequence length
        # shape: scalar

        # Anchor Lyapunov scale by pushing V(0) -> 0
        # NOTE: This grounds the learned function, forcing the origin (zero state) 
        #       to be the global minimum/zero-energy reference point.
        v0 = self._compute_v(torch.zeros_like(z_pred_t_plus_one))
        v0_loss = torch.mean(v0)

        # Get energy statistics to prevent trivial collapse
        v_values = torch.cat([v_t, v_t_plus_one], dim=1)  # shape: (B, 2 * loss_sequence_length)
        v_mean = torch.mean(v_values)
        v_var = torch.var(v_values, correction=1)
        # NOTE: You can set `correction=0` to match tf.math.reduce_variance

        # Normalized squared hinge for maintaining anti-collapse pressure scale-stable
        # NOTE: If targets are small, the penalty is still O(1) when collapsed.
        #       This prevents the network from simply outputting 0 for everything.
        mean_gap = F.relu(1.0 - v_mean / torch.clamp(self.collapse_target_mean, min=1e-12))
        var_gap = F.relu(1.0 - v_var / torch.clamp(self.collapse_target_var, min=1e-12))

        collapse_loss = torch.square(mean_gap) + torch.square(var_gap)
        collapse_ratio = collapse_loss / torch.clamp(stability_loss, min=1e-12)

        # Aggregate final loss
        loss = stability_loss + self.mu * v0_loss + self.collapse_weight * collapse_loss
        loss = loss * self.beta

        # Gather metrics (detached from computational graph)
        metrics = {
            "lyapunov_loss": loss.detach().item(),
            "lyapunov_stability_loss": stability_loss.detach().item(),
            "lyapunov_v0_loss": v0_loss.detach().item(),
            "lyapunov_mu": self.mu.item(),
            "lyapunov_collapse_loss": collapse_loss.detach().item(),
            "lyapunov_collapse_ratio": collapse_ratio.detach().item(),
            "lyapunov_v_mean": v_mean.detach().item(),
            "lyapunov_v_var": v_var.detach().item(),
        }

        return loss.unsqueeze(0), lyapunov_violation, metrics


if __name__ == "__main__":
    # Define dummy dimensions for simple testing
    B = 2         # batch size
    L = 15        # full sequence length
    L_loss = 10   # sequence length to calculate loss for
    C = 4         # number of channels
    N_t = 50      # number of tokens (vocabulary size)
    E = 128       # embedding dimension

    # ------------------
    # Cross-Entropy Loss
    # ------------------
    print("Initializing dummy tensors...")
    # Unnormalized logits of shape (B, L, C, N_t)
    dummy_y_pred = torch.randn(B, L, C, N_t)

    # Ground truth token indices of shape (B, L, C)
    # Values must be integers in the range [0, N_t - 1]
    dummy_y_true = torch.randint(0, N_t, (B, L, C))

    # Instantiate the cross-entropy loss layer
    print(f"Instantiating CrossEntropyLoss with loss_sequence_length={L_loss}...")
    loss_fn = CrossEntropyLoss(loss_sequence_length=L_loss, top_k=[1, 3])

    # Execute forward pass
    print("Running forward pass...")
    loss, y_pred_loss, metrics = loss_fn(dummy_y_pred, dummy_y_true)

    # Validate outputs
    assert loss.shape == (1,), f"Expected loss shape (1,), but got {loss.shape}."
    assert y_pred_loss.shape == (B, L_loss, C, N_t), \
        f"Expected y_pred_loss shape {(B, L_loss, C, N_t)}, but got {y_pred_loss.shape}."

    # Output results
    print("-" * 40)
    print(f"Loss value:      {loss.item():.4f}")
    print(f"Top-1 Accuracy:  {metrics['top_1_acc'].item():.4f}")
    print(f"Top-3 Accuracy:  {metrics['top_3_acc'].item():.4f}")
    print("-" * 40)

    # -------------
    # Lyapunov Loss
    # -------------
    print("Initializing dummy tensors ...")
    z_pred_t_plus_one = torch.randn(B, L, C, E)
    z_t = torch.randn(B, L, C, E)

    # Instantiate Lyapunov loss layer
    print(f"Instantiating LyapunovLoss with loss_sequence_length={L_loss}...")
    input_features = C * E
    lyapunov_layer = LyapunovLoss(
        input_features=input_features, 
        loss_sequence_length=L_loss,
    )

    # Execute forward pass
    loss, lyapunov_fn, metrics_dict = lyapunov_layer(z_pred_t_plus_one, z_t)

    # Loss backpropagation
    loss.backward()

    # Output results
    print("-" * 40)
    print("Loss:", loss.item())
    print("Metrics:", metrics_dict)
    print("-" * 40)
