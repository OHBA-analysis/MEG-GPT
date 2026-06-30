"""
MEG-GPT generator class during inference time.

Mathematical Notation:
    - B   : batch size
    - L   : sequence length
    - C   : number of channels
    - E   : embedding dimension
    - N_t : number of tokens (vocabulary size)
    - F   : number of frequency bins
    - T   : number of time steps (time samples)
"""

# Import packages
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from tqdm.auto import trange
from typing import List, Optional, Union
from meg_gpt.utils.sampling import sample_from_logits


class MEGGPTGenerator:
    """
    MEG-GPT generator class providing autoregressive sampling logic
    for the MEG-GPT PyTorch Lightning module.

    Parameters
    ----------
    model : nn.Module
        The MEG-GPT model.
    tokenizer : pl.LightningModule
        The tokenizer LightningModule used for tokenization.
    """
    def __init__(self, model: nn.Module, tokenizer: pl.LightningModule):
        self.model = model
        self.tokenizer = tokenizer

        self.model.eval()
        self.tokenizer.eval()

    @property
    def device(self) -> torch.device:
        """Helper to infer the device from the model parameters."""
        return next(self.model.parameters()).device

    @torch.no_grad()
    def one_step_sample(
        self,
        x: torch.Tensor,
        extra_labels: List[torch.Tensor],
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        typical_p: Optional[float] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generates the next token in a sequence using autoregressive sampling.

        Parameters
        ----------
        x : torch.Tensor
            Input data sequence to the model. Shape is (B, L + 1, C).
        extra_labels : List[torch.Tensor]
            List of extra label tensors, each of shape (B, L + 1).
        top_p : float, optional
            The cumulative probability threshold for top-p sampling.
        top_k : int, optional
            The number of top logits to sample from.
        typical_p : float, optional
            The cumulative probability threshold for typical sampling (Meister et al., 2023).
        temperature : float, optional
            Scaling factor for the logits to control randomness. Must be > 0.

        Returns
        -------
        token_samples : torch.Tensor
            The sampled tokens of shape (B, C).
        """
        # Get token prediction logits
        outputs = self.model(x, extra_labels)
        y_pred_logits = outputs["logits"]  # shape: (B, l_out, C, N_t)

        # Get last token logits
        logits = y_pred_logits[:, -1:] / temperature
        # shape: (B, 1, C, N_t)

        # Sample the next token
        token_samples = sample_from_logits(logits, top_p, top_k, typical_p)  # shape: (B, 1, C)

        # Remove time dimension (since we only generated a single time step)
        token_samples = token_samples.squeeze(1)  # shape: (B, C)

        return token_samples

    @torch.no_grad()
    def generate_tokens(
        self,
        n_samples: int,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        typical_p: Optional[float] = None,
        temperature: float = 1.0,
        batch_size: Optional[int] = None,
        prompt: Optional[Union[np.ndarray, torch.Tensor]] = None,
        extra_labels: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Generates tokens using the model.

        Parameters
        ----------
        n_samples : int
            Number of new token samples per sequence to generate.
        top_p : float, optional
            Top p proportion of values to keep for nucleus sampling.
        top_k : int, optional
            Top k number of values to keep for top-k sampling.
        typical_p : float, optional
            Cumulative probability mass of the locally typical set to retain
            (Meister et al., 2023).
        temperature : float, optional
            Temperature for sampling from the logits. Higher values increase randomness.
        batch_size : int, optional
            Batch size for generating the samples (number of independent sequences).
            If None, the batch size in the configuration is used.
        prompt : torch.Tensor or np.ndarray, optional
            Prior context prompt to start the generation.
            If None, a random sequence is used as the prompt based on token frequencies.
            If provided, the shape must be (B, L, C) or (L, C).
        extra_labels : List[torch.Tensor or np.ndarray], optional
            List of static extra labels (e.g. subject IDs, tasks).
            Each label should be of shape (B, L + 1).

        Returns
        -------
        generated_tokens : torch.Tensor
            Generated tokens of shape (B, T, C).

        FAQ
        ---
        **What is the relationship between `n_samples` and `sequence_length`?**
        - `sequence_length` sets the size of the sliding context window the model "sees" at each step.
        - `n_samples` specifies how many *new* time points to generate.
        - The model generates exactly one token per channel at a time, sliding the context window forward for `n_samples` iterations.

        **What is the relationship between `batch_size` and `n_samples`?**
        - `batch_size` determines the number of independent sequences (e.g., simulated subjects or trials) generated in parallel.
        - The final output shape will be `(batch_size, n_samples, n_channels)`.
        - The context prompt (length = `sequence_length`) used to warm up the generation is stripped from the returned output.
        """
        # Set hyperparameters
        batch_size = batch_size or self.model.config.training.batch_size
        n_channels = self.model.config.n_channels
        sequence_length = self.model.config.sequence_length

        # Helper function to generate random tokens based on token frequencies
        def _random_tokens() -> torch.Tensor:
            try:
                token_weights = self.tokenizer.vocab["total_token_counts"].astype(np.float32)
            except AttributeError:
                token_weights = np.ones(
                    max(1, self.model.config.input_embedding.n_tokens - 1),
                    dtype=np.float32,
                )  # use uniform weights if no token counts are available
            # NOTE: We intentionally do not sample token ID 0 (outlier token). However, it assumes
            #       that the token ID 0 is present in the input data.

            token_weights /= np.sum(token_weights)
            weights_tensor = torch.tensor(token_weights, device=self.device)

            # Perform multinomial sampling
            flat_size = batch_size * sequence_length * n_channels
            samples = torch.multinomial(
                weights_tensor, num_samples=flat_size, replacement=True
            ) + 1
            tokens = samples.view(batch_size, sequence_length, n_channels)
            return tokens.to(torch.int64)

        # Handle prompt
        if prompt is None:
            prompt = _random_tokens()
        elif isinstance(prompt, np.ndarray):
            prompt = torch.from_numpy(prompt).to(self.device).long()
        elif isinstance(prompt, torch.Tensor):
            prompt = prompt.to(self.device).long()
        else:
            raise ValueError("Prompt must be a numpy array or torch Tensor.")

        if prompt.shape != (batch_size, sequence_length, n_channels):
            if prompt.dim() == 2 and prompt.shape == (sequence_length, n_channels):
                prompt = prompt.unsqueeze(0).expand(batch_size, -1, -1)  # broadcast to batch dimension
            else:
                raise ValueError(
                    "Prompt must have shape (batch_size, sequence_length, n_channels) or (sequence_length, n_channels)."
                )

        # Handle extra labels
        # NOTE: These are static labels that are not rolled out with the generated tokens.
        #       They remain fixed during generation and are passed to the model at each time step.
        formatted_extra_labels = []
        if extra_labels is not None:
            for lbl in extra_labels:
                if isinstance(lbl, np.ndarray):
                    lbl = torch.from_numpy(lbl).to(self.device)
                elif isinstance(lbl, torch.Tensor):
                    lbl = lbl.to(self.device)
                else:
                    raise ValueError("Extra labels must be numpy arrays or torch Tensors.")

                if lbl.shape != (batch_size, sequence_length + 1):
                    raise ValueError("Each extra label must have shape (batch_size, sequence_length + 1).")

                formatted_extra_labels.append(lbl)

        # Pre-allocate tensors fully on target device for speed
        # Prevents CPU/GPU sync overhead during generation
        generated_tokens = torch.zeros(
            (batch_size, sequence_length + n_samples, n_channels),
            dtype=torch.int64,
            device=self.device,
        )
        generated_tokens[:, :sequence_length] = prompt

        # Token generation
        for i in trange(
            sequence_length, sequence_length + n_samples, desc="Generating tokens"
        ):
            x_input = generated_tokens[:, i - sequence_length : i + 1]

            next_token = self.one_step_sample(
                x=x_input,
                extra_labels=formatted_extra_labels,
                top_p=top_p,
                top_k=top_k,
                typical_p=typical_p,
                temperature=temperature,
            )

            # Assign generated token into pre-allocated tensors
            generated_tokens[:, i] = next_token  # shape: (B, C)

        return generated_tokens[:, sequence_length:]

    def generate_data(self, **kwargs):
        """
        Generates data from the MEG-GPT model.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the `generate_tokens` method.

        Returns
        -------
        reconstructed_data : np.ndarray
            The reconstructed data corresponding to the generated tokens.
            Shape is (B, T, C).
        """
        outputs = self.generate_tokens(**kwargs)
        reconstructed_data = self.tokenizer.reconstruct_data(
            list(outputs.cpu().numpy())
        )
        return reconstructed_data
