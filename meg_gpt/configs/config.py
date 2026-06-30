"""Configuration class for MEG-GPT."""

# Import packages
import numpy as np
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional
from meg_gpt.typing import Label


@dataclass
class InputEmbeddingConfig:
    n_tokens: Optional[int] = None
    embedding_dim: Optional[int] = None
    token_embedding_dim: Optional[int] = None
    pos_embedding_dim: Optional[int] = None
    pos_embedding_type: str = "absolute"
    channel_embedding_dim: Optional[int] = None
    extra_label_specs: Optional[List[Label]] = None


@dataclass
class TransformerDecoderConfig:
    n_heads: int = 1
    model_dim: Optional[int] = None
    n_patches_out: Optional[List[int]] = None
    patch_len_out: Optional[List[int]] = None
    n_patches_in: Optional[List[int]] = None
    patch_len_in: Optional[List[int]] = None
    unpatched_len_in: Optional[List[int]] = None
    l_unpatched_b: Optional[List[int]] = None
    l_patched_b: Optional[List[int]] = None
    do_chan_attention: Optional[List[bool]] = None
    do_cross_attention: Optional[List[bool]] = None
    chan_attention_mask: Optional[List[Optional[str]]] = None

    full_channel_attention_dropout: Optional[float] = None
    channel_attention_channel_dropout: float = 0.0
    time_attention_channel_dropout: float = 0.0
    chan_attn_chandim: Optional[int] = None
    feed_forward_dim: Optional[int] = None
    feed_forward_activation: str = "relu"
    dropout: float = 0.0
    norm_type: str = "layer"
    n_groups: Optional[int] = None


@dataclass
class LossConfig:
    loss_sequence_length: Optional[int] = None
    top_k: Optional[List[int]] = None
    label_smoothing: float = 0.0


@dataclass
class TrainingConfig:
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "name": "adam",
        "learning_rate": 1e-3,
        "eps": 1e-7,
    })
    lr_scheduler: Optional[Dict[str, Any]] = None
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: str = "norm"
    accumulate_grad_batches: int = 1
    batch_size: int = 32
    n_epochs: int = 10
    val_split: float = 0.1
    multi_gpu: bool = False


@dataclass
class MEGGPTConfig:
    """
    MEG-GPT based on the decoder-only transformer architecture.
    """
    name: str = "meg_gpt"

    # Base defaults
    sequence_length: Optional[int] = None
    n_channels: Optional[int] = None

    # Input defaults
    input_embedding: InputEmbeddingConfig = field(
        default_factory=InputEmbeddingConfig
    )

    # Transformer decoder defaults
    transformer_decoder: TransformerDecoderConfig = field(
        default_factory=TransformerDecoderConfig
    )

    # Loss defaults
    loss: LossConfig = field(default_factory=LossConfig)

    # Training defaults
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def set_config(self, config: DictConfig) -> None:
        self.sequence_length = config.get("sequence_length", self.sequence_length)
        self.n_channels = config.get("n_channels", self.n_channels)

        self._set_input_config(config.get("input_embedding", self.input_embedding))
        self._set_loss_config(config.get("loss", self.loss))
        self._set_decoder_config(config.get("transformer_decoder", self.transformer_decoder))
        self._set_training_config(config.get("training", self.training))

    def _set_input_config(self, config: DictConfig) -> None:
        self.input_embedding = OmegaConf.merge(
            OmegaConf.structured(self.input_embedding), config
        )

    def _set_loss_config(self, config: DictConfig) -> None:
        self.loss = OmegaConf.merge(
            OmegaConf.structured(self.loss), config
        )

    def _set_training_config(self, config: DictConfig) -> None:
        self.training = OmegaConf.merge(
            OmegaConf.structured(self.training), config
        )

    def _set_decoder_config(self, config: DictConfig) -> None:
        if config.n_patches_out is None:
            config.n_patches_out = config.n_patches_in[1:] + [self.loss.loss_sequence_length]
            config.patch_len_out = config.patch_len_in[1:] + [1]
        
        for i in range(len(config.l_unpatched_b)):
            if config.l_unpatched_b[i] is None:
                config.l_unpatched_b[i] = config.n_patches_in[i] * config.patch_len_in[i]

            if config.l_patched_b[i] is None:
                config.l_patched_b[i] = config.n_patches_in[i]

            if config.chan_attention_mask[i] is not None:
                config.chan_attention_mask[i] = np.load(config.chan_attention_mask[i])

        if self.sequence_length is None:
            self.sequence_length = config.n_patches_in[0] * config.patch_len_in[0]

        self.transformer_decoder = OmegaConf.merge(
            OmegaConf.structured(self.transformer_decoder), config
        )

    def validate(self) -> None:
        self._validate_base_config()
        self._validate_input_config()
        self._validate_decoder_config()
        self._validate_loss_config()
        self._validate_training_config()

    def _validate_base_config(self) -> None:
        assert self.sequence_length is not None, "sequence_length must be set"
        assert self.n_channels is not None, "n_channels must be set"

    def _validate_input_config(self) -> None:
        cfg = self.input_embedding
        assert cfg.n_tokens is not None, "n_tokens must be set"
        assert cfg.n_tokens > 0, "n_tokens must be greater than 0"
        assert cfg.embedding_dim is not None, "embedding_dim must be set"
        VALID_POS_EMBEDDING_TYPES = ["absolute", "sinusoidal"]
        assert (
            cfg.pos_embedding_type in VALID_POS_EMBEDDING_TYPES
        ), f"pos_embedding_type must be one of {VALID_POS_EMBEDDING_TYPES}"

    def _validate_decoder_config(self) -> None:
        cfg = self.transformer_decoder
        assert cfg.n_heads > 0, "n_heads must be greater than 0"
        assert cfg.model_dim is not None, "model_dim must be set"
        assert (
            cfg.model_dim % cfg.n_heads == 0
        ), "model_dim must be divisible by n_heads"
        assert len(cfg.n_patches_in) > 0, "n_patches_in must be a non-empty list (where length is the number of layers)"
        assert len(cfg.patch_len_in) > 0, "patch_len_in must be a non-empty list (where length is the number of layers)"
        assert len(cfg.n_patches_out) > 0, "n_patches_out must be a non-empty list (where length is the number of layers)"
        assert len(cfg.patch_len_out) > 0, "patch_len_out must be a non-empty list (where length is the number of layers)"
        assert len(cfg.unpatched_len_in) > 0, "unpatched_len_in must be a non-empty list (where length is the number of layers)"
        assert len(cfg.l_unpatched_b) > 0, "l_unpatched_b must be a non-empty list (where length is the number of layers)"
        assert len(cfg.l_patched_b) > 0, "l_patched_b must be a non-empty list (where length is the number of layers)"

        assert cfg.feed_forward_dim is not None, "feed_forward_dim must be set"
        assert cfg.feed_forward_dim > 0, "feed_forward_dim must be greater than 0"
        assert 0.0 <= cfg.dropout < 1.0, "dropout must be in [0, 1)."
        assert 0.0 <= cfg.channel_attention_channel_dropout < 1.0, \
            "channel_attention_channel_dropout must be in [0, 1)."
        assert 0.0 <= cfg.time_attention_channel_dropout < 1.0, \
            "time_attention_channel_dropout must be in [0, 1)."
        ACTIVATION_TYPES = ["relu", "leaky_relu", "gelu", "swish", "silu"]
        assert cfg.feed_forward_activation in ACTIVATION_TYPES, f"activation_type must be one of {ACTIVATION_TYPES}"
        NORM_TYPES = ["layer", "batch", "group"]
        assert cfg.norm_type in NORM_TYPES, f"norm_type must be one of {NORM_TYPES}"
        assert cfg.n_groups is None or cfg.n_groups > 0, "n_groups must be None or greater than 0"
        if cfg.n_groups is not None:
            assert (
                cfg.model_dim % cfg.n_groups == 0
            ), "model_dim must be divisible by n_groups"

        fields = [
            cfg.n_patches_in,
            cfg.patch_len_in,
            cfg.n_patches_out,
            cfg.patch_len_out,
            cfg.unpatched_len_in,
            cfg.l_unpatched_b,
            cfg.l_patched_b,
            cfg.do_chan_attention,
            cfg.do_cross_attention,
            cfg.chan_attention_mask
        ]

        assert len({len(f) for f in fields}) == 1, (
            "All layer-by-layer configuration lists must have the same length."
        )

        for i in range(len(cfg.n_patches_in)):
            assert cfg.n_patches_in[i] > 0, "n_patches_in must be greater than 0"
            assert cfg.patch_len_in[i] > 0, "patch_len_in must be greater than 0"

            assert (
                    cfg.n_patches_out[i] * cfg.patch_len_out[i]
                    <= cfg.n_patches_in[i] * cfg.patch_len_in[i]
                ), f"Layer {i}: Output sequence length must be less than or equal to input sequence length"

            assert (
                cfg.unpatched_len_in[i] <= cfg.n_patches_in[i] * cfg.patch_len_in[i]
            ), f"Layer {i}: unpatched_len_in must be less than or equal to input sequence length"

            if i > 0:
                assert (
                    cfg.n_patches_out[i - 1] * cfg.patch_len_out[i - 1] == cfg.n_patches_in[i] * cfg.patch_len_in[i]
                ), f"""
                Output sequence length of layer {i - 1} does not match input sequence length of layer {i}
                Layer {i - 1}: {cfg.n_patches_in[i - 1] * cfg.patch_len_in[i - 1]} -> {cfg.n_patches_out[i - 1] * cfg.patch_len_out[i - 1]}
                Layer {i}: {cfg.n_patches_in[i] * cfg.patch_len_in[i]} -> {cfg.n_patches_out[i] * cfg.patch_len_out[i]}
                """

        assert (
            self.sequence_length == (cfg.n_patches_in[0] * cfg.patch_len_in[0])
        ), "sequence length must match input patch number and length."

    def _validate_loss_config(self) -> None:
        cfg = self.loss
        assert cfg.loss_sequence_length is not None, "loss_sequence_length must be set"
        assert (
            cfg.loss_sequence_length > 0
        ), "loss_sequence_length must be greater than 0"
        assert (
            0.0 <= cfg.label_smoothing < 1.0
        ), "label_smoothing must be in [0, 1)"

    def _validate_training_config(self) -> None:
        cfg = self.training
        assert cfg.optimizer is not None, "optimizer must be set"
        if cfg.gradient_clip_val is not None:
            assert cfg.gradient_clip_val > 0, "gradient_clip_val must be positive"
        assert cfg.gradient_clip_algorithm in ("norm", "value"), \
            "gradient_clip_algorithm must be 'norm' or 'value'"
        assert cfg.accumulate_grad_batches >= 1, \
            "accumulate_grad_batches must be >= 1"
        assert cfg.batch_size > 0, "batch_size must be greater than 0"
        assert cfg.n_epochs > 0, "n_epochs must be greater than 0"
        assert 0 < cfg.val_split < 1, "val_split must be between 0 and 1"
