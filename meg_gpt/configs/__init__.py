"""Wrapper class for MEG-GPT model configurations."""

# Import packages
from dataclasses import dataclass
from omegaconf import DictConfig
from meg_gpt.configs.config import MEGGPTConfig


@dataclass
class Config:
    """
    Base configuration class for building and training tokenizers.
    """
    config_class: MEGGPTConfig

    def set_config(self, config: DictConfig) -> None:
        self.config_class.set_config(config)

    def validate(self) -> None:
        self.config_class.validate()    


def get_config(config: DictConfig) -> Config:
    """
    Returns a Config object based on the provided configuration.

    Parameters
    ----------
    config : DictConfig
        Dictionary containing the config.

    Returns
    -------
    cfg: Config
        Config object containing the MEG-GPT configuration.
    """
    # Initialize config class
    cfg = Config(MEGGPTConfig())

    # Set model and training configurations
    cfg.set_config(config)

    # Validate configuration
    cfg.validate()

    return cfg
