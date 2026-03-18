"""Common dataclasses and type aliases for the project."""

# Import packages
from dataclasses import dataclass
from typing import Optional


@dataclass
class Label:
    """Class for extra input labels."""
    name: str
    n_classes: int
    label_dim: Optional[int] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if not self.name:
            raise ValueError("name must be a valid, non-empty string.")

        if self.n_classes is None or self.n_classes <= 0:
            raise ValueError(f"n_classes must be greater than 0. Got {self.n_classes}")

        if self.label_dim is not None and self.label_dim <= 0:
            raise ValueError(f"label_dim must be greater than 0. Got {self.label_dim}")
