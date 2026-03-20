"""Prototype matching for inference-time classification refinement.

Loads pre-computed prototype embeddings and matches crop features against them
using cosine similarity. Used as a fallback when the classifier has low confidence.

SUBMISSION CODE: No forbidden imports (os, subprocess, etc.). Uses pathlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load_prototypes(path: Path) -> dict[str, Any]:
    """Load pre-computed prototype embeddings from a .pt file.

    Args:
        path: Path to the prototypes.pt file.

    Returns:
        Dict with keys:
            - embeddings: Tensor [num_prototypes, feature_dim] on CPU
            - class_ids: list[int] of category IDs matching each row
            - model_name: str identifying the model used to compute embeddings
    """
    data: dict[str, Any] = torch.load(str(path), map_location="cpu", weights_only=True)  # nosec B614
    return data


def match_prototypes(
    features: torch.Tensor,
    prototypes: dict,
    top_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match feature vectors against prototype embeddings via cosine similarity.

    Args:
        features: Tensor [B, feature_dim], L2-normalized feature vectors.
        prototypes: Dict from load_prototypes() with 'embeddings' and 'class_ids'.
        top_k: Number of top matches to return per query.

    Returns:
        Tuple of:
            - similarities: Tensor [B, top_k] of cosine similarity scores
            - class_ids: Tensor [B, top_k] of matched category IDs (int)
    """
    embeddings = prototypes["embeddings"].to(features.device)  # [P, D]
    class_id_list = prototypes["class_ids"]  # list of ints, length P

    # L2-normalize features (should already be normalized, but ensure it)
    features = F.normalize(features, dim=1)

    # Cosine similarity: [B, D] @ [D, P] -> [B, P]
    similarity_matrix = features @ embeddings.t()

    # Get top-k matches
    top_similarities, top_indices = similarity_matrix.topk(top_k, dim=1)  # [B, top_k]

    # Map indices to class IDs
    class_ids_tensor = torch.tensor(class_id_list, device=features.device)
    matched_class_ids = class_ids_tensor[top_indices]  # [B, top_k]

    return top_similarities, matched_class_ids
