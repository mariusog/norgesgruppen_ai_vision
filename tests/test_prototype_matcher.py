"""Tests for src/prototype_matcher.py — load_prototypes and match_prototypes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.prototype_matcher import load_prototypes, match_prototypes


class TestLoadPrototypes:
    def test_load_prototypes_returns_expected_keys(self, tmp_path: Path) -> None:
        """load_prototypes should return a dict with embeddings, class_ids, model_name."""
        data = {
            "embeddings": torch.randn(10, 128),
            "class_ids": list(range(10)),
            "model_name": "efficientnet_b3",
        }
        pt_file = tmp_path / "prototypes.pt"
        torch.save(data, str(pt_file))

        result = load_prototypes(pt_file)

        assert "embeddings" in result
        assert "class_ids" in result
        assert "model_name" in result
        assert result["model_name"] == "efficientnet_b3"

    def test_load_prototypes_embeddings_shape(self, tmp_path: Path) -> None:
        """Loaded embeddings should preserve shape."""
        embeddings = torch.randn(20, 256)
        data = {
            "embeddings": embeddings,
            "class_ids": list(range(20)),
            "model_name": "test_model",
        }
        pt_file = tmp_path / "prototypes.pt"
        torch.save(data, str(pt_file))

        result = load_prototypes(pt_file)
        assert result["embeddings"].shape == (20, 256)

    def test_load_prototypes_on_cpu(self, tmp_path: Path) -> None:
        """Loaded embeddings should be on CPU (map_location='cpu')."""
        data = {
            "embeddings": torch.randn(5, 64),
            "class_ids": [0, 1, 2, 3, 4],
            "model_name": "test",
        }
        pt_file = tmp_path / "prototypes.pt"
        torch.save(data, str(pt_file))

        result = load_prototypes(pt_file)
        assert result["embeddings"].device.type == "cpu"


class TestMatchPrototypes:
    def test_cosine_similarity_identical_vectors(self) -> None:
        """Cosine similarity between identical vectors should be ~1.0."""
        embeddings = torch.nn.functional.normalize(torch.randn(5, 64), dim=1)
        prototypes = {
            "embeddings": embeddings,
            "class_ids": [10, 20, 30, 40, 50],
        }
        # Query is identical to prototype 2 (class_id=30)
        query = embeddings[2:3].clone()

        sims, class_ids = match_prototypes(query, prototypes, top_k=1)

        assert sims.shape == (1, 1)
        assert class_ids.shape == (1, 1)
        assert sims[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert class_ids[0, 0].item() == 30

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Cosine similarity between orthogonal vectors should be ~0.0."""
        # Create two orthogonal unit vectors
        e1 = torch.zeros(1, 4)
        e1[0, 0] = 1.0
        e2 = torch.zeros(1, 4)
        e2[0, 1] = 1.0

        prototypes = {
            "embeddings": e2,
            "class_ids": [99],
        }

        sims, _ = match_prototypes(e1, prototypes, top_k=1)
        assert sims[0, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_top_k_returns_k_matches(self) -> None:
        """match_prototypes with top_k=3 should return 3 matches per query."""
        embeddings = torch.nn.functional.normalize(torch.randn(10, 32), dim=1)
        prototypes = {
            "embeddings": embeddings,
            "class_ids": list(range(10)),
        }
        query = torch.nn.functional.normalize(torch.randn(2, 32), dim=1)

        sims, class_ids = match_prototypes(query, prototypes, top_k=3)

        assert sims.shape == (2, 3)
        assert class_ids.shape == (2, 3)

    def test_top_k_similarities_are_sorted_descending(self) -> None:
        """Top-k similarities should be in descending order."""
        embeddings = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
        prototypes = {
            "embeddings": embeddings,
            "class_ids": list(range(8)),
        }
        query = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)

        sims, _ = match_prototypes(query, prototypes, top_k=5)

        for i in range(4):
            assert sims[0, i].item() >= sims[0, i + 1].item()

    def test_batch_query(self) -> None:
        """Multiple queries in a batch produce independent results."""
        embeddings = torch.nn.functional.normalize(torch.randn(5, 16), dim=1)
        prototypes = {
            "embeddings": embeddings,
            "class_ids": [0, 1, 2, 3, 4],
        }

        # Batch of 3 queries, each identical to a different prototype
        queries = torch.stack([embeddings[0], embeddings[2], embeddings[4]])

        sims, class_ids = match_prototypes(queries, prototypes, top_k=1)

        assert sims.shape == (3, 1)
        assert class_ids[0, 0].item() == 0
        assert class_ids[1, 0].item() == 2
        assert class_ids[2, 0].item() == 4
        # All top-1 similarities should be ~1.0
        for i in range(3):
            assert sims[i, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_class_ids_map_correctly(self) -> None:
        """Returned class_ids should correspond to the correct prototypes."""
        # Create clearly separable prototypes
        embeddings = torch.eye(4)  # 4 orthogonal unit vectors
        prototypes = {
            "embeddings": embeddings,
            "class_ids": [100, 200, 300, 400],
        }

        # Query closest to prototype index 3 (class_id 400)
        query = torch.zeros(1, 4)
        query[0, 3] = 1.0

        sims, class_ids = match_prototypes(query, prototypes, top_k=1)
        assert class_ids[0, 0].item() == 400
        assert sims[0, 0].item() == pytest.approx(1.0, abs=1e-5)
