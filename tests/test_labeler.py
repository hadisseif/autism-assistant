import numpy as np

from facts import CATEGORY_KEYS
from labeler import SemanticLabeler, apply_e5_prefix, resolve_model


def test_assign_labels_multi_label():
    dummy = type("Dummy", (), {"cat_keys": CATEGORY_KEYS})()
    scores = np.array([
        [0.1, 0.5, 0.4],
        [0.35, 0.2, 0.1],
    ], dtype=float)

    result = SemanticLabeler.assign_labels(
        dummy, scores, threshold=0.3, multi_label=True
    )

    assert result[0]["top_label"] == CATEGORY_KEYS[1]
    assert result[0]["best_label"] == CATEGORY_KEYS[1]
    assert result[0]["labels"] == [CATEGORY_KEYS[1], CATEGORY_KEYS[2]]
    assert result[0]["confident"] is True

    assert result[1]["top_label"] == CATEGORY_KEYS[0]
    assert result[1]["best_label"] == CATEGORY_KEYS[0]
    assert result[1]["labels"] == [CATEGORY_KEYS[0]]
    assert result[1]["confident"] is True


def test_multi_label_returns_none_when_all_scores_below_threshold():
    dummy = type("Dummy", (), {"cat_keys": CATEGORY_KEYS})()
    scores = np.array([
        [0.10, 0.22, 0.31, 0.18, 0.15, 0.29, 0.21]
    ], dtype=float)

    result = SemanticLabeler.assign_labels(
        dummy, scores, threshold=0.35, multi_label=True
    )

    assert result[0]["top_label"] == "NONE"
    assert result[0]["labels"] == ["NONE"]
    assert result[0]["best_label"] == CATEGORY_KEYS[2]
    assert result[0]["top_score"] == 0.31
    assert result[0]["confident"] is False


def test_assign_labels_none_when_not_confident_single_label():
    dummy = type("Dummy", (), {"cat_keys": CATEGORY_KEYS})()
    scores = np.array([[0.1, 0.2, 0.3]], dtype=float)

    result = SemanticLabeler.assign_labels(
        dummy, scores, threshold=0.5, multi_label=False
    )

    assert result[0]["top_label"] == "NONE"
    assert result[0]["labels"] == ["NONE"]
    assert result[0]["best_label"] == CATEGORY_KEYS[2]
    assert result[0]["top_score"] == 0.3
    assert result[0]["confident"] is False


def test_best_label_is_preserved_when_none_is_returned():
    dummy = type("Dummy", (), {"cat_keys": CATEGORY_KEYS})()
    scores = np.array([[0.34, 0.20, 0.10]], dtype=float)

    result = SemanticLabeler.assign_labels(
        dummy, scores, threshold=0.35, multi_label=True
    )

    assert result[0]["top_label"] == "NONE"
    assert result[0]["best_label"] == CATEGORY_KEYS[0]
    assert result[0]["top_score"] == 0.34


def test_resolve_model_keys():
    key, name, prefix = resolve_model(model_key="e5-large")
    assert key == "e5-large"
    assert "e5" in name.lower()
    assert prefix == "e5"

    key, name, prefix = resolve_model(model_key="bge-m3")
    assert key == "bge-m3"
    assert "bge-m3" in name.lower()
    assert prefix == "none"


def test_apply_e5_prefix():
    texts = apply_e5_prefix(["hello"], role="query")
    assert texts == ["query: hello"]

    texts = apply_e5_prefix(["passage: already"], role="passage")
    assert texts == ["passage: already"]
