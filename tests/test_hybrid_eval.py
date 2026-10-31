from unittest.mock import patch

from app.evaluation.hybrid_eval import evaluate


@patch("app.evaluation.hybrid_eval.search_enterprise_knowledge")
def test_evaluate_recall_and_mrr(mock_search):
    mock_search.return_value = [
        {"source": "irrelevant.pdf"},
        {"source": "target.pdf"},
    ]
    metrics = evaluate(
        [{"query": "test question", "relevant_sources": ["target.pdf"]}],
        k=2,
    )
    assert metrics["examples"] == 1
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr@2"] == 0.5
