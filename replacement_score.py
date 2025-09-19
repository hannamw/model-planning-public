from typing import List, NamedTuple, Optional, Union

import torch
from transformer_lens import HookedTransformerConfig
from circuit_tracer.graph import Graph, normalize_matrix, compute_influence

def compute_graph_scores(graph: Graph) -> tuple[float, float]:
    """Compute metrics for evaluating how well the graph captures the model's computation.

    This function calculates two complementary scores that measure how much of the model's
    computation flows through interpretable feature nodes versus reconstruction error nodes:

    1. Replacement Score: Measures the fraction of end-to-end influence from input tokens
       to output logits that flows through feature nodes rather than error nodes. This is
       a strict metric that rewards complete explanations where tokens influence logits
       entirely through features.

    2. Completeness Score: Measures the fraction of incoming edges to all nodes (weighted
       by each node's influence on the output) that originate from feature or token nodes
       rather than error nodes. This metric gives partial credit for nodes that are mostly
       explained by features, even if some error influence remains.

    Args:
        graph: The computation graph containing nodes for features, errors, tokens, and logits,
               along with their connections and influence weights.

    Returns:
        tuple[float, float]: A tuple containing:
            - replacement_score: Fraction of token-to-logit influence through features (0-1)
            - completeness_score: Weighted fraction of non-error inputs across all nodes (0-1)

    Note:
        Higher scores indicate better model interpretability, with 1.0 representing perfect
        reconstruction where all computation flows through interpretable features. Lower
        scores indicate more reliance on error nodes, suggesting incomplete feature coverage.
    """
    n_logits = len(graph.logit_tokens)
    n_tokens = len(graph.input_tokens)
    n_features = len(graph.selected_features)
    error_start = n_features
    error_end = error_start + n_tokens * graph.cfg.n_layers
    token_end = error_end + n_tokens


    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities

    normalized_matrix = normalize_matrix(graph.adjacency_matrix)
    node_influence = compute_influence(normalized_matrix, logit_weights)
    token_influence = node_influence[error_end:token_end].sum()
    error_influence = node_influence[error_start:error_end].sum()

    replacement_score = token_influence / (token_influence + error_influence)

    non_error_fractions = 1 - normalized_matrix[:, error_start:error_end].sum(dim=-1)
    output_influence = node_influence + logit_weights
    completeness_score = (non_error_fractions * output_influence).sum() / output_influence.sum()

    return replacement_score.item(), completeness_score.item()