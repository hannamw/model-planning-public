#%%
from pathlib import Path
from collections import defaultdict, Counter
import json
import multiprocessing as mp
from functools import partial
from typing import Optional, Callable

from tqdm import tqdm
import pandas as pd

from circuit_tracer.graph import Graph, prune_graph

from feature_utils import get_feature_batch_optimized

def count_terms_in_texts(texts: list[str], target_terms: list[str]) -> dict[str, int]:
    """
    Count occurrences of multiple terms in texts using substring matching.
    
    Args:
        texts: List of text strings to analyze
        target_terms: List of terms to count
        
    Returns:
        Dictionary mapping each term to its count
    """
    counts = {}
    for term in target_terms:
        counts[term] = sum(1 for text in texts if term.lower() in text.lower())
    return counts

def get_feature_texts_cached(features_to_get: list[tuple[int, int]], graph_scan: dict, cache_dir: str = 'cache/feature_texts') -> dict[tuple[int, int], list[str]]:
    """
    Get only the top quantile example texts for features, with lightweight caching.
    
    Args:
        features_to_get: List of (layer, feature_idx) tuples
        graph_scan: Dictionary mapping layer to transcoder_id
        cache_dir: Directory to cache the lightweight text data
        
    Returns:
        Dictionary mapping (layer, feature_idx) to list of example texts
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cached_features = {}
    features_to_fetch = []
    
    # Check what we already have cached
    for layer, feature_idx in features_to_get:
        transcoder_id = graph_scan[layer]
        feature_cache_file = cache_path / transcoder_id.replace('/', '_') / f"{feature_idx}.json"
        if feature_cache_file.exists():
            try:
                with open(feature_cache_file, 'r') as f:
                    cached_features[(layer, feature_idx)] = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                features_to_fetch.append((transcoder_id, feature_idx))
        else:
            features_to_fetch.append((transcoder_id, feature_idx))
    
    # Fetch missing features
    if features_to_fetch:
        print(f"Fetching {len(features_to_fetch)} features not in cache...")
        feature_jsons = get_feature_batch_optimized(features_to_fetch, cache_dir=None)
        
        for (transcoder_id, feature_idx), data in feature_jsons.items():
            layer = int(transcoder_id.split('-')[-1])
            try:
                # Extract only the top quantile example texts
                texts = [''.join(ex['tokens']) for ex in data['examples_quantiles'][0]['examples']]
                cached_features[(layer, feature_idx)] = texts
                
                # Save to lightweight cache
                feature_cache_file = cache_path / transcoder_id.replace('/', '_') / f"{feature_idx}.json"
                feature_cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(feature_cache_file, 'w') as f:
                    json.dump(texts, f, indent=2)
                    
            except (KeyError, IndexError) as e:
                print(f"Got the following error while processing {(transcoder_id, feature_idx)}", e)
                # If examples_quantiles doesn't exist or is empty, cache empty list
                cached_features[(layer, feature_idx)] = []
                feature_cache_file = cache_path / transcoder_id.replace('/', '_') / f"{feature_idx}.json"
                feature_cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(feature_cache_file, 'w') as f:
                    json.dump([], f)
    
    return cached_features

def load_model_results(model_name: str, results_dir: str = 'results/logit-lens'):
    """Load results and metadata for a specific model"""
    results_dir = Path(results_dir)
    model_dir = results_dir / model_name
    metadata_path = model_dir / 'metadata.csv'
    metadata = pd.read_csv(metadata_path)
    return metadata

def process_model_relevant_nodes(
    model: str, 
    node_threshold: Optional[float] = 0.9, 
    edge_threshold: Optional[float] = 0.99, 
    results_dir: str = 'results/is-are-animals-repeat',
    save_dir: str = 'results/relevant_nodes_refined',
    overwrite: bool = False
):
    """
    Process a single model to count profession and related term mentions for all active features.
    
    Args:
        model: Model name (e.g. 'qwen3-0.6b-relu-lowl0')
        node_threshold: Threshold for pruning graph nodes (None to disable pruning)
        edge_threshold: Threshold for pruning graph edges (None to disable pruning)
        results_dir: Directory containing logit lens results
        save_dir: Directory to save relevant node results
        overwrite: Whether to overwrite existing output files (default: False)
    
    Returns:
        dict: Summary of processed examples
    """
    
    # Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B
    model_name_parts = model.split('-')
    size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
    if size_part.endswith('B'):
        size_part = size_part[:-1] + 'B'
    logit_lens_model_name = f"Qwen3-{size_part}"
    
    metadata = load_model_results(logit_lens_model_name, results_dir)
    
    graph_dir = Path('graphs_diff') / model
    save_model_dir = Path(save_dir) / logit_lens_model_name
    save_model_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'model': model,
        'logit_lens_model_name': logit_lens_model_name,
        'node_threshold': node_threshold,
        'edge_threshold': edge_threshold,
        'processed_examples': 0
    }
    
    # Process each example based on metadata
    for _, row in tqdm(metadata.iterrows(), desc=f"Processing {model}"):
        correct_article = row['correct_articles']
        profession = row['professions']
        
        # Check if output file already exists and skip if overwrite=False
        save_path = save_model_dir / f"{correct_article}-{profession}.json"
        if save_path.exists() and not overwrite:
            summary['processed_examples'] += 1
            continue
        
        # Generate filename based on metadata
        filename = f"{correct_article}-{profession}.pt"
        graph_file = graph_dir / filename
            
        graph = Graph.from_pt(str(graph_file))
        
        # Handle pruning based on threshold values
        if node_threshold is None and edge_threshold is None:
            # No pruning - use all selected features
            selected_features = graph.selected_features
        else:
            # Apply pruning with provided thresholds
            pruning_kwargs = {}
            if node_threshold is not None:
                pruning_kwargs['node_threshold'] = node_threshold
            if edge_threshold is not None:
                pruning_kwargs['edge_threshold'] = edge_threshold
            
            pruned_graph = prune_graph(graph, **pruning_kwargs)
            node_mask = pruned_graph.node_mask[:len(graph.selected_features)]
            selected_features = graph.selected_features[node_mask]

        # Get all active features in (layer, pos, feature_idx) format
        active_features_list = []
        no_pos_features_to_pos = defaultdict(set)
        for layer, pos, feature_idx in graph.active_features[selected_features].numpy():
            # Convert numpy int64 to regular Python int for JSON serialization
            layer, pos, feature_idx = int(layer), int(pos), int(feature_idx)
            active_features_list.append((layer, pos, feature_idx))
            no_pos_features_to_pos[(layer, feature_idx)].add((layer, pos, feature_idx))
        
        # Get feature texts for unique (layer, feature_idx) pairs
        feature_texts = get_feature_texts_cached(
            list(no_pos_features_to_pos.keys()),
            graph.scan
        )

        # Count profession and related term mentions for each active feature
        feature_counts = {}
        
        # For each active (layer, pos, feature_idx) tuple, get the counts
        for layer, pos, feature_idx in active_features_list:
            texts = feature_texts.get((layer, feature_idx), [])
            
            # Count related terms
            term_counts = count_terms_in_texts(texts, [])
            related_terms_count = sum(term_counts.values())
            
            feature_key = (layer, pos, feature_idx)
            feature_counts[feature_key] = {
                'numbers_count': related_terms_count,
                'total_count': related_terms_count,
                'term_breakdown': term_counts
            }
        
        # Sort features by total count (profession + related terms)
        sorted_features = sorted(
            feature_counts.items(),
            key=lambda x: x[1]['total_count'],
            reverse=True
        )
        
        # Save results for this example
        save_data = {
            'correct_article': correct_article,
            'profession': profession,
            'related_terms': related_terms,
            'node_threshold': node_threshold,
            'edge_threshold': edge_threshold,
            'feature_counts': {
                f"{layer}_{pos}_{feature_idx}": counts
                for (layer, pos, feature_idx), counts in sorted_features
            },
            'total_features_analyzed': len(active_features_list)
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        summary['processed_examples'] += 1
    
    return summary

def process_model_wrapper(model, **kwargs):
    """Wrapper function for multiprocessing"""
    return process_model_relevant_nodes(model, **kwargs)

def process_all_models_parallel(
    models,
    node_threshold: Optional[float] = 0.9,
    edge_threshold: Optional[float] = 0.99,
    results_dir: str = 'results/logit-lens',
    save_dir: str = 'results/relevant_nodes_refined',
    n_processes: int = None,
    overwrite: bool = False
):
    """
    Process multiple models in parallel using multiprocessing.
    
    Args:
        models: List of model names to process
        node_threshold: Threshold for pruning graph nodes (None to disable pruning)
        edge_threshold: Threshold for pruning graph edges (None to disable pruning)
        results_dir: Directory containing logit lens results
        save_dir: Directory to save relevant node results
        n_processes: Number of processes to use (defaults to CPU count)
        overwrite: Whether to overwrite existing output files (default: False)
    
    Returns:
        list: List of summaries for each processed model
    """
    if n_processes is None:
        n_processes = min(len(models), mp.cpu_count())
    
    print(f"Processing {len(models)} models using {n_processes} processes...")
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_model_wrapper,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        results_dir=results_dir,
        save_dir=save_dir,
        overwrite=overwrite
    )
    
    # Process models in parallel
    with mp.Pool(n_processes) as pool:
        summaries = pool.map(process_func, models)
    
    return summaries

if __name__ == "__main__":
    models = ['qwen3-0.6b-relu-lowl0', 'qwen3-1.7b-relu-lowl0', 'qwen3-4b-relu', 'qwen3-8b-relu', 'qwen3-14b-relu-lowl0']
    summaries = process_all_models_parallel(
        models=models,
        node_threshold=0.9,      # Use None to disable node pruning
        edge_threshold=0.99,     # Use None to disable edge pruning
        overwrite=True          # Set to True to overwrite existing files
    )

# %%
