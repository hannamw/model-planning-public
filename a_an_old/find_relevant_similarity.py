#%%
from pathlib import Path
from collections import defaultdict, Counter
import json
import multiprocessing as mp
from functools import partial
from typing import Optional, Callable

from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

from circuit_tracer.graph import Graph, prune_graph

from feature_utils import get_feature_batch_optimized
#%%

def sentence_transformer_relevance_function(
    texts: list[str], 
    target_word: str, 
    model_name: str = 'all-MiniLM-L6-v2', 
    similarity_threshold: float = 0.5,
    device: str = 'cuda'
) -> tuple[bool, dict]:
    """
    Sentence transformer-based relevance using semantic similarity.
    
    Args:
        texts: List of text strings to analyze
        target_word: The word/phrase to check relevance against
        model_name: Name of the sentence transformer model to use
        similarity_threshold: Minimum similarity score to consider relevant
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Tuple of (is_relevant, metadata_dict)
    """
    if not texts:
        return False, {'mean_similarity': 0.0}
    
    model = SentenceTransformer(model_name, device=device)
    
    # Encode target word and texts
    target_embedding = model.encode([target_word])
    text_embeddings = model.encode(texts)
    
    # Compute cosine similarities
    similarities = np.dot(text_embeddings, target_embedding.T).flatten()
    
    # Normalize similarities to [0, 1] range (cosine similarity is in [-1, 1])
    similarities = (similarities + 1) / 2
    
    mean_similarity = float(np.mean(similarities))
    return mean_similarity >= similarity_threshold, {'mean_similarity': mean_similarity}

def batched_sentence_transformer_relevance_function(
    feature_texts: dict[tuple[int, int], list[str]],
    target_word: str,
    graph_scan: dict,
    model_name: str = 'all-MiniLM-L6-v2',
    similarity_threshold: float = 0.6,
    device: str = 'cuda'
) -> dict[str, dict]:
    """
    Batched sentence transformer-based relevance using semantic similarity.
    Processes all features at once for efficiency.
    
    Args:
        feature_texts: Dictionary mapping (layer, feature_idx) to list of texts
        target_word: The word/phrase to check relevance against
        graph_scan: Dictionary mapping layer to transcoder_id
        model_name: Name of the sentence transformer model to use
        similarity_threshold: Minimum similarity score to consider relevant
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Dictionary mapping feature_key to metadata dict with relevance info
    """
    if not feature_texts:
        return {}
    
    model = SentenceTransformer(model_name, device=device)
    
    # Encode target word once
    target_embedding = model.encode([target_word])
    
    # Collect all texts and track which feature they belong to
    all_texts = []
    text_to_feature = []  # Maps text index to (layer, feature_idx)
    
    for (layer, feature_idx), texts in feature_texts.items():
        for text in texts:
            all_texts.append(text)
            text_to_feature.append((layer, feature_idx))
    
    if not all_texts:
        return {f"{graph_scan[layer]}-{feature_idx}": {'mean_similarity': 0.0, 'is_relevant': False} 
                for layer, feature_idx in feature_texts.keys()}
    
    # Batch encode all texts at once
    print(f"Batch encoding {len(all_texts)} texts...")
    all_text_embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=True)
    
    # Compute similarities for all texts at once
    all_similarities = np.dot(all_text_embeddings, target_embedding.T).flatten()
    # Normalize to [0, 1] range
    all_similarities = (all_similarities + 1) / 2
    
    # Group similarities by feature and compute mean for each feature
    feature_relevance_data = {}
    
    for (layer, feature_idx), texts in feature_texts.items():
        transcoder_id = graph_scan[layer]
        feature_key = f"{transcoder_id}-{feature_idx}"
        
        if not texts:
            feature_relevance_data[feature_key] = {
                'mean_similarity': 0.0,
                'is_relevant': False
            }
            continue
        
        # Find similarities for this feature's texts
        feature_similarities = []
        for i, (text_layer, text_feature_idx) in enumerate(text_to_feature):
            if text_layer == layer and text_feature_idx == feature_idx:
                feature_similarities.append(all_similarities[i])
        
        mean_similarity = float(np.mean(feature_similarities))
        is_relevant = mean_similarity >= similarity_threshold
        
        feature_relevance_data[feature_key] = {
            'mean_similarity': mean_similarity,
            'is_relevant': is_relevant
        }
    
    return feature_relevance_data

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
    profession_threshold: int = 5,
    results_dir: str = 'results/logit-lens',
    save_dir: str = 'results/relevant_nodes_batched',
    overwrite: bool = False,
    relevance_function: Optional[Callable[[list[str], str], tuple[bool, dict]]] = None,
    device: str = 'cuda'
):
    """
    Process a single model to compute similarity scores for all active nodes.
    Uses batched sentence transformer processing for efficiency.
    
    Args:
        model: Model name (e.g. 'qwen3-0.6b-relu-lowl0')
        node_threshold: Threshold for pruning graph nodes (None to disable pruning)
        edge_threshold: Threshold for pruning graph edges (None to disable pruning)
        profession_threshold: Minimum count for profession mentions to be considered relevant (used with default threshold function)
        results_dir: Directory containing logit lens results
        save_dir: Directory to save similarity results
        overwrite: Whether to overwrite existing output files (default: False)
        relevance_function: Function that takes (texts, target_word) and returns (is_relevant, metadata_dict)
        device: Device to use for sentence transformer ('cuda' or 'cpu')
    
    Returns:
        dict: Summary of processed examples and features analyzed
    """
    
    # Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B
    model_name_parts = model.split('-')
    size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
    if size_part.endswith('B'):
        size_part = size_part[:-1] + 'B'
    logit_lens_model_name = f"Qwen3-{size_part}"
    
    metadata = load_model_results(logit_lens_model_name, results_dir)
    
    graph_dir = Path('attribution_graphs') / model
    save_model_dir = Path(save_dir) / logit_lens_model_name
    save_model_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'model': model,
        'logit_lens_model_name': logit_lens_model_name,
        'node_threshold': node_threshold,
        'edge_threshold': edge_threshold,
        'profession_threshold': profession_threshold,
        'processed_examples': 0,
        'total_features_analyzed': 0
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
            node_mask = slice(None)  # This will select all features
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
        

        # Map features to their positions and collect unique (layer, feature_idx) pairs
        no_pos_features_to_pos = defaultdict(set)
        all_active_nodes = []
        
        for layer, pos, feature_idx in graph.active_features[selected_features].numpy():
            # Convert numpy int64 to regular Python int for JSON serialization
            layer, pos, feature_idx = int(layer), int(pos), int(feature_idx)
            no_pos_features_to_pos[(layer, feature_idx)].add((layer, pos, feature_idx))
            all_active_nodes.append((layer, pos, feature_idx))
        
        # Use the lightweight caching function instead of full JSON data
        feature_texts = get_feature_texts_cached(
            list(no_pos_features_to_pos.keys()),
            graph.scan
        )
        
        # Store similarity for every active node
        node_similarities = {}
        
        # Use batched sentence transformer processing for all similarity computation
        if relevance_function is None:
            # Use batched sentence transformer processing by default
            batch_relevance_data = batched_sentence_transformer_relevance_function(
                feature_texts, profession, graph.scan, device=device
            )
            
            # Map similarity scores back to individual (layer, pos, feature_idx) nodes
            for (layer, feature_idx), node_positions in no_pos_features_to_pos.items():
                transcoder_id = graph.scan[layer]
                feature_key = f"{transcoder_id}-{feature_idx}"
                
                if feature_key in batch_relevance_data:
                    similarity_score = batch_relevance_data[feature_key]['mean_similarity']
                    
                    # Assign the same similarity score to all positions of this feature
                    for node_tuple in node_positions:
                        node_similarities[node_tuple] = similarity_score
                else:
                    # Assign 0.0 similarity if no data available
                    for node_tuple in node_positions:
                        node_similarities[node_tuple] = 0.0
        else:
            # Use provided custom relevance function
            for (layer, feature_idx), texts in feature_texts.items():
                # Check similarity using the custom relevance function
                _, metadata = relevance_function(texts, profession)
                similarity_score = metadata['mean_similarity']
                
                # Assign similarity to all positions of this feature
                for node_tuple in no_pos_features_to_pos[(layer, feature_idx)]:
                    node_similarities[node_tuple] = similarity_score
        
        # Save results for this example
        save_data = {
            'correct_article': correct_article,
            'profession': profession,
            'node_threshold': node_threshold,
            'edge_threshold': edge_threshold,
            'node_similarities': {f"{layer},{pos},{feature_idx}": similarity 
                                 for (layer, pos, feature_idx), similarity in node_similarities.items()},
            'total_nodes_analyzed': len(all_active_nodes)
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        summary['processed_examples'] += 1
        summary['total_features_analyzed'] += len(all_active_nodes)
    
    return summary

def process_model_wrapper(model, **kwargs):
    """Wrapper function for multiprocessing"""
    return process_model_relevant_nodes(model, **kwargs)

def process_all_models_parallel(
    models,
    node_threshold: Optional[float] = 0.9,
    edge_threshold: Optional[float] = 0.99,
    profession_threshold: int = 5,
    results_dir: str = 'results/logit-lens',
    save_dir: str = 'results/relevant_nodes_batched',
    n_processes: int = None,
    overwrite: bool = False,
    relevance_function: Optional[Callable[[list[str], str], tuple[bool, dict]]] = None,
    device: str = 'cuda'
):
    """
    Process multiple models in parallel using multiprocessing.
    Computes similarity scores for all active nodes using batched sentence transformer processing.
    
    Args:
        models: List of model names to process
        node_threshold: Threshold for pruning graph nodes (None to disable pruning)
        edge_threshold: Threshold for pruning graph edges (None to disable pruning)
        profession_threshold: Minimum count for profession mentions to be considered relevant (used with default threshold function)
        results_dir: Directory containing logit lens results
        save_dir: Directory to save similarity results
        n_processes: Number of processes to use (defaults to CPU count)
        overwrite: Whether to overwrite existing output files (default: False)
        relevance_function: Function that takes (texts, target_word) and returns (is_relevant, metadata_dict)
        device: Device to use for sentence transformer ('cuda' or 'cpu')
    
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
        profession_threshold=profession_threshold,
        results_dir=results_dir,
        save_dir=save_dir,
        overwrite=overwrite,
        relevance_function=relevance_function,
        device=device
    )
    
    # Process models in parallel
    with mp.Pool(n_processes) as pool:
        summaries = pool.map(process_func, models)
    
    return summaries

#%%
if __name__ == "__main__":
    models = ['qwen3-0.6b-relu-lowl0', 'qwen3-1.7b-relu-lowl0', 'qwen3-4b-relu', 'qwen3-8b-relu', 'qwen3-14b-relu-lowl0']

    # Using batched sentence transformer-based relevance (default behavior)
    summaries_transformer = process_all_models_parallel(
        models=models,
        node_threshold=0.9,
        edge_threshold=0.99,
        overwrite=True,
        device='cuda'      # Use GPU acceleration
    )