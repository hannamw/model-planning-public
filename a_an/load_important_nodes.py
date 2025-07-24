#%%
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from tqdm import tqdm

def load_model_results(model_name: str, results_dir: str = 'results/logit-lens'):
    """Load results and metadata for a specific model"""
    results_dir = Path(results_dir)
    model_dir = results_dir / model_name
    metadata_path = model_dir / 'metadata.csv'
    metadata = pd.read_csv(metadata_path)
    return metadata

def convert_model_name(model: str) -> str:
    """Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B"""
    model_name_parts = model.split('-')
    size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
    if size_part.endswith('B'):
        size_part = size_part[:-1] + 'B'
    return f"Qwen3-{size_part}"

def load_threshold_based_nodes(
    model: str,
    filename: str,
    count_threshold: int = 5,
    threshold_results_dir: str = 'results/relevant_nodes'
) -> Tuple[List[Tuple[int, int, int]], Dict, int]:
    """
    Load threshold-based (count-based) relevant features from file.
    
    Args:
        model: Model name (e.g. 'qwen3-0.6b-relu-lowl0')
        filename: The filename to load (e.g. 'a-banker.json')
        count_threshold: Minimum count for features to be considered relevant
        threshold_results_dir: Directory containing threshold-based results
        
    Returns:
        Tuple of (relevant_features_list, feature_relevance_data, total_features_analyzed)
    """
    logit_lens_model_name = convert_model_name(model)
    threshold_file = Path(threshold_results_dir) / logit_lens_model_name / filename
    
    with open(threshold_file, 'r') as f:
        data = json.load(f)
    
    # Filter features with count > 0
    feature_counts = data['feature_counts']
    relevant_features = []
    filtered_feature_counts = {}
    
    for feature_key, count in feature_counts.items():
        if count > count_threshold:
            # Parse feature key format: "layer_pos_feature_idx"
            parts = feature_key.split('_')
            layer, pos, feature_idx = int(parts[0]), int(parts[1]), int(parts[2])
            relevant_features.append((layer, pos, feature_idx))
            filtered_feature_counts[feature_key] = {'count': count}
    
    total_features_analyzed = data['total_features_analyzed']
    
    return relevant_features, filtered_feature_counts, total_features_analyzed

def load_similarity_based_nodes(
    model: str,
    filename: str,
    similarity_threshold: float = 0.63,
    similarity_results_dir: str = 'results/relevant_nodes_batched'
) -> Tuple[List[Tuple[int, int, int]], Dict, int]:
    """
    Load similarity-based relevant features from file, applying the specified threshold.
    
    Args:
        model: Model name (e.g. 'qwen3-0.6b-relu-lowl0')
        correct_article: The correct article ('a' or 'an')
        profession: The profession name
        similarity_threshold: Threshold for similarity-based results
        similarity_results_dir: Directory containing similarity-based results
        
    Returns:
        Tuple of (relevant_features_list, feature_relevance_data, total_features_analyzed)
    """
    logit_lens_model_name = convert_model_name(model)
    similarity_file = Path(similarity_results_dir) / logit_lens_model_name / filename
    
    with open(similarity_file, 'r') as f:
        data = json.load(f)
    
    # Filter features based on similarity threshold
    node_similarities = data['node_similarities']
    relevant_features = []
    filtered_similarities = {}
    
    for feature_key, similarity in node_similarities.items():
        if similarity >= similarity_threshold:
            # Parse feature key format: "layer,pos,feature_idx"
            parts = feature_key.split(',')
            layer, pos, feature_idx = int(parts[0]), int(parts[1]), int(parts[2])
            relevant_features.append((layer, pos, feature_idx))
            filtered_similarities[feature_key] = {'mean_similarity': similarity}
    
    # Get total features analyzed (may not be in similarity files, so set to None)
    total_features_analyzed = data.get('total_features_analyzed', None)
    
    return relevant_features, filtered_similarities, total_features_analyzed

def load_important_nodes(
    model: str,
    correct_article: str,
    profession: str,
    count_threshold: int = 5,
    similarity_threshold: float = 0.63,
    threshold_results_dir: str = 'results/relevant_nodes',
    similarity_results_dir: str = 'results/relevant_nodes_batched',
    position_filter: Optional[int] = None,
) -> Dict:
    """
    Load important nodes for a specific model and example by taking the union of 
    filtered count-based and similarity-based results.
    
    Args:
        model: Model name (e.g. 'qwen3-0.6b-relu-lowl0')
        correct_article: The correct article ('a' or 'an')
        profession: The profession name
        count_threshold: Minimum count for threshold-based results (features with count > count_threshold)
        similarity_threshold: Threshold for similarity-based results
        threshold_results_dir: Directory containing threshold-based results
        similarity_results_dir: Directory containing similarity-based results
        position_filter: If specified, only keep features at this position. Use -1 for last (highest) position.
        
    Returns:
        Dictionary with loaded results and metadata about the loading process
    """
    filename = f"{correct_article}-{profession}.json"
    
    # First, determine max position from ALL features (before filtering) if needed
    # TODO: do this more intelligently.
    position_to_filter = position_filter
    if position_filter == -1:
        # Load all features without thresholds to find max position
        all_threshold_features, _, _ = load_threshold_based_nodes(
            model, filename, count_threshold=0, threshold_results_dir=threshold_results_dir
        )
        all_similarity_features, _, _ = load_similarity_based_nodes(
            model, filename, similarity_threshold=0.0, similarity_results_dir=similarity_results_dir
        )
        all_unfiltered_features = set(all_threshold_features + all_similarity_features)
        
        if all_unfiltered_features:
            position_to_filter = max(pos for layer, pos, feature_idx in all_unfiltered_features)
        else:
            position_to_filter = None
    
    # Now load features with the specified thresholds
    threshold_features, threshold_relevance, threshold_total = load_threshold_based_nodes(
        model, filename, count_threshold, threshold_results_dir
    )

    similarity_features, similarity_relevance, similarity_total = load_similarity_based_nodes(
        model, filename, similarity_threshold, similarity_results_dir
    )

    # Take union of both feature sets
    all_features_set = set(threshold_features + similarity_features)
    final_features = list(all_features_set)
    
    # Apply position filtering if requested
    if position_to_filter is not None:
        # Filter features to only those at the specified position
        final_features = [(layer, pos, feature_idx) for layer, pos, feature_idx in final_features 
                        if pos == position_to_filter]
    
    # Combine relevance data
    final_relevance = {}
    final_relevance.update(threshold_relevance)
    
    # Add similarity data, converting feature tuples to the appropriate key format
    for feature_tuple in similarity_features:
        layer, pos, feature_idx = feature_tuple
        similarity_key = f"{layer},{pos},{feature_idx}"
        if similarity_key in similarity_relevance:
            # Convert to threshold format key for consistency
            threshold_key = f"{layer}_{pos}_{feature_idx}"
            if threshold_key in final_relevance:
                # Merge data if both exist
                final_relevance[threshold_key].update(similarity_relevance[similarity_key])
            else:
                # Add similarity data with threshold format key
                final_relevance[threshold_key] = similarity_relevance[similarity_key]
    
    # Filter relevance data to only include features that survived position filtering
    final_feature_keys = set(f"{layer}_{pos}_{feature_idx}" for layer, pos, feature_idx in final_features)
    final_relevance = {key: value for key, value in final_relevance.items() if key in final_feature_keys}
    
    # Use total from whichever source has it available
    final_total = threshold_total if threshold_total is not None else similarity_total
    
    # Add position filtering info to the result
    position_info = {}
    if position_filter is not None:
        if position_filter == -1 and final_features:
            position_info['position_filtered_to'] = max(pos for layer, pos, feature_idx in final_features)
            position_info['position_filter_method'] = 'last_position'
        elif position_filter != -1:
            position_info['position_filtered_to'] = position_filter
            position_info['position_filter_method'] = 'specified_position'
    
    return {
        'model': model,
        'logit_lens_model_name': convert_model_name(model),
        'correct_article': correct_article,
        'profession': profession,
        'relevant_features': final_features,
        'feature_relevance_data': final_relevance,
        'total_features_analyzed': final_total,
        'num_relevant_features': len(final_features),
        'threshold_available': len(threshold_features),
        'similarity_available': len(similarity_features),
        'similarity_threshold_used': similarity_threshold,
        'count_threshold_used': count_threshold,
        **position_info
    }

def load_important_nodes_for_model(
    model: str,
    count_threshold: int = 5,
    similarity_threshold: float = 0.63,
    threshold_results_dir: str = 'results/relevant_nodes',
    similarity_results_dir: str = 'results/relevant_nodes_batched',
    results_dir: str = 'results/logit-lens',
    position_filter: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Load important nodes for all examples in a model.
    
    Args:
        model: Model name (e.g. 'qwen3-0.6b-relu-lowl0')
        count_threshold: Minimum count for threshold-based results (features with count > count_threshold)
        similarity_threshold: Threshold for similarity-based results
        threshold_results_dir: Directory containing threshold-based results
        similarity_results_dir: Directory containing similarity-based results
        results_dir: Directory containing logit lens results (for metadata)
        position_filter: If specified, only keep features at this position. Use -1 for last (highest) position.
        
    Returns:
        Dictionary mapping "{correct_article}-{profession}" to loaded results
    """
    
    logit_lens_model_name = convert_model_name(model)
    metadata = load_model_results(logit_lens_model_name, results_dir)
    
    results = {}
    
    iterator = tqdm(metadata.iterrows(), desc=f"Loading nodes for {model}", total=len(metadata))
    
    for _, row in iterator:
        correct_article = row['correct_articles']
        profession = row['professions']
        
        result = load_important_nodes(
            model=model,
            correct_article=correct_article,
            profession=profession,
            count_threshold=count_threshold,
            similarity_threshold=similarity_threshold,
            threshold_results_dir=threshold_results_dir,
            similarity_results_dir=similarity_results_dir,
            position_filter=position_filter,
        )
        
        example_key = f"{correct_article}-{profession}"
        results[example_key] = result
    
    return results

def load_important_nodes_for_all_models(
    models: List[str],
    count_threshold: int = 5,
    similarity_threshold: float = 0.63,
    threshold_results_dir: str = 'results/relevant_nodes',
    similarity_results_dir: str = 'results/relevant_nodes_batched',
    results_dir: str = 'results/logit-lens',
    position_filter: Optional[int] = None,
) -> Dict[str, Dict[str, Dict]]:
    """
    Load important nodes for all examples across multiple models.
    
    Args:
        models: List of model names
        count_threshold: Minimum count for threshold-based results (features with count > count_threshold)
        similarity_threshold: Threshold for similarity-based results
        threshold_results_dir: Directory containing threshold-based results
        similarity_results_dir: Directory containing similarity-based results
        results_dir: Directory containing logit lens results (for metadata)
        position_filter: If specified, only keep features at this position. Use -1 for last (highest) position.
        
    Returns:
        Dictionary mapping model_name -> example_key -> loaded_results
    """
    
    all_results = {}
    
    for model in models:
        model_results = load_important_nodes_for_model(
            model=model,
            count_threshold=count_threshold,
            similarity_threshold=similarity_threshold,
            threshold_results_dir=threshold_results_dir,
            similarity_results_dir=similarity_results_dir,
            results_dir=results_dir,
            position_filter=position_filter,
        )
        
        all_results[model] = model_results
    
    return all_results


#%%
if __name__ == "__main__":
    # Example usage
    models = ['qwen3-0.6b-relu-lowl0', 'qwen3-1.7b-relu-lowl0', 'qwen3-4b-relu', 'qwen3-8b-relu', 'qwen3-14b-relu-lowl0']
    
    # Load with moderate similarity threshold and union approach, filtering to last position
    all_results = load_important_nodes_for_all_models(
        models=models,
        count_threshold=5,           # Include all count-based nodes with count > 5
        similarity_threshold=0.63,   # Higher threshold for similarity-based results
        position_filter=-1       # Only keep features at the highest position
    )
    
    # Example of accessing results for a specific model and example
    model = 'qwen3-14b-relu-lowl0'
    example_key = 'a-banker'  # or any other example
    
    if model in all_results and example_key in all_results[model]:
        result = all_results[model][example_key]
        print(f"\n=== Example: {model} - {example_key} ===")
        print(f"Number of relevant features: {result['num_relevant_features']}")
        print(f"Total features analyzed: {result['total_features_analyzed']}")
        print(f"Count-based features available: {result['threshold_available']}")
        print(f"Similarity-based features available: {result['similarity_available']}")
        print(f"Count threshold used: {result['count_threshold_used']}")
        print(f"Similarity threshold used: {result['similarity_threshold_used']}")

        
        # Show first few relevant features
        if result['relevant_features']:
            print(f"First few relevant features: {result['relevant_features'][:5]}")

# %%
