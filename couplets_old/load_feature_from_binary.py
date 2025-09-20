
#%%
import json
import gzip
import struct
from pathlib import Path
from typing import Dict, Any, Union

from tqdm import tqdm

#%%
def load_feature_from_binary(
    path: Union[str, Path],
    layer_idx: int,
    feature_idx: int
) -> Dict[str, Any]:
    """
    Load feature data from binary format, equivalent to JavaScript loadFeatureFromBinary.
    
    Args:
        index_path: Path to the index.json.gz file and bins
        layer_idx: Layer index (no need to unpack from featureIndex)
        feature_idx: Feature index within the layer (no need to unpack from featureIndex)
        
    Returns:
        Dictionary containing the feature data
        
    Raises:
        FileNotFoundError: If index or bin file doesn't exist
        KeyError: If layer_idx not found in index
        IndexError: If feature_idx is out of bounds
        ValueError: If offsets or filename are missing
    """
    # Load and parse index.json.gz
    path = Path(path)
    index_path = path / 'index.json.gz'
    with gzip.open(index_path, 'rt') as f:
        index_data = json.load(f)
    
    # Get layer data
    if layer_idx >= len(index_data) or layer_idx < 0:
        raise KeyError(f"Layer {layer_idx} not found in index")
    
    layer_data = index_data[str(layer_idx)]
    
    # Extract offsets and filename
    if 'offsets' not in layer_data or 'filename' not in layer_data:
        raise ValueError(f"Missing offsets or filename for layer {layer_idx}")
    
    offsets = layer_data['offsets']
    bin_filename = layer_data['filename']
    
    # Validate feature index
    if feature_idx >= len(offsets) - 1 or feature_idx < 0:
        raise IndexError(f"Feature {feature_idx} out of bounds for layer {layer_idx}")
    
    # Get byte range
    start_byte = offsets[feature_idx]
    end_byte = offsets[feature_idx + 1]
    
    # Check if feature has any data
    if start_byte == end_byte:
        raise ValueError(f"Feature {feature_idx} in layer {layer_idx} has no data (empty feature)")
    
    # Load binary data from the specified byte range
    bin_file_path = path / bin_filename
    with open(bin_file_path, 'rb') as f:
        f.seek(start_byte)
        compressed_data = f.read(end_byte - start_byte)
    
    # Parse binary data (following the JavaScript bin parsing logic)
    # First 4 bytes contain the length of compressed data
    data_length = struct.unpack('<I', compressed_data[:4])[0]
    
    # Extract and decompress the JSON data (using gzip)
    compressed_json = compressed_data[4:4 + data_length]
    decompressed_json = gzip.decompress(compressed_json).decode('utf-8')
    feature_data = json.loads(decompressed_json)
    
    return feature_data
#%%#%%
def load_features_batch(path: Union[str, Path], layer_idx: int, feature_indices: list[int]) -> Dict[int, Dict[str, Any]]:
    """
    Load multiple features from the same layer in one batch operation.
    This is much faster than loading them individually because it:
    1. Loads the index only once
    2. Reads from the binary file in larger chunks
    3. Reduces file I/O operations
    
    Args:
        path: Path to the features directory
        layer_idx: Layer index
        feature_indices: List of feature indices to load
        
    Returns:
        Dictionary mapping feature_idx -> feature_data
    """
    if not feature_indices:
        return {}
    
    path = Path(path)
    index_path = path / 'index.json.gz'
    
    # Load index once
    with gzip.open(index_path, 'rt') as f:
        index_data = json.load(f)
    
    # Get layer data
    if layer_idx >= len(index_data) or layer_idx < 0:
        raise KeyError(f"Layer {layer_idx} not found in index")
    
    layer_data = index_data[str(layer_idx)]
    offsets = layer_data['offsets']
    bin_filename = layer_data['filename']
    
    # Validate all feature indices
    for feature_idx in feature_indices:
        if feature_idx >= len(offsets) - 1 or feature_idx < 0:
            raise IndexError(f"Feature {feature_idx} out of bounds for layer {layer_idx}")
    
    # Sort features by their byte positions for efficient reading
    sorted_features = sorted(feature_indices)
    
    # Read all data in one go if features are contiguous or close together
    min_start = offsets[sorted_features[0]]
    max_end = offsets[sorted_features[-1] + 1]
    
    bin_file_path = path / bin_filename
    with open(bin_file_path, 'rb') as f:
        f.seek(min_start)
        bulk_data = f.read(max_end - min_start)
    
    # Parse each feature from the bulk data
    results = {}
    for feature_idx in feature_indices:
        start_byte = offsets[feature_idx] - min_start
        end_byte = offsets[feature_idx + 1] - min_start
        
        # Skip features with no data (identical consecutive offsets)
        if start_byte == end_byte:
            print(f"Warning: Feature {feature_idx} in layer {layer_idx} has no data (empty feature), skipping")
            results[feature_idx] = None
            continue
            
        compressed_data = bulk_data[start_byte:end_byte]
        
        # Check if we have enough data
        if len(compressed_data) < 4:
            raise ValueError(f"Layer: {layer_idx}, Feature {feature_idx}: compressed_data only has {len(compressed_data)} bytes, need at least 4. "
                           f"start_byte={start_byte}, end_byte={end_byte}, "
                           f"offsets[{feature_idx}]={offsets[feature_idx]}, offsets[{feature_idx + 1}]={offsets[feature_idx + 1]}, "
                           f"min_start={min_start}, bulk_data_len={len(bulk_data)}")
        
        # Parse binary data (same as original function)
        data_length = struct.unpack('<I', compressed_data[:4])[0]
        compressed_json = compressed_data[4:4 + data_length]
        decompressed_json = gzip.decompress(compressed_json).decode('utf-8')
        feature_data = json.loads(decompressed_json)
        
        results[feature_idx] = feature_data
    
    return results


def get_features_top_acts_batch(model: str, layer: int, features: list[int]) -> Dict[int, Dict[str, Any]]:
    """
    Get top acts for multiple features in one batch - much faster than individual calls.
    
    Args:
        model: Model name
        layer: Layer index
        features: List of feature indices
        
    Returns:
        Dictionary mapping feature_idx -> processed feature data
    """
    # Load all features in batch
    features_data = load_features_batch(f'/root/model-planning/features/{model}/features', layer, features)
    
    # Process each feature
    results = {}
    for feature_idx, feature_data in features_data.items():
        if feature_data is None:
            results[feature_idx] = None
            continue
        
        top_quantile = feature_data['examples_quantiles'][0]
        assert top_quantile['quantile_name'] == 'Top'
        
        tokens = []
        acts = []
        top_indices = []
        top_tokens = []
        
        for example in top_quantile['examples']:
            tokens.append(example['tokens'])
            acts.append(example['tokens_acts_list'])
            i = max(range(len(example['tokens_acts_list'])), key=lambda i: example['tokens_acts_list'][i])
            top_indices.append(i)
            top_tokens.append(example['tokens'][i])
        
        try:
            results[feature_idx] = {
                'top_logits': feature_data['top_logits'],
                'bottom_logits': feature_data['bottom_logits'],
                'frequency': feature_data['activation_frequency'],
                'tokens': tokens,
                'acts': acts,
                'top_indices': top_indices,
                'top_tokens': top_tokens
            }
        except KeyError as e:
            print(feature_data.keys())
            raise e
    
    return results


def get_features_top_acts_from_list(model: str, layer_feature_pairs: list[tuple[int, int]], verbose=False) -> Dict[tuple[int, int], Dict[str, Any]]:
    """
    Convenience function to get top acts for a list of (layer, feature) pairs.
    Automatically groups by layer for optimal batch loading.
    
    Args:
        model: Model name
        layer_feature_pairs: List of (layer, feature) tuples
        
    Returns:
        Dictionary mapping (layer, feature) -> processed feature data
    """
    from collections import defaultdict
    
    # Group features by layer
    features_by_layer = defaultdict(list)
    for layer, feature in layer_feature_pairs:
        features_by_layer[layer].append(feature)
    
    # Load all features for each layer in batches
    all_results = {}
    for layer, features in tqdm(features_by_layer.items(), disable=not verbose):
        batch_results = get_features_top_acts_batch(model, layer, features)
        # Convert back to (layer, feature) keys
        for feature, result in batch_results.items():
            all_results[(layer, feature)] = result
    
    return all_results


def get_feature_top_acts(model, layer, feature):
    feature_data = load_feature_from_binary(f'/root/model-planning/features/{model}/features', layer, feature)
    top_quantile = feature_data['examples_quantiles'][0]
    assert top_quantile['quantile_name'] == 'Top'
    tokens = []
    acts = []
    top_indices = []
    top_tokens = []
    for example in feature_data['examples_quantiles'][0]['examples']:
        tokens.append(example['tokens'])
        acts.append(example['tokens_acts_list'])
        i = max(range(len(example['tokens_acts_list'])), key=lambda i: example['tokens_acts_list'][i])
        top_indices.append(i)
        top_tokens.append(example['tokens'][i])

    return {
        'top_logits': feature_data['top_logits'],
        'bottom_logits': feature_data['bottom_logits'],
        'frequency': feature_data['activation_frequency'],
        'tokens': tokens,
        'acts': acts,
        'top_indices': top_indices,
        'top_tokens': top_tokens
        }
