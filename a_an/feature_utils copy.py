#%%
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urljoin
from tqdm import tqdm
import concurrent.futures


def get_feature(transcoder_id: str, feature_idx: int, base_url: str = "https://d1fk9w8oratjix.cloudfront.net", 
                cache_dir: Optional[str] = '/root/model-planning/features') -> Dict[str, Any]:
    """
    Retrieve feature data from the circuit tracer frontend, with optional caching.
    
    Args:
        transcoder_id: The model/transcoder identifier (e.g., 'gemma-2-2b')
        feature_idx: The feature index to retrieve
        base_url: Base URL of the circuit tracer server (default: https://d1fk9w8oratjix.cloudfront.net)
        cache_dir: Optional directory to cache features locally 
        
    Returns:
        Dictionary containing the feature data
        
    Raises:
        requests.RequestException: If the HTTP request fails
        json.JSONDecodeError: If the response is not valid JSON
        FileNotFoundError: If the feature file doesn't exist (404 response)
    """
    # Check cache first if cache_dir is provided
    if cache_dir:
        cache_path = Path(cache_dir) / transcoder_id / f"{feature_idx}.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached_feature = json.load(f)
                cached_feature['featureIndex'] = feature_idx
                cached_feature['scan'] = transcoder_id
                return cached_feature
            except json.JSONDecodeError:
                print(f"Warning: Corrupted JSON in cache for feature {feature_idx} ({transcoder_id}). Deleting cache file.")
                cache_path.unlink()
    
    # Construct the feature URL based on the pattern from the JavaScript code
    feature_path = f"features/{transcoder_id}/{feature_idx}.json"
    feature_url = urljoin(base_url.rstrip('/') + '/', feature_path)
    
    try:
        response = requests.get(feature_url)
        response.raise_for_status()
        feature_data = response.json()
                            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("Feature 404'd:", feature_idx)
            # Return a "dead" feature object like the JavaScript code does
            feature_data = {
                'isDead': True,
                'statistics': {},
                'featureIndex': feature_idx,
                'scan': transcoder_id
            }
        else:
            raise e
    except requests.exceptions.RequestException as e:
        raise requests.RequestException(f"Failed to retrieve feature {feature_idx} for {transcoder_id}: {e}")

    # Save to cache if cache_dir is provided
    if cache_dir:
        cache_path = Path(cache_dir) / transcoder_id / f"{feature_idx}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(feature_data, f, indent=2)

    return feature_data


def get_feature_batch(features_to_get: list[tuple[str, int]], base_url: str = "https://d1fk9w8oratjix.cloudfront.net", 
                        cache_dir: Optional[str] = '/root/model-planning/features') -> Dict[tuple[str, int], Dict[str, Any]]:
    """
    Retrieve multiple features at once, with optional caching.
    
    Args:
        features_to_get: A list of (transcoder_id, feature_idx) tuples to retrieve
        base_url: Base URL of the circuit tracer server
        cache_dir: Optional directory to cache features locally
        
    Returns:
        Dictionary mapping feature indices to their data
    """
    results = {}
    for transcoder_id, feature_idx in tqdm(features_to_get):
        try:
            results[(transcoder_id, feature_idx)] = get_feature(transcoder_id, feature_idx, base_url, cache_dir)
        except Exception as e:
            print(f"Warning: Failed to retrieve feature {feature_idx}: {e}")
            results[(transcoder_id, feature_idx)] = {
                'isDead': True,
                'statistics': {},
                'featureIndex': feature_idx,
                'scan': transcoder_id,
                'error': str(e)
            }
    return results


def get_feature_batch_parallel(
    features_to_get: list[tuple[str, int]], 
    base_url: str = "https://d1fk9w8oratjix.cloudfront.net", 
    cache_dir: Optional[str] = '/root/model-planning/features',
    max_workers: int = 16
) -> Dict[tuple[str, int], Dict[str, Any]]:
    """
    Retrieve multiple features in parallel, with optional caching.
    
    Args:
        features_to_get: A list of (transcoder_id, feature_idx) tuples to retrieve
        base_url: Base URL of the circuit tracer server
        cache_dir: Optional directory to cache features locally
        max_workers: The maximum number of threads to use for parallel fetching.
        
    Returns:
        Dictionary mapping (transcoder_id, feature_idx) tuples to their data
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary to map futures to their input parameters
        future_to_feature = {
            executor.submit(get_feature, transcoder_id, feature_idx, base_url, cache_dir): (transcoder_id, feature_idx) 
            for transcoder_id, feature_idx in features_to_get
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_feature), total=len(features_to_get), desc="Fetching features"):
            transcoder_id, feature_idx = future_to_feature[future]
            try:
                feature_data = future.result()
                results[(transcoder_id, feature_idx)] = feature_data
            except Exception as e:
                print(f"Warning: Failed to retrieve feature {feature_idx} for {transcoder_id}: {e}")
                results[(transcoder_id, feature_idx)] = {
                    'isDead': True,
                    'statistics': {},
                    'featureIndex': feature_idx,
                    'scan': transcoder_id,
                    'error': str(e)
                }
    return results


# Example usage
if __name__ == "__main__":
    import time
    import random
    
    # Test configuration
    transcoder_id = 'Qwen3-14B/Qwen3-14b-relu-lowl0-0'
    batch_sizes = [1, 5, 10, 20, 50]  # Different batch sizes to test
    feature_range = range(10000, 11000)  # Pool of feature indices to sample from
    
    
    for batch_size in batch_sizes:
        # Generate random feature indices for this batch
        features_to_get = [(transcoder_id, idx) for idx in random.sample(list(feature_range), batch_size)]
        
        # Time the batch retrieval
        start_time = time.time()
        results = get_feature_batch(features_to_get)
        end_time = time.time()
        
        # Calculate stats
        elapsed = end_time - start_time
        per_feature = elapsed / batch_size
        
        print(f"\nBatch size: {batch_size}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Time per feature: {per_feature:.2f} seconds")
        print(f"Features/second: {batch_size/elapsed:.2f}")
        
        # Check success rate
        dead_features = sum(1 for r in results.values() if r.get('isDead', False))
        success_rate = (batch_size - dead_features) / batch_size * 100
        print(f"Success rate: {success_rate:.1f}%")
# %%
