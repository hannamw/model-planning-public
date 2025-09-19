import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys

def _run_async(coro):
    """
    Helper function to run async code in both sync and async contexts.
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're already in an event loop, create a task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(coro)

class FastFeatureRetriever:
    def __init__(self, base_url: str = "https://d1fk9w8oratjix.cloudfront.net/features/"):
        self.base_url = base_url
        self.cache: Dict[str, Any] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_initialized = False
    
    # Synchronous wrapper methods for non-async usage
    def get_feature_sync(self, transcoder_id: str, feature_idx: int, use_cache: bool = True) -> Dict[str, Any]:
        """
        Synchronous wrapper for get_feature. Use this in non-async contexts.
        
        Args:
            transcoder_id: Transcoder ID (e.g., 'gemma-2-2b')
            feature_idx: Feature index number
            use_cache: Whether to use in-memory cache
        """
        async def _async_wrapper():
            async with FastFeatureRetriever(self.base_url) as retriever:
                return await retriever.get_feature(transcoder_id, feature_idx, use_cache)
        
        return _run_async(_async_wrapper())
    
    def get_features_batch_sync(self, feature_specs: List[Tuple[str, int]], use_cache: bool = True, max_retries: int = 2) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """
        Synchronous wrapper for get_features_batch. Use this in non-async contexts.
        
        Args:
            feature_specs: List of (transcoder_id, feature_idx) tuples
            use_cache: Whether to use in-memory cache
            max_retries: Number of retry attempts for failed requests
        """
        async def _async_wrapper():
            async with FastFeatureRetriever(self.base_url) as retriever:
                return await retriever.get_features_batch(feature_specs, use_cache, max_retries)
        
        return _run_async(_async_wrapper())
    
    def get_feature_range_sync(self, transcoder_id: str, start_idx: int, end_idx: int, use_cache: bool = True) -> Dict[int, Dict[str, Any]]:
        """
        Synchronous wrapper for get_feature_range. Use this in non-async contexts.
        
        Args:
            transcoder_id: Transcoder ID
            start_idx: Starting feature index (inclusive)
            end_idx: Ending feature index (inclusive)
            use_cache: Whether to use in-memory cache
        """
        async def _async_wrapper():
            async with FastFeatureRetriever(self.base_url) as retriever:
                return await retriever.get_feature_range(transcoder_id, start_idx, end_idx, use_cache)
        
        return _run_async(_async_wrapper())
    
    def debug_url_sync(self, transcoder_id: str, feature_idx: int) -> Dict[str, Any]:
        """
        Synchronous wrapper for debug_url. Use this in non-async contexts.
        """
        async def _async_wrapper():
            async with FastFeatureRetriever(self.base_url) as retriever:
                return await retriever.debug_url(transcoder_id, feature_idx)
        
        return _run_async(_async_wrapper())
        
    async def __aenter__(self):
        # Configure session for maximum performance
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection limit
            limit_per_host=50,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; FeatureRetriever/1.0)',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Cache-Control': 'max-age=3600'  # Leverage CDN caching
            }
        )
        self._session_initialized = True
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
            self._session_initialized = False
    
    async def get_feature(self, transcoder_id: str, feature_idx: int, use_cache: bool = True) -> Dict[str, Any]:
        """
        Retrieve a single feature JSON file.
        
        Args:
            transcoder_id: Transcoder ID (e.g., 'gemma-2-2b')
            feature_idx: Feature index number
            use_cache: Whether to use in-memory cache
        """
        # Ensure session is initialized
        if not self._session_initialized or self.session is None:
            raise RuntimeError("Session not initialized. Use 'async with FastFeatureRetriever() as retriever:' pattern")
        
        url = f"{self.base_url}{transcoder_id}/{feature_idx}.json"
        
        if use_cache and url in self.cache:
            return self.cache[url]
        
        try:
            async with self.session.get(url) as response:
                # Log response details for debugging
                if response.status != 200:
                    response_text = await response.text()
                    print(f"HTTP {response.status} for {url}: {response_text[:200]}...")
                    return None
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if 'application/json' not in content_type and 'text/plain' not in content_type:
                    print(f"Unexpected content-type '{content_type}' for {url}")
                
                # Get response text first to handle potential JSON decode errors
                response_text = await response.text()
                
                # Check if response is empty
                if not response_text.strip():
                    print(f"Empty response from {url}")
                    return None
                
                # Try to parse JSON
                try:
                    data = json.loads(response_text)
                    if data is None:
                        print(f"JSON parsed to None for {url}")
                        return None
                    
                    if use_cache:
                        self.cache[url] = data
                    return data
                    
                except json.JSONDecodeError as json_err:
                    print(f"JSON decode error for {url}: {json_err}")
                    print(f"Response preview: {response_text[:200]}...")
                    return None
                    
        except aiohttp.ClientError as client_err:
            print(f"Client error fetching {url}: {client_err}")
            return None
        except asyncio.TimeoutError:
            print(f"Timeout fetching {url}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching {url}: {type(e).__name__}: {e}")
            return None
    
    async def get_features_batch(self, feature_specs: List[Tuple[str, int]], use_cache: bool = True, max_retries: int = 2) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """
        Retrieve multiple features concurrently using (transcoder_id, feature_idx) tuples.
        
        Args:
            feature_specs: List of (transcoder_id, feature_idx) tuples
            use_cache: Whether to use in-memory cache
            max_retries: Number of retry attempts for failed requests
        """
        async def fetch_with_retry(transcoder_id: str, feature_idx: int) -> Tuple[Tuple[str, int], Dict[str, Any]]:
            feature_spec = (transcoder_id, feature_idx)
            
            for attempt in range(max_retries + 1):
                result = await self.get_feature(transcoder_id, feature_idx, use_cache)
                if result is not None:
                    return (feature_spec, result)
                
                if attempt < max_retries:
                    wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                    print(f"Retrying {feature_spec} in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(wait_time)
            
            print(f"Failed to retrieve {feature_spec} after {max_retries + 1} attempts")
            return (feature_spec, None)
        
        # Create tasks for all feature specs
        tasks = [
            asyncio.create_task(fetch_with_retry(transcoder_id, feature_idx))
            for transcoder_id, feature_idx in feature_specs
        ]
        
        # Wait for all tasks to complete
        results = {}
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                print(f"Task failed with exception: {task_result}")
                continue
            
            feature_spec, data = task_result
            if data is not None:
                results[feature_spec] = data
        
        return results
    
    async def get_feature_range(self, transcoder_id: str, start_idx: int, end_idx: int, use_cache: bool = True) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve a range of features efficiently for a single transcoder.
        
        Args:
            transcoder_id: Transcoder ID
            start_idx: Starting feature index (inclusive)
            end_idx: Ending feature index (inclusive)
            use_cache: Whether to use in-memory cache
        """
        feature_specs = [(transcoder_id, idx) for idx in range(start_idx, end_idx + 1)]
        batch_results = await self.get_features_batch(feature_specs, use_cache)
        
        # Convert back to just feature_idx -> data mapping for convenience
        return {feature_idx: data for (_, feature_idx), data in batch_results.items()}
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache.clear()
    
    async def debug_url(self, transcoder_id: str, feature_idx: int) -> Dict[str, Any]:
        """
        Debug a specific URL to understand what's happening.
        
        Args:
            transcoder_id: Transcoder ID
            feature_idx: Feature index
        """
        # Ensure session is initialized
        if self.session is None:
            raise RuntimeError("Session not initialized. Use 'async with FastFeatureRetriever() as retriever:' pattern")
        
        url = f"{self.base_url}{transcoder_id}/{feature_idx}.json"
        print(f"Debugging URL: {url}")
        
        try:
            async with self.session.get(url) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                print(f"Content-Type: {response.headers.get('content-type', 'Not set')}")
                print(f"Content-Length: {response.headers.get('content-length', 'Not set')}")
                
                response_text = await response.text()
                print(f"Response length: {len(response_text)} characters")
                print(f"Response preview (first 500 chars): {response_text[:500]}")
                
                if response.status == 200:
                    try:
                        data = json.loads(response_text)
                        print(f"JSON parsed successfully. Type: {type(data)}")
                        if isinstance(data, dict):
                            print(f"JSON keys: {list(data.keys())[:10]}...")  # First 10 keys
                        return data
                    except json.JSONDecodeError as e:
                        print(f"JSON decode failed: {e}")
                        return None
                else:
                    print(f"Non-200 status code: {response.status}")
                    return None
                    
        except Exception as e:
            print(f"Request failed: {type(e).__name__}: {e}")
            return None


# Convenience functions for non-async usage
def get_feature(transcoder_id: str, feature_idx: int, use_cache: bool = True, base_url: str = "https://d1fk9w8oratjix.cloudfront.net/features/") -> Dict[str, Any]:
    """
    Simple synchronous function to get a single feature.
    Works in both regular Python and Jupyter notebooks.
    
    Usage:
        feature = get_feature("gemma-2-2b", 42)
    """
    async def _async_wrapper():
        async with FastFeatureRetriever(base_url) as retriever:
            return await retriever.get_feature(transcoder_id, feature_idx, use_cache)
    
    return _run_async(_async_wrapper())

def get_features_batch(feature_specs: List[Tuple[str, int]], use_cache: bool = True, max_retries: int = 2, base_url: str = "https://d1fk9w8oratjix.cloudfront.net/features/") -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Simple synchronous function to get multiple features.
    Works in both regular Python and Jupyter notebooks.
    
    Usage:
        specs = [("gemma-2-2b", 1), ("gemma-2-2b", 5), ("other-model", 10)]
        features = get_features_batch(specs)
    """
    async def _async_wrapper():
        async with FastFeatureRetriever(base_url) as retriever:
            return await retriever.get_features_batch(feature_specs, use_cache, max_retries)
    
    return _run_async(_async_wrapper())

def get_feature_range(transcoder_id: str, start_idx: int, end_idx: int, use_cache: bool = True, base_url: str = "https://d1fk9w8oratjix.cloudfront.net/features/") -> Dict[int, Dict[str, Any]]:
    """
    Simple synchronous function to get a range of features.
    Works in both regular Python and Jupyter notebooks.
    
    Usage:
        features = get_feature_range("gemma-2-2b", 0, 99)
    """
    async def _async_wrapper():
        async with FastFeatureRetriever(base_url) as retriever:
            return await retriever.get_feature_range(transcoder_id, start_idx, end_idx, use_cache)
    
    return _run_async(_async_wrapper())


# Example usage functions
async def example_single_feature():
    """Example: Retrieve a single feature"""
    async with FastFeatureRetriever() as retriever:
        start_time = time.time()
        feature = await retriever.get_feature("gemma-2-2b", 0)
        end_time = time.time()
        
        print(f"Retrieved feature 0 in {end_time - start_time:.3f} seconds")
        if feature:
            print(f"Feature keys: {list(feature.keys())}")
        return feature

async def example_batch_features():
    """Example: Retrieve multiple features from different transcoders"""
    async with FastFeatureRetriever() as retriever:
        # Mix of different transcoder_ids and feature indices
        feature_specs = [
            ("gemma-2-2b", 0),
            ("gemma-2-2b", 5),
            ("gemma-2-2b", 10),
            ("some-other-model", 0),  # This will fail gracefully if model doesn't exist
            ("gemma-2-2b", 15),
            ("gemma-2-2b", 20)
        ]
        
        start_time = time.time()
        features = await retriever.get_features_batch(feature_specs)
        end_time = time.time()
        
        print(f"Retrieved {len(features)} features in {end_time - start_time:.3f} seconds")
        print(f"Feature specs retrieved: {list(features.keys())}")
        print(f"Average time per feature: {(end_time - start_time) / len(features):.3f} seconds")
        return features

async def example_range_features():
    """Example: Retrieve a range of features from single transcoder"""
    async with FastFeatureRetriever() as retriever:
        start_time = time.time()
        features = await retriever.get_feature_range("gemma-2-2b", 0, 19)
        end_time = time.time()
        
        print(f"Retrieved features 0-19 in {end_time - start_time:.3f} seconds")
        print(f"Cache stats: {retriever.get_cache_stats()}")
        return features

async def benchmark_performance():
    """Benchmark the retrieval performance"""
    async with FastFeatureRetriever() as retriever:
        # Warm up
        await retriever.get_feature("gemma-2-2b", 0)
        
        # Benchmark single requests
        single_times = []
        for i in range(5):
            start = time.time()
            await retriever.get_feature("gemma-2-2b", i, use_cache=False)
            single_times.append(time.time() - start)
        
        # Benchmark batch requests with mixed specs
        feature_specs = [("gemma-2-2b", i) for i in range(20, 40)]
        batch_start = time.time()
        await retriever.get_features_batch(feature_specs, use_cache=False)
        batch_time = time.time() - batch_start
        
        print(f"\nPerformance Benchmark:")
        print(f"Single request average: {sum(single_times) / len(single_times):.3f}s")
        print(f"Batch request (20 features): {batch_time:.3f}s")
        print(f"Batch average per feature: {batch_time / 20:.3f}s")
        print(f"Speedup factor: {(sum(single_times) / len(single_times)) / (batch_time / 20):.1f}x")

async def save_features_to_disk():
    """Example: Save features to local files"""
    async with FastFeatureRetriever() as retriever:
        # Get features from multiple transcoders
        feature_specs = [("gemma-2-2b", i) for i in range(0, 10)]
        features = await retriever.get_features_batch(feature_specs)
        
        # Create output directory
        output_dir = Path("features_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save each feature
        for (transcoder_id, feature_idx), feature_data in features.items():
            output_file = output_dir / f"{transcoder_id}_feature_{feature_idx}.json"
            with open(output_file, 'w') as f:
                json.dump(feature_data, f, indent=2)
        
        print(f"Saved {len(features)} features to {output_dir}")

async def example_mixed_batch():
    """Example: Retrieve features from multiple transcoders with different indices"""
    async with FastFeatureRetriever() as retriever:
        # Real-world example with different transcoder_ids and scattered feature indices
        feature_specs = [
            ("gemma-2-2b", 1),
            ("gemma-2-2b", 15),
            ("gemma-2-2b", 47),
            ("another-model", 3),  # Different transcoder
            ("gemma-2-2b", 100),
            ("yet-another-model", 0),  # Different transcoder
            ("gemma-2-2b", 250)
        ]
        
        start_time = time.time()
        features = await retriever.get_features_batch(feature_specs)
        end_time = time.time()
        
        print(f"Mixed batch retrieval:")
        print(f"Requested {len(feature_specs)} features, got {len(features)} successfully")
        print(f"Time taken: {end_time - start_time:.3f} seconds")
        
        # Show which ones succeeded
        for spec in feature_specs:
            status = "✓" if spec in features else "✗"
            print(f"  {status} {spec[0]}, feature {spec[1]}")
        
        return features

# Non-async usage examples
def example_sync_usage():
    """Examples of how to use the synchronous interface"""
    print("=== Synchronous Usage Examples ===")
    
    # Single feature
    print("Getting single feature...")
    feature = get_feature("gemma-2-2b", 0)
    if feature:
        print(f"✓ Got feature with keys: {list(feature.keys())[:5]}...")
    else:
        print("✗ Failed to get feature")
    
    # Batch features
    print("\nGetting batch features...")
    feature_specs = [
        ("gemma-2-2b", 1),
        ("gemma-2-2b", 5),
        ("gemma-2-2b", 10)
    ]
    features = get_features_batch(feature_specs)
    print(f"✓ Got {len(features)} out of {len(feature_specs)} requested features")
    
    # Range features
    print("\nGetting range of features...")
    range_features = get_feature_range("gemma-2-2b", 0, 4)
    print(f"✓ Got {len(range_features)} features from range 0-4")
    
    return features

# Main execution
if __name__ == "__main__":
    print("Fast Feature Retrieval Script")
    print("=" * 50)
    
    # Show synchronous usage first
    example_sync_usage()
    print()
    
    # Debug the specific problematic URL first
    print("Debugging problematic URL...")
    asyncio.run(debug_specific_feature())
    print()
    
    # Run async examples
    print("=== Async Usage Examples ===")
    asyncio.run(example_single_feature())
    print()
    asyncio.run(example_batch_features())
    print()
    asyncio.run(example_mixed_batch())
    print()
    asyncio.run(example_range_features())
    print()
    asyncio.run(benchmark_performance())
    print()
    asyncio.run(save_features_to_disk())