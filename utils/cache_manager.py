"""Cache manager for pipeline steps"""

import json
import hashlib
from pathlib import Path
from typing import Any, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class CacheManager:
    """Manages caching of pipeline step results"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _generate_cache_key(step_name: str, video_path: str) -> str:
        """Generate a unique cache key based on step name, video path, and parameters"""
        # Get video file hash (using modification time and size for efficiency)
        video_file = Path(video_path)
        if video_file.exists():
            video_hash = f"{video_file.stat().st_mtime}_{video_file.stat().st_size}"
        else:
            video_hash = "unknown"

        # Combine all components
        key_components = f"{step_name}_{video_path}_{video_hash}"

        # Generate MD5 hash for a clean filename
        cache_key = hashlib.md5(key_components.encode()).hexdigest()

        return cache_key

    def get_cache_path(self, step_name: str, video_path: str) -> Path:
        """Get the cache file path for a specific step"""
        cache_key = self._generate_cache_key(step_name, video_path)
        return self.cache_dir / f"{step_name}_{cache_key}.json"

    def load(self, step_name: str, video_path: str) -> Optional[Any]:
        """Load cached results if they exist"""
        cache_path = self.get_cache_path(step_name, video_path)

        if not cache_path.exists():
            logger.info(f"No cache found for {step_name}")
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            # Verify that all referenced files still exist
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, str) and key.endswith(('_frame', '_path', 'frame_path')):
                                if not Path(value).exists():
                                    logger.info(f"Cache invalid: referenced file {value} no longer exists")
                                    return None

            logger.info(f"Loaded cached results for {step_name} from {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache for {step_name}: {str(e)}")
            return None

    def save(self, step_name: str, video_path: str, data: Any):
        """Save results to cache"""
        cache_path = self.get_cache_path(step_name, video_path)

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved cache for {step_name} to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {step_name}: {str(e)}")

    def clear(self, step_name: Optional[str] = None):
        """Clear cache files. If step_name is provided, only clear that step's cache"""
        if step_name:
            pattern = f"{step_name}_*.json"
        else:
            pattern = "*.json"

        deleted_count = 0
        for cache_file in self.cache_dir.glob(pattern):
            try:
                cache_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {str(e)}")

        logger.info(f"Cleared {deleted_count} cache file(s)")
