"""Step 3: Detect and describe people in the video"""

from pathlib import Path
from typing import List, Optional

from utils.logger import setup_logger
from utils.config import Config
from utils.cache_manager import CacheManager
from utils.openai_worker import OpenAIWorker
from schemas import VideoInterval, Person

logger = setup_logger(__name__)


async def detect_and_describe_people(
    cleaned_video_intervals: List[VideoInterval],
    config: Config,
    input_video_path: Optional[str] = None,
    cache_manager: Optional[CacheManager] = None,
) -> List[Person]:
    """
    Detect and describe all people appearing in the cleaned frames

    Args:
        cleaned_video_intervals: List of cleaned video intervals from the text removal step
        config: Pipeline configuration
        input_video_path: Optional path to input video for cache key generation
        cache_manager: Optional cache manager for caching results

    Returns:
        List of person profiles with consolidated attributes
    """
    logger.info(f"Starting person detection using {len(cleaned_video_intervals)} frame pairs")

    # Check cache first
    if cache_manager and input_video_path:
        cached_data = cache_manager.load("person_detection", input_video_path)
        if cached_data:
            logger.info("Using cached person detection results")
            return [Person(**item) for item in cached_data]

    # Get OpenAI worker
    openai_worker = OpenAIWorker.get_instance()

    # Collect all frames for analysis (both start and end frames in each interval)
    frames_to_analyze: list[Path] = []
    for video_interval in cleaned_video_intervals:
        frames_to_analyze.append(video_interval.start_frame_path)
        frames_to_analyze.append(video_interval.end_frame_path)

    logger.info(f"Analyzing {len(frames_to_analyze)} frames for people")

    # Analyze each frame for people
    frame_analyses = []
    for frame_path in frames_to_analyze:
        analysis = await openai_worker.analyze_frame_for_people(
            frame_path,
            config,
        )
        frame_analyses.append(analysis)

    person_registry = await openai_worker.consolidate_person_descriptions(frame_analyses, config)
    logger.info(f"Detected {len(person_registry)} unique individuals in video")

    # Save to cache
    if cache_manager and input_video_path:
        cache_data = [person.model_dump(mode='json') for person in person_registry]
        cache_manager.save("person_detection", input_video_path, cache_data)

    return person_registry
