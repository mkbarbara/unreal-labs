"""Step 2: Remove text from all extracted frames in video intervals"""

from pathlib import Path
from typing import List, Optional

from utils.logger import setup_logger
from utils.config import Config
from utils.cache_manager import CacheManager
from utils.falai_worker import FalAIWorker
from utils.download_file import download_file
from schemas import VideoInterval

logger = setup_logger(__name__)


async def remove_text_from_intervals(
    video_intervals: List[VideoInterval],
    work_dir: Path,
    config: Config,
    input_video_path: Optional[str] = None,
    cache_manager: Optional[CacheManager] = None,
) -> List[VideoInterval]:
    """
    Remove text from all extracted frames in video intervals

    Args:
        video_intervals: List of FrameData objects with start_frame and end_frame paths
        work_dir: Working directory for cleaned frames
        config: Pipeline configuration
        input_video_path: Optional path to input video for cache key generation
        cache_manager: Optional cache manager for caching results

    Returns:
        Updated video_intervals with cleaned frame paths
    """
    logger.info(f"Cleaning up text from {len(video_intervals)} intervals")

    work_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first
    if cache_manager and input_video_path:
        cached_data = cache_manager.load("text_removal", input_video_path,
                                         model=config.img2img_model)
        if cached_data:
            logger.info("Using cached text removal results")
            return [VideoInterval(**item) for item in cached_data]

    fal_client = FalAIWorker.get_instance()
    cleaned_frame_pairs = []

    for video_interval in video_intervals:
        interval_index = video_interval.index
        logger.info(f"Processing interval {interval_index}")

        # Process start frame
        start_cleaned_path = work_dir / f"interval_{interval_index:03d}_start_cleaned.jpg"
        await remove_text_from_single_frame(
            video_interval.start_frame_path,
            start_cleaned_path,
            fal_client,
            config
        )

        # Process end frame
        end_cleaned_path = work_dir / f"interval_{interval_index:03d}_end_cleaned.jpg"
        await remove_text_from_single_frame(
            video_interval.end_frame_path,
            end_cleaned_path,
            fal_client,
            config
        )

        # Create updated frame data with cleaned paths
        cleaned_data = VideoInterval(
            index=video_interval.index,
            start_frame_path=start_cleaned_path,
            end_frame_path=end_cleaned_path,
            start_time=video_interval.start_time,
            end_time=video_interval.end_time,
            duration=video_interval.duration,
            fps=video_interval.fps,
        )
        cleaned_frame_pairs.append(cleaned_data)

        logger.info(f"Cleaned frames for interval {interval_index}")

    logger.info(f"Successfully cleaned all frames")

    # Save to cache
    if cache_manager and input_video_path:
        cache_data = [frame_data.model_dump(mode='json') for frame_data in cleaned_frame_pairs]
        cache_manager.save("text_removal", input_video_path, cache_data,
                          model=config.img2img_model)

    return cleaned_frame_pairs


async def remove_text_from_single_frame(
    frame_path: Path,
    output_path: Path,
    fal_client,
    config: Config
) -> Path:
    """
    Remove text from a single frame

    Args:
        frame_path: Path to original frame
        output_path: Path for cleaned frame
        fal_client: FalAI worker instance
        config: Pipeline configuration

    Returns:
        Path to cleaned frame
    """
    try:
        # Upload frame
        frame_url = await fal_client.upload_file(frame_path)

        # Remove text using the image editing model
        result = await fal_client.generate(
            model=config.img2img_model,
            arguments={
                "prompt": "remove all text and captions from image, keep everything else intact",
                "image_urls": [frame_url],
                "aspect_ratio": "9:16",
            }
        )

        # Download cleaned image
        cleaned_url = result["images"][0]["url"]
        await download_file(cleaned_url, str(output_path))

        logger.debug(f"Cleaned frame: {frame_path} -> {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Failed to remove text from {frame_path}: {str(e)}")
        raise
