"""Step 1: Splits video into fixed intervals"""

import cv2
from pathlib import Path
from typing import List, Optional

from utils.logger import setup_logger
from utils.cache_manager import CacheManager
from utils.audio_utils import extract_audio
from schemas import VideoInterval

logger = setup_logger(__name__)


def _save_frame(cap, frame_num, path):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Failed to read frame at position {frame_num}")
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])


async def split_video_into_intervals(
    input_video_path: str,
    work_dir: Path,
    audio_dir: Path,
    interval: int,
    cache_manager: Optional[CacheManager] = None,
) -> List[VideoInterval]:
    """
    Splits video into intervals. Extracts start and end frames for each interval.
    Only complete intervals are extracted - partial intervals at the end are skipped.

    Args:
        input_video_path: Path to source video
        work_dir: Working directory for frame outputs
        audio_dir: Directory for audio outputs
        interval: Fixed duration in seconds for each interval
        cache_manager: Optional cache manager for caching results

    Returns:
        List of VideoInterval objects containing description for each interval
    """
    logger.info(f"Extracting frames at {interval}s intervals from {input_video_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first
    if cache_manager:
        cached_data = cache_manager.load("frame_extraction", input_video_path)
        if cached_data:
            logger.info("Using cached frame extraction results")
            return [VideoInterval(**item) for item in cached_data]

    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    logger.info(f"Video properties: Duration={duration:.2f}s, FPS={fps}, Total frames={total_frames}")

    # Calculate frames per interval
    interval_frames = int(fps * interval)

    logger.info(f"Extracting intervals every {interval}s ({interval_frames} frames)")

    frame_pairs = []
    interval_index = 0

    while True:
        # Calculate start and end times for this interval
        start_time = interval_index * interval
        end_time = start_time + interval

        # Check if we have enough video duration for a complete interval
        if end_time > duration:
            logger.info(
                f"Skipping incomplete interval at end (would need {start_time:.2f}s - {end_time:.2f}s, but video ends at {duration:.2f}s)")
            break

        # Convert times to frame numbers
        start_frame_num = int(start_time * fps)
        end_frame_num = int(end_time * fps) - 1

        # Safety check
        if end_frame_num >= total_frames:
            logger.info(
                f"Skipping incomplete interval at end (would need frames {start_frame_num}-{end_frame_num}, but video ends at frame {total_frames - 1})")
            break

        start_frame_path = work_dir / f"interval_{interval_index:03d}_start.jpg"
        end_frame_path = work_dir / f"interval_{interval_index:03d}_end.jpg"

        # Extract & save start & end frames
        _save_frame(cap, start_frame_num, start_frame_path)
        _save_frame(cap, end_frame_num, end_frame_path)

        # Calculate actual duration
        duration_interval = end_time - start_time

        # Extract audio for this interval
        audio_path = audio_dir / f"interval_{interval_index:03d}_audio.wav"
        extract_audio(
            input_video_path,
            str(audio_path),
            start_time=start_time,
            duration=duration_interval
        )

        frame_pairs.append(
            VideoInterval(
                index=interval_index,
                start_frame_path=start_frame_path,
                end_frame_path=end_frame_path,
                start_time=start_time,
                end_time=end_time,
                duration=duration_interval,
                fps=fps,
                audio_path=str(audio_path),
            )
        )

        logger.info(
            f"Extracted interval {interval_index}: "
            f"frames {start_frame_num}-{end_frame_num} "
            f"({start_time:.2f}s - {end_time:.2f}s)"
        )

        interval_index += 1

    cap.release()
    logger.info(f"Extracted {len(frame_pairs)} intervals from video")

    # Save to cache
    if cache_manager:
        cache_data = [frame_data.model_dump(mode='json') for frame_data in frame_pairs]
        cache_manager.save("frame_extraction", input_video_path, cache_data)

    return frame_pairs
