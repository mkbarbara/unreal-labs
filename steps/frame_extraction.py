"""Step 2: Extract the first and the last frames from each clip"""

import cv2
from pathlib import Path
from typing import List, Dict

from utils.logger import setup_logger
from utils.config import Config

logger = setup_logger(__name__)


async def extract_conditioning_frames(
    clips: List[Dict],
    work_dir: Path,
    config: Config
) -> List[Dict[str, str]]:
    """
    Extract the first and the last frames from each clip

    Args:
        clips: List of clip metadata from step 1
        work_dir: Working directory for frame outputs
        config: Pipeline configuration

    Returns:
        List of dicts with frame info: {clip_index, start_frame, end_frame, clip_path}
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    conditioning_frames = []

    for clip in clips:
        clip_index = clip["clip_index"]
        clip_path = clip["path"]

        logger.info(f"Extracting frames from clip {clip_index}")

        try:
            cap = cv2.VideoCapture(clip_path)

            if not cap.isOpened():
                logger.error(f"Failed to open clip: {clip_path}")
                continue

            # Extract first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame = cap.read()

            if not ret:
                logger.error(f"Failed to read first frame from clip {clip_index}")
                cap.release()
                continue

            # Extract last frame
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, last_frame = cap.read()

            if not ret:
                logger.error(f"Failed to read last frame from clip {clip_index}")
                cap.release()
                continue

            cap.release()

            # Save frames
            start_frame_path = work_dir / f"clip_{clip_index:03d}_start.{config.frame_format}"
            end_frame_path = work_dir / f"clip_{clip_index:03d}_end.{config.frame_format}"

            cv2.imwrite(
                str(start_frame_path),
                first_frame,
                [cv2.IMWRITE_JPEG_QUALITY, config.frame_quality]
            )
            cv2.imwrite(
                str(end_frame_path),
                last_frame,
                [cv2.IMWRITE_JPEG_QUALITY, config.frame_quality]
            )

            conditioning_frames.append({
                "clip_index": clip_index,
                "start_frame": str(start_frame_path),
                "end_frame": str(end_frame_path),
                "clip_path": clip_path,
                "audio_path": clip.get("audio_path"),  # Original audio path
                "duration": clip["duration"],
                "fps": clip["fps"]
            })

            logger.info(f"Extracted frames for clip {clip_index}")

        except Exception as e:
            logger.error(f"Failed to extract frames from clip {clip_index}: {str(e)}")
            continue

    logger.info(f"Extracted frames for {len(conditioning_frames)} clips")
    return conditioning_frames
