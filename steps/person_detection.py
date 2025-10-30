"""Step 0: Detect and describe people in the video"""

import cv2
from pathlib import Path
from typing import List, Dict

from utils.logger import setup_logger
from utils.config import Config
from utils.openai_worker import OpenAIWorker

logger = setup_logger(__name__)


async def detect_and_describe_people(
    input_video_path: str,
    work_dir: Path,
    config: Config
) -> List[Dict]:
    """
    Detect and describe all people appearing in the video

    Args:
        input_video_path: Path to source video
        work_dir: Working directory for temporary files
        config: Pipeline configuration

    Returns:
        List of person profiles with consolidated attributes
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting person detection for video: {input_video_path}")

    # Get OpenAI worker
    openai_worker = OpenAIWorker.get_instance()

    # Sample frames from the video
    sampled_frames = await sample_frames_for_analysis(
        input_video_path,
        work_dir,
        config.person_detection_frame_interval,
    )

    logger.info(f"Sampled {len(sampled_frames)} frames for analysis")

    # Analyze each frame for people
    frame_analyses = []
    for frame_data in sampled_frames:
        analysis = await openai_worker.analyze_frame_for_people(
            frame_data["path"],
            frame_data["frame_number"],
            config,
        )
        frame_analyses.append(analysis)

    person_registry = await openai_worker.consolidate_person_descriptions(frame_analyses)
    logger.info(f"Detected {len(person_registry)} unique individuals in video")

    # Clean up sampled frames
    for frame_data in sampled_frames:
        Path(frame_data["path"]).unlink(missing_ok=True)

    return person_registry


async def sample_frames_for_analysis(
    video_path: str,
    output_dir: Path,
    interval_seconds: float = 2.0,
) -> List[Dict]:
    """
    Sample frames from the video at regular intervals

    Args:
        video_path: Path to video file
        output_dir: Where to save sampled frames
        interval_seconds: Time interval between samples

    Returns:
        List of dicts with frame paths and metadata
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = int(fps * interval_seconds)
    sampled_frames = []

    frame_number = 0
    sample_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            frame_path = output_dir / f"sample_{sample_index:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)

            sampled_frames.append({
                "path": str(frame_path),
                "frame_number": frame_number,
                "timestamp": frame_number / fps
            })

            sample_index += 1

        frame_number += 1

    cap.release()

    logger.info(f"Sampled {len(sampled_frames)} frames from {total_frames} total frames")

    return sampled_frames
