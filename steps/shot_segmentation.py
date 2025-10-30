"""Step 1: Segment video into fixed-duration clips"""

import cv2
import subprocess
from pathlib import Path
from typing import List, Dict

from utils.logger import setup_logger
from utils.config import Config
from utils.audio_utils import extract_audio

logger = setup_logger(__name__)


async def segment_video(
    video_path: str,
    work_dir: Path,
    config: Config
) -> List[Dict[str, any]]:
    """
    Segment video into fixed-duration clips

    Args:
        video_path: Path to input video
        work_dir: Working directory for clip outputs
        config: Pipeline configuration

    Returns:
        List of dicts with clip info: {path, start_time, end_time, duration}
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Segmenting video: {video_path}")

    try:
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        cap.release()

        logger.info(f"Video duration: {total_duration:.2f}s, FPS: {fps}")

        # Calculate clip intervals
        clip_duration = config.max_clip_duration  # Use max_clip_duration as a fixed interval
        clips = []

        start_time = 0.0
        clip_index = 0

        while start_time < total_duration:
            end_time = min(start_time + clip_duration, total_duration)
            duration = end_time - start_time

            # Do not add clips that are not divided by 8s - temporary limitation of veo3.1 vie fal.ai
            if duration < clip_duration:
                logger.warning(f"Can not generate clips that are not equal to 8s via fal.ai")
                break

            # Extract clip using ffmpeg
            clip_path = work_dir / f"clip_{clip_index:03d}.mp4"

            # Use re-encoding instead of copy to ensure proper keyframes
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),  # Seek before input for faster processing
                '-i', video_path,
                '-t', str(duration),
                '-c:v', 'libx264',  # Re-encode video with H.264
                '-preset', 'medium',  # Balanced speed/quality
                '-crf', '23',  # Quality (lower = better, 23 is default)
                '-c:a', 'aac',  # Re-encode audio
                '-b:a', '128k',  # Audio bitrate
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                '-y',  # Overwrite output file
                str(clip_path)
            ]

            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Extract audio from clip
            audio_path = work_dir / f"clip_{clip_index:03d}_audio.aac"
            extracted_audio = extract_audio(str(clip_path), str(audio_path))

            clips.append({
                "path": str(clip_path),
                "audio_path": extracted_audio,  # May be None if no audio
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "fps": fps,
                "clip_index": clip_index
            })

            logger.info(f"Clip {clip_index}: {duration:.2f}s ({start_time:.2f}s - {end_time:.2f}s)")

            start_time = end_time
            clip_index += 1

        logger.info(f"Segmented into {len(clips)} clips")
        return clips

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Video segmentation failed: {str(e)}")
        raise