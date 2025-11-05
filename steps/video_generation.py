"""Step 6: Generate video clips using Veo3.1"""

from pathlib import Path
from typing import List, Optional

from schemas import VideoInterval
from utils.logger import setup_logger
from utils.config import Config
from utils.falai_worker import FalAIWorker
from utils.download_file import download_file
from utils.audio_utils import merge_video_audio, adjust_audio_duration

logger = setup_logger(__name__)


async def generate_single_video(
    start_frame_path: Path,
    end_frame_path: Path,
    duration: float,
    output_path: Path,
    prompt: str,
    config: Config,
    # audio_path: Optional[str] = None
) -> str:
    """
    Generate a single video clip using Veo3.1

    Args:
        start_frame_path: Path to edited start frame
        end_frame_path: Path to edited end frame
        duration: Clip duration in seconds
        output_path: Where to save generated video
        prompt: Generation instructions
        config: Pipeline configuration
        # audio_path: Optional path to original audio to merge with generated video

    Returns:
        Path to generated video
    """
    logger.info(f"Generating video: {output_path}")

    try:
        # Get FalAIWorker singleton instance
        fal_client = FalAIWorker.get_instance()

        # Upload frames to fal.ai
        start_url = await fal_client.upload_file(str(start_frame_path))
        end_url = await fal_client.upload_file(str(end_frame_path))

        # Submit request
        result = await fal_client.generate(
            model=config.video_model,
            arguments={
                "prompt": prompt,
                "first_frame_url": start_url,
                "last_frame_url": end_url,
                "duration": f"{round(duration)}s",
                "aspect_ratio": "9:16",
                "resolution": "720p",
                "generate_audio": False,
            }
        )

        # Download generated video
        video_url = result["video"]["url"]
        download_file_path = await download_file(video_url, str(output_path))
        return download_file_path

        # # If audio path provided, merge with original audio
        # if audio_path and Path(audio_path).exists():
        #     # Download to temporary path first
        #     temp_video_path = output_path.replace(".mp4", "_no_audio.mp4")
        #     await download_file(video_url, temp_video_path)
        #
        #     # Adjust audio duration to match generated video duration
        #     temp_audio_path = output_path.replace(".mp4", "_adjusted_audio.aac")
        #     adjusted_audio = adjust_audio_duration(audio_path, duration, temp_audio_path)
        #
        #     # Merge video with audio
        #     final_path = merge_video_audio(temp_video_path, adjusted_audio, output_path)
        #
        #     # Clean up temporary files
        #     Path(temp_video_path).unlink(missing_ok=True)
        #     Path(temp_audio_path).unlink(missing_ok=True)
        #
        #     logger.info(f"Saved generated video with original audio: {final_path}")
        #     return final_path
        # else:
        #     # No audio to merge, just download video
        #     download_file_path = await download_file(video_url, output_path)
        #     logger.info(f"Saved generated video (no audio): {download_file_path}")
        #     return download_file_path

    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}")
        raise


async def generate_video_clips(
    edited_video_intervals: List[VideoInterval],
    work_dir: Path,
    config: Config
) -> List[str]:
    """
    Generate video clips for all edited frame pairs

    Args:
        edited_video_intervals: Edited video intervals
        work_dir: Working directory for outputs
        config: Pipeline configuration

    Returns:
        List of paths to generated video clips
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load generation prompt
    prompt = config.get_prompt("video_generation")

    generated_clips = []

    for video_interval in edited_video_intervals:
        interval_index = video_interval.index
        logger.info(f"Generating video for interval {interval_index}")

        try:
            output_path = work_dir / f"interval_{interval_index:03d}_generated.mp4"

            generated_path = await generate_single_video(
                video_interval.start_frame_path,
                video_interval.end_frame_path,
                video_interval.duration,
                output_path,
                prompt,
                config,
                # audio_path=video_interval.get("audio_path")
            )

            generated_clips.append(generated_path)
            logger.info(f"Generated clip {clip_index}")

        except Exception as e:
            logger.error(f"Failed to generate video for clip {clip_index}: {str(e)}")
            raise

    logger.info(f"Generated {len(generated_clips)} video clips")
    return generated_clips
