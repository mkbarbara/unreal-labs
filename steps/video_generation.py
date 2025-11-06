"""Step 6: Generate new video clips using Veo3.1"""

from pathlib import Path
from typing import List

from schemas import VideoInterval
from utils.logger import setup_logger
from utils.config import Config
from utils.falai_worker import FalAIWorker
from utils.download_file import download_file
from utils.audio_utils import merge_video_audio

logger = setup_logger(__name__)


async def generate_single_interval(
    start_frame_path: Path,
    end_frame_path: Path,
    duration: float,
    output_path: str,
    prompt: str,
    config: Config,
) -> str:
    """
    Generate a single video interval using Veo3.1

    Args:
        start_frame_path: Path to edited start frame
        end_frame_path: Path to edited end frame
        duration: Interval duration in seconds
        output_path: Where to save generated video
        prompt: Generation instructions
        config: Pipeline configuration

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
                "generate_audio": False,    # we will use audio from the original video
            }
        )

        # Download generated video
        video_url = result["video"]["url"]
        download_file_path = await download_file(video_url, str(output_path))
        return download_file_path

    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}")
        raise


async def generate_video_intervals(
    edited_video_intervals: List[VideoInterval],
    work_dir: Path,
    config: Config
) -> List[str]:
    """
    Generate video intervals for all edited frame pairs

    Args:
        edited_video_intervals: Edited video intervals
        work_dir: Working directory for outputs
        config: Pipeline configuration

    Returns:
        List of paths to generated video intervals
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load generation prompt
    prompt = config.get_prompt("video_generation")

    generated_intervals = []

    for video_interval in edited_video_intervals:
        interval_index = video_interval.index
        logger.info(f"Generating video for interval {interval_index}")

        try:
            # Check if we need to merge audio later
            has_audio = video_interval.audio_path and Path(video_interval.audio_path).exists()

            # If audio exists, generate to the temp path first
            if has_audio:
                temp_output_path = work_dir / f"interval_{interval_index:03d}_generated_no_audio.mp4"
                final_output_path = work_dir / f"interval_{interval_index:03d}_generated.mp4"
            else:
                final_output_path = work_dir / f"interval_{interval_index:03d}_generated.mp4"
                temp_output_path = final_output_path

            generated_path = await generate_single_interval(
                video_interval.start_frame_path,
                video_interval.end_frame_path,
                video_interval.duration,
                str(temp_output_path),
                prompt,
                config,
            )

            # Merge with original audio if available
            if has_audio:
                logger.info(f"Merging interval {interval_index} with original audio")

                # Merge video with adjusted audio
                final_path = merge_video_audio(
                    str(generated_path),
                    str(video_interval.audio_path),
                    str(final_output_path)
                )

                # Clean up temporary files
                Path(temp_output_path).unlink(missing_ok=True)

                generated_intervals.append(final_path)
            else:
                logger.info(f"No audio available for interval {interval_index}")
                generated_intervals.append(str(final_output_path))

            logger.info(f"Generated interval {interval_index}")

        except Exception as e:
            logger.error(f"Failed to generate video for interval {interval_index}: {str(e)}")
            raise

    logger.info(f"Generated {len(generated_intervals)} video intervals")
    return generated_intervals
