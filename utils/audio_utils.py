"""Utilities for audio extraction and manipulation"""

import subprocess

from utils.logger import setup_logger

logger = setup_logger(__name__)


def extract_audio(video_path: str, audio_output_path: str):
    """
    Extract audio from the video file

    Args:
        video_path: Path to video file
        audio_output_path: Path to save extracted audio
    """
    try:
        # Check if video has audio track
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]

        probe_result = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            check=False
        )

        # If no audio stream found
        if probe_result.returncode != 0 or not probe_result.stdout.strip():
            logger.warning(f"No audio track found in {video_path}")
            return

        # Extract audio
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'aac',  # Use AAC codec
            '-b:a', '128k',  # Audio bitrate
            '-y',  # Overwrite output
            audio_output_path
        ]

        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        logger.debug(f"Extracted audio: {audio_output_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio: {e.stderr}")
    except Exception as e:
        logger.error(f"Audio extraction failed: {str(e)}")


def merge_video_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Merge video with audio track

    Args:
        video_path: Path to video file (without audio or with audio to replace)
        audio_path: Path to audio file
        output_path: Path to save merged video

    Returns:
        Path to merged video
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copy video codec
            '-c:a', 'aac',  # Re-encode audio to AAC
            '-b:a', '128k',  # Audio bitrate
            '-map', '0:v:0',  # Map video from first input
            '-map', '1:a:0',  # Map audio from second input
            '-shortest',  # Finish encoding when shortest stream ends
            '-y',  # Overwrite output
            output_path
        ]

        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        logger.info(f"Merged video with audio: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to merge video and audio: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Video-audio merge failed: {str(e)}")
        raise


def adjust_audio_duration(audio_path: str, duration: float, output_path: str) -> str:
    """
    Adjust audio duration (trim or loop)

    Args:
        audio_path: Path to audio file
        duration: Target duration in seconds
        output_path: Path to save adjusted audio

    Returns:
        Path to adjusted audio
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', audio_path,
            '-t', str(duration),  # Trim to duration
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            output_path
        ]

        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        logger.debug(f"Adjusted audio duration to {duration}s: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to adjust audio duration: {e.stderr}")
        raise
