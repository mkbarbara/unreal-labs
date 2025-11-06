"""Step 7: Reassemble video intervals into final output"""

from pathlib import Path
from typing import List
import subprocess

from utils.logger import setup_logger
from utils.config import Config

logger = setup_logger(__name__)


async def reassemble_video(
    generated_intervals: List[str],
    output_path: str,
) -> str:
    """
    Concatenate all generated intervals into the final video

    Args:
        generated_intervals: List of paths to generated intervals from Step. 6 (in order)
        output_path: Path for final output video
        config: Pipeline configuration

    Returns:
        Path to final reassembled video
    """
    logger.info(f"Reassembling {len(generated_intervals)} intervals into final video")

    try:
        # Create a temporary file list for ffmpeg concat
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        concat_file = output_dir / "concat_list.txt"

        with open(concat_file, 'w') as f:
            for video_interval in generated_intervals:
                # ffmpeg concat demuxer format
                f.write(f"file '{Path(video_interval).absolute()}'\n")

        # Use ffmpeg to concatenate videos
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',  # Copy streams without re-encoding for speed
            '-y',  # Overwrite output file
            output_path
        ]

        logger.info(f"Running ffmpeg: {' '.join(cmd)}")

        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Clean up concat file
        concat_file.unlink()

        logger.info(f"Final video created: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Video reassembly failed: {str(e)}")
        raise
