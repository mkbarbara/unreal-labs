import subprocess
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)


def add_text_layer(
    video_path: Path,
    text_layer_path: Path,
    output_path: Path
) -> Path:
    """
    Add extracted text layer overlay to the final video using ffmpeg

    Args:
        video_path: Path to the input video
        text_layer_path: Path to the text layer PNG file
        output_path: Path for the output video with text overlay

    Returns:
        Path to the video with text overlay
    """
    logger.info(f"Adding text layer from {text_layer_path} to video")

    if not text_layer_path.exists():
        logger.warning(f"Text layer not found at {text_layer_path}, skipping overlay")
        return video_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-loop", "1",
        "-i", str(text_layer_path),
        "-filter_complex",
        "[1:v][0:v]scale2ref=w=iw:h=ih[ovl][vid];[vid][ovl]overlay=0:0:shortest=1:format=auto",
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        logger.info(f"Text layer added successfully: {output_path}")
        return output_path
    else:
        logger.warning(f"Failed to add text layer: {result.stderr}")
        return video_path
