"""Utilities for audio extraction and manipulation"""

import json
import subprocess
import shutil
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


def extract_audio(video_path: str, audio_output_path: str, start_time: float = None, duration: float = None):
    """
    Extract audio from the video file

    Args:
        video_path: Path to video file
        audio_output_path: Path to save extracted audio
        start_time: Optional start time in seconds for audio segment
        duration: Optional duration in seconds for audio segment
    """
    in_path = Path(video_path)
    if not in_path.exists():
        raise ValueError(f"Input file not found: {video_path}")

    out_path = Path(audio_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    codec_args = ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]

    # ---- Probe media (if ffprobe available) for duration and audio presence
    media_duration = None
    has_audio = True
    if shutil.which("ffprobe"):
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index,codec_name:format=duration",
            "-of", "json", str(in_path)
        ]
        pr = subprocess.run(probe_cmd, capture_output=True, text=True)
        if pr.returncode == 0:
            try:
                info = json.loads(pr.stdout or "{}")
                streams = info.get("streams", [])
                has_audio = len(streams) > 0
                if "format" in info and "duration" in info["format"]:
                    media_duration = float(info["format"]["duration"])
            except Exception:
                pass  # Non-fatal: continue best-effort
        else:
            # ffprobe failed; continue but skip duration guard
            pass

    if not has_audio:
        raise RuntimeError("No audio stream found in the input video.")

    if media_duration is not None and start_time is not None and start_time >= media_duration:
        raise ValueError(
            f"start_time ({start_time}s) is beyond media duration ({media_duration:.2f}s)."
        )

    # Helper to run ffmpeg and capture stderr for diagnostics
    def _run_ffmpeg(cmd):
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode, res.stderr.strip()

    # Build a command (fast seek: -ss before -i)
    def _build_cmd(fast_seek: bool):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if fast_seek and start_time is not None:
            cmd += ["-ss", f"{start_time}"]
        cmd += ["-i", str(in_path)]
        if (not fast_seek) and start_time is not None:
            cmd += ["-ss", f"{start_time}"]
        cmd += [
            "-map", "0:a:0",  # first audio track explicitly
            "-vn", "-sn", "-dn"
        ]
        if duration is not None:
            cmd += ["-t", f"{duration}"]
        cmd += codec_args
        cmd += [str(out_path)]
        return cmd

    # 1) Try fast seek
    cmd1 = _build_cmd(fast_seek=True)
    code1, err1 = _run_ffmpeg(cmd1)

    # If failed or produced empty file, try accurate seek
    need_retry = (code1 != 0) or (out_path.exists() and out_path.stat().st_size == 0)
    if need_retry:
        # Remove 0-byte placeholder before retry to avoid confusion
        if out_path.exists() and out_path.stat().st_size == 0:
            try:
                out_path.unlink()
            except Exception:
                pass

        cmd2 = _build_cmd(fast_seek=False)
        code2, err2 = _run_ffmpeg(cmd2)

        if code2 != 0 or (out_path.exists() and out_path.stat().st_size == 0):
            details = "\n--- ffmpeg (fast seek) ---\n" + err1
            details += "\n--- ffmpeg (accurate seek) ---\n" + err2
            raise RuntimeError(f"Audio extraction failed. Diagnostics:\n{details}")

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Audio extraction failed: output file not created or is empty.")

    return str(out_path)


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
