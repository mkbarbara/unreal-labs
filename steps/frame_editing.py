"""Step 3: Edit demographic attributes on conditioning frames"""

import base64
from pathlib import Path
from typing import List, Dict

from utils.logger import setup_logger
from utils.config import Config
from utils.falai_worker import FalAIWorker
from utils.openai_worker import OpenAIWorker
from utils.download_file import download_file

logger = setup_logger(__name__)


def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def detect_people_in_frame(
    frame_path: str,
    openai_worker: OpenAIWorker,
) -> List[Dict]:
    """
    Detect people in a specific frame

    Args:
        frame_path: Path to frame
        openai_worker: OpenAI worker instance

    Returns:
        List of detected people in the frame
    """
    try:
        result = await openai_worker.analyze_frame_for_people(frame_path, 0)
        return result.get("people", [])
    except Exception as e:
        logger.warning(f"Failed to detect people in frame {frame_path}: {str(e)}")
        return []


async def edit_single_frame(
    frame_path: str,
    output_path: str,
    prompt: str,
    config: Config
) -> str:
    """
    Edit a single frame using Gemini Flash via OpenAI worker

    Args:
        frame_path: Path to frame to edit
        output_path: Where to save edited frame
        prompt: Editing instructions
        config: Pipeline configuration

    Returns:
        Path to edited frame
    """
    logger.info(f"Editing frame: {frame_path}")

    try:
        # Get FalAIWorker singleton instance
        fal_client = FalAIWorker.get_instance()

        # Upload source frame
        source_url = await fal_client.upload_file(frame_path)

        # Generate edited frame
        result = await fal_client.generate(
            model=config.img2img_model,
            arguments={
                "prompt": prompt,
                "image_urls": [source_url],
                "aspect_ratio": "9:16",
            }
        )

        # Download edited image
        edited_url = result["images"][0]["url"]

        download_file_path = await download_file(edited_url, output_path)
        logger.info(f"Saved edited frame: {download_file_path}")
        return download_file_path

    except Exception as e:
        logger.error(f"Frame editing failed: {str(e)}")
        raise


async def edit_frames(
    conditioning_frames: List[Dict],
    person_registry: List[Dict],
    transformation_theme: str,
    work_dir: Path,
    config: Config
) -> List[Dict[str, str]]:
    """
    Edit demographic attributes on all conditioning frames

    Args:
        conditioning_frames: Frame data from step 2
        person_registry: Detected people registry from person detection step
        transformation_theme: Overall transformation theme (e.g., "Black people")
        work_dir: Working directory for outputs
        config: Pipeline configuration

    Returns:
        List of dicts with edited frame info
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Get OpenAI client for dynamic prompt generation
    openai_worker = OpenAIWorker.get_instance()

    edited_frames = []

    for frame_data in conditioning_frames:
        clip_index = frame_data["clip_index"]

        logger.info(f"Editing frames for clip {clip_index}")

        try:
            # Detect people in the start frame
            start_people = await detect_people_in_frame(
                frame_data["start_frame"],
                openai_worker,
            )

            # Generate dynamic prompt for start frame
            start_prompt = await openai_worker.generate_transformation_prompt(
                people_in_frame=start_people,
                transformation_theme=transformation_theme,
                person_registry=person_registry,
            )

            # Edit start frame
            start_edited_path = work_dir / f"clip_{clip_index:03d}_start_edited.jpg"
            await edit_single_frame(
                frame_data["start_frame"],
                str(start_edited_path),
                start_prompt,
                config,
            )

            # Detect people in the end frame
            end_people = await detect_people_in_frame(
                frame_data["end_frame"],
                openai_worker,
            )

            # Generate dynamic prompt for end frame
            end_prompt = await openai_worker.generate_transformation_prompt(
                people_in_frame=end_people,
                transformation_theme=transformation_theme,
                person_registry=person_registry,
            )

            # Edit end frame
            end_edited_path = work_dir / f"clip_{clip_index:03d}_end_edited.jpg"
            await edit_single_frame(
                frame_data["end_frame"],
                str(end_edited_path),
                end_prompt,
                config,
            )

            edited_frames.append({
                "clip_index": clip_index,
                "start_frame_edited": str(start_edited_path),
                "end_frame_edited": str(end_edited_path),
                "original_clip_path": frame_data["clip_path"],
                "audio_path": frame_data.get("audio_path"),
                "duration": frame_data["duration"],
                "fps": frame_data["fps"],
            })

            logger.info(f"Edited frames for clip {clip_index}")

        except Exception as e:
            logger.error(f"Failed to edit frames for clip {clip_index}: {str(e)}")
            continue

    logger.info(f"Edited frames for {len(edited_frames)} clips")
    return edited_frames
