"""Step 5: Edit cleaned video intervals with reference images"""

from pathlib import Path
from typing import List, Optional

from utils.logger import setup_logger
from utils.config import Config
from utils.falai_worker import FalAIWorker
from utils.openai_worker import OpenAIWorker
from utils.download_file import download_file
from utils.cache_manager import CacheManager
from schemas import VideoInterval, Person

logger = setup_logger(__name__)


async def detect_people_in_frame(
    frame_path: Path,
    openai_worker: OpenAIWorker,
    config: Config,
) -> List[Person]:
    """
    Detect people in a specific frame

    Args:
        frame_path: Path to frame
        openai_worker: OpenAI worker instance
        config: Pipeline configuration

    Returns:
        List of detected people in the frame
    """
    try:
        return await openai_worker.analyze_frame_for_people(frame_path, config)
    except Exception as e:
        logger.warning(f"Failed to detect people in frame {frame_path}: {str(e)}")
        return []


async def edit_single_frame(
    frame_path: Path,
    output_path: Path,
    prompt: str,
    config: Config,
    reference_images: List[Path] = None
) -> None:
    """
    Edit a single frame using the Gemini Flash model with optional reference images

    Args:
        frame_path: Path to frame to edit
        output_path: Where to save edited frame
        prompt: Editing instructions
        config: Pipeline configuration
        reference_images: Optional list of reference image paths for people in frame

    Returns:
        Path to edited frame
    """
    logger.info(f"Editing frame: {frame_path}")

    try:
        # Get FalAIWorker singleton instance
        fal_client = FalAIWorker.get_instance()

        # Upload source frame
        source_url = await fal_client.upload_file(str(frame_path))

        # Prepare image URLs list - source frame first, then reference images
        image_urls = [source_url]

        # Upload reference images if provided
        if reference_images:
            for ref_path in reference_images:
                ref_url = await fal_client.upload_file(str(ref_path))
                image_urls.append(ref_url)
            logger.info(f"Using {len(reference_images)} reference images")

        # Generate edited frame
        result = await fal_client.generate(
            model=config.img2img_model,
            arguments={
                "prompt": prompt,
                "image_urls": image_urls,
                "aspect_ratio": "9:16",
            }
        )

        # Download edited image
        edited_url = result["images"][0]["url"]

        await download_file(edited_url, str(output_path))
        logger.info(f"Saved edited frame: {str(output_path)}")

    except Exception as e:
        logger.error(f"Frame editing failed: {str(e)}")
        raise


async def edit_frames(
    cleaned_video_intervals: List[VideoInterval],
    new_person_registry: List[Person],
    work_dir: Path,
    config: Config,
    input_video_path: Optional[str] = None,
    cache_manager: Optional[CacheManager] = None,
) -> List[VideoInterval]:
    """
    Edit demographic attributes on all cleaned frames by transforming original people to new people

    Args:
        cleaned_video_intervals: List of cleaned video intervals from the text removal step
        new_person_registry: New people registry based on transformation theme
        work_dir: Working directory for outputs
        config: Pipeline configuration
        input_video_path: Optional path to input video for cache key generation
        cache_manager: Optional cache manager for caching results

    Returns:
        List of edited VideoInterval objects
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first
    if cache_manager and input_video_path:
        cached_data = cache_manager.load("edit_frames", input_video_path)
        if cached_data:
            logger.info("Using cached text removal results")
            return [VideoInterval(**item) for item in cached_data]

    # Get OpenAI client for dynamic prompt generation
    openai_worker = OpenAIWorker.get_instance()

    edited_intervals: List[VideoInterval] = []

    for video_interval in cleaned_video_intervals:
        interval_index = video_interval.index
        logger.info(f"Editing frames for interval {interval_index}")

        try:
            # Use the cleaned frames
            start_frame_path = video_interval.start_frame_path
            end_frame_path = video_interval.end_frame_path

            # Detect people in the start frame
            start_frame_people = await openai_worker.analyze_frame_for_people(start_frame_path, config)

            # Get reference images for detected people in the start frame (from NEW person registry)
            start_reference_images = get_reference_images_for_people(
                start_frame_people,
                new_person_registry,
                work_dir,
            )

            # Generate dynamic prompt for start frame using BOTH registries
            start_prompt = await generate_transformation_prompt_with_mapping(
                people_in_frame=start_frame_people,
                new_person_registry=new_person_registry,
                openai_worker=openai_worker,
                config=config,
            )

            # Edit the start frame with reference images
            start_edited_path = work_dir / f"interval_{interval_index:03d}_start_edited.jpg"
            await edit_single_frame(
                start_frame_path,
                start_edited_path,
                start_prompt,
                config,
                start_reference_images,
            )

            # Detect people in the end frame
            end_frame_people = await openai_worker.analyze_frame_for_people(end_frame_path, config)

            # Get reference images for detected people in the end frame
            end_reference_images = get_reference_images_for_people(
                end_frame_people,
                new_person_registry,
                work_dir,
            )

            # Generate dynamic prompt for end frame using BOTH registries
            end_prompt = await generate_transformation_prompt_with_mapping(
                people_in_frame=end_frame_people,
                new_person_registry=new_person_registry,
                openai_worker=openai_worker,
                config=config,
            )

            # Edit end frame with reference images
            end_edited_path = work_dir / f"interval_{interval_index:03d}_end_edited.jpg"
            await edit_single_frame(
                end_frame_path,
                end_edited_path,
                end_prompt,
                config,
                reference_images=end_reference_images,
            )

            edited_intervals.append(VideoInterval(
                index=interval_index,
                start_frame_path=start_edited_path,
                end_frame_path=end_edited_path,
                start_time=video_interval.start_time,
                end_time=video_interval.end_time,
                duration=video_interval.duration,
                fps=video_interval.fps,
                audio_path=video_interval.audio_path,
            ))

            logger.info(f"Edited frames for interval {interval_index}")

        except Exception as e:
            logger.error(f"Failed to edit frames for interval {interval_index}: {str(e)}")
            continue

    # Save to cache
    if cache_manager and input_video_path:
        cache_data = [frame_data.model_dump(mode='json') for frame_data in edited_intervals]
        cache_manager.save("edit_frames", input_video_path, cache_data)

    logger.info(f"Edited frames for {len(edited_intervals)} intervals")
    return edited_intervals


async def generate_transformation_prompt_with_mapping(
    people_in_frame: List[Person],
    new_person_registry: List[Person],
    openai_worker: OpenAIWorker,
    config: Config,
) -> str:
    """
    Generate transformation prompt that maps original people to new people

    Args:
        people_in_frame: People detected in the current frame
        new_person_registry: New people from step 4 (transformation targets)
        openai_worker: OpenAI worker instance
        config: Pipeline configuration

    Returns:
        Transformation prompt for the frame
    """
    # Build transformation mappings
    new_people_in_frame: List[Person] = []

    for person_in_frame in people_in_frame:
        person_id = person_in_frame.person_id

        # Find the original and new person by person_id
        new_person: Optional[Person] = None

        for new in new_person_registry:
            if new.person_id == person_id:
                new_person = new
                break

        new_people_in_frame.append(new_person)

    # Create the transformation prompt
    transformation_prompt = await openai_worker.generate_transformation_prompt(
        people_in_frame=people_in_frame,
        new_people_in_frame=new_people_in_frame,
        config=config,
    )
    logger.info(f"Transformation prompt: {transformation_prompt}")

    return transformation_prompt


def get_reference_images_for_people(
    people_in_frame: List[Person],
    person_registry: List[Person],
    work_dir: Path,
) -> List[Path]:
    """
    Get reference image paths for people detected in a frame

    Args:
        people_in_frame: List of people detected in current frame
        person_registry: Complete registry with reference images
        work_dir: Working directory for reference images

    Returns:
        List of reference image paths for matched people
    """
    reference_images = []

    for person_in_frame in people_in_frame:
        # Try to match this person to someone in the registry
        for registered_person in person_registry:
            # Simple matching by person_id if available
            if person_in_frame.person_id == registered_person.person_id:
                reference_path = work_dir.parent / "reference_images" / f"{person_in_frame.person_id}_new_reference.jpg"
                reference_images.append(reference_path)
                break

    return reference_images
