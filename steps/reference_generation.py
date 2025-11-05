"""Step 4: Generate reference images of new people based on the transformation theme"""

from pathlib import Path
from typing import List, Optional

from utils.logger import setup_logger
from utils.config import Config
from utils.falai_worker import FalAIWorker
from utils.openai_worker import OpenAIWorker
from utils.download_file import download_file
from utils.cache_manager import CacheManager
from schemas import Person

logger = setup_logger(__name__)


async def generate_reference_images(
    original_person_registry: List[Person],
    transformation_theme: str,
    work_dir: Path,
    config: Config,
    input_video_path: Optional[str] = None,
    cache_manager: Optional[CacheManager] = None,
) -> List[Person]:
    """
    Generate reference images for NEW people based on the transformation theme.
    It creates entirely new people matching the transformation theme while maintaining the same count.

    Args:
        original_person_registry: List of detected people from the person detection step (original people in video)
        transformation_theme: Transformation theme (e.g., "Black people", "Asian couple")
        work_dir: Working directory for outputs
        config: Pipeline configuration
        input_video_path: Optional path to input video for cache key generation
        cache_manager: Optional cache manager for caching results

    Returns:
        New person registry with transformed people descriptions and reference image paths
    """
    logger.info(f"Generating {len(original_person_registry)} new people for theme: {transformation_theme}")

    work_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first
    if cache_manager and input_video_path:
        cached_data = cache_manager.load("reference_images", input_video_path)
        if cached_data:
            logger.info("Using cached person detection results")
            return [Person(**item) for item in cached_data]

    # Generate new person descriptions using OpenAI based on the transformation theme
    openai_worker = OpenAIWorker.get_instance()
    new_person_registry = await openai_worker.generate_new_people_descriptions(
        original_person_registry,
        transformation_theme,
        config
    )

    if len(new_person_registry) != len(original_person_registry):
        logger.warning(
            f"Number of generated people ({len(new_person_registry)}) doesn't match "
            f"original count ({len(original_person_registry)}). Adjusting..."
        )
        # Ensure we have the same number of people
        if len(new_person_registry) < len(original_person_registry):
            # Duplicate the last person if needed
            while len(new_person_registry) < len(original_person_registry):
                new_person_registry.append(new_person_registry[-1].model_copy())
        else:
            # Truncate if too many
            new_person_registry = new_person_registry[:len(original_person_registry)]

    fal_client = FalAIWorker.get_instance()

    for idx, new_person in enumerate(new_person_registry):
        # Use the original person_id for consistency in mapping
        original_person_id = original_person_registry[idx].person_id
        new_person.person_id = original_person_id

        logger.info(f"Generating reference image for new person {original_person_id}")

        try:
            person_description = new_person.description
            logger.info(f"New person description: {person_description}")

            # Create a detailed prompt for generating a full-body reference image
            prompt_template = config.get_prompt("reference_generation")
            prompt = prompt_template.format(
                description=person_description,
                clothing=new_person.clothing
            )

            # Generate the reference image
            result = await fal_client.generate(
                model=config.reference_model,
                arguments={
                    "prompt": prompt,
                    "aspect_ratio": "9:16",
                    "num_images": 1,
                }
            )

            # Download the generated image
            reference_url = result["images"][0]["url"]
            reference_path = work_dir / f"{original_person_id}_new_reference.jpg"

            await download_file(reference_url, str(reference_path))

            logger.info(f"Generated reference image for new person {original_person_id}: {reference_path}")

        except Exception as e:
            logger.error(f"Failed to generate reference for new person {original_person_id}: {str(e)}")
            raise

    logger.info(f"Successfully generated {len(new_person_registry)} new reference images")

    # Save to cache
    if cache_manager and input_video_path:
        cache_data = [person.model_dump(mode='json') for person in new_person_registry]
        cache_manager.save("reference_images", input_video_path, cache_data)

    return new_person_registry
