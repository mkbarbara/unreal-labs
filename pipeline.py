import asyncio
from pathlib import Path

from steps.split_video import split_video_into_intervals
from steps.text_removal import remove_text_from_intervals
from steps.person_detection import detect_and_describe_people
from steps.reference_generation import generate_reference_images
from steps.frame_editing import edit_frames
from steps.video_generation import generate_video_intervals
from steps.reassembly import reassemble_video
from steps.extract_text_layer import extract_text_layer
from steps.add_text_layer import add_text_layer
from utils.config import Config
from utils.cache_manager import CacheManager
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VideoLocalizationPipeline:
    """Main pipeline orchestrator for video localization"""

    def __init__(self, config: Config):
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manager = CacheManager(self.work_dir / "cache")

    async def run(
        self,
        input_video_path: str,
        transformation_theme: str,
    ) -> Path:
        """
        Run the complete video localization pipeline

        Args:
            input_video_path: Path to source video
            transformation_theme: Overall transformation theme (e.g., "Black people", "Asian people", "Elderly people")

        Returns:
            Path to the generated video
        """
        logger.info("Starting video localization pipeline")
        logger.info(f"Input video: {input_video_path}")
        logger.info(f"Transformation theme: {transformation_theme}")

        try:
            # Step 0: Extract text layers from video
            logger.info("Step 0: Extracting text layers from video")
            extract_text_layer(
                work_dir=self.work_dir / "extracted_text_layer",
                input_video_path=input_video_path,
            )

            # Step 1: Splits video into fixed intervals
            video_intervals = await split_video_into_intervals(
                input_video_path,
                work_dir=self.work_dir / "extracted_frames",
                audio_dir=self.work_dir / "extracted_audios",
                interval=self.config.frame_interval,
                cache_manager=self.cache_manager,
            )
            logger.info(f"Extracted {len(video_intervals)} video intervals")

            # Step 2: Remove text from all extracted frames in video intervals for better performance
            cleaned_video_intervals = await remove_text_from_intervals(
                video_intervals,
                work_dir=self.work_dir / "cleaned_frames",
                config=self.config,
                input_video_path=input_video_path,
                cache_manager=self.cache_manager,
            )
            logger.info(f"Cleaned {len(cleaned_video_intervals)} video intervals")

            # Step 3: Detect and describe people using cleaned frames
            original_person_registry = await detect_and_describe_people(
                cleaned_video_intervals,
                config=self.config,
                input_video_path=input_video_path,
                cache_manager=self.cache_manager,
            )
            logger.info(f"Person registry: {original_person_registry}")

            # Step 4: Generate reference images of new people based on the transformation theme
            new_person_registry = await generate_reference_images(
                original_person_registry,
                transformation_theme,
                work_dir=self.work_dir / "reference_images",
                config=self.config,
                input_video_path=input_video_path,
                cache_manager=self.cache_manager,
            )
            logger.info(f"Person registry: {new_person_registry}")

            # Step 5: Edit cleaned video intervals with reference images
            edited_intervals = await edit_frames(
                cleaned_video_intervals,
                new_person_registry,
                work_dir=self.work_dir / "edited_frames",
                config=self.config,
                input_video_path=input_video_path,
                cache_manager=self.cache_manager,
            )
            logger.info(f"Edited {len(edited_intervals)} intervals")

            # Step 6: Generate video clips
            logger.info("Step 6: Generating new video intervals with Veo3.1")
            generated_intervals = await generate_video_intervals(
                edited_intervals,
                work_dir=self.work_dir / "videos",
                config=self.config
            )
            logger.info(f"Generated {len(generated_intervals)} video intervals")

            # Step 7: Reassemble the video
            logger.info("Step 7: Reassembling video")
            reassembled_video = await reassemble_video(
                generated_intervals,
                self.work_dir / "reassembled_video.mp4",
            )
            logger.info(f"Video reassembled: {reassembled_video}")

            # Step 8: Add the extracted text layer to the reassembled video
            logger.info("Step 8: Adding extracted text layer to the reassembled video")
            final_video = add_text_layer(
                video_path=reassembled_video,
                text_layer_path=self.work_dir / "extracted_text_layer" / "text_rgba.png",
                output_path=self.work_dir / "final_video.mp4",
            )
            logger.info(f"Final video with text layer: {final_video}")

            return final_video

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


async def main():
    """Example usage"""
    config = Config()

    pipeline = VideoLocalizationPipeline(config)

    # Example demographic description
    transformation_theme = "Black people"

    result = await pipeline.run(
        input_video_path="assets/video.mp4",
        transformation_theme=transformation_theme,
    )

    print(f"Video localization complete: {result}")


if __name__ == "__main__":
    asyncio.run(main())
