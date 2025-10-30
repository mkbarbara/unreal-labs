import asyncio
from pathlib import Path

from steps.person_detection import detect_and_describe_people
from steps.shot_segmentation import segment_video
from steps.frame_extraction import extract_conditioning_frames
from steps.frame_editing import edit_frames
from steps.video_generation import generate_video_clips
from steps.reassembly import reassemble_video
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VideoLocalizationPipeline:
    """Main pipeline orchestrator for video localization"""

    def __init__(self, config: Config):
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        input_video_path: str,
        transformation_theme: str,
        output_path: str
    ) -> str:
        """
        Run the complete video localization pipeline

        Args:
            input_video_path: Path to source video
            transformation_theme: Overall transformation theme (e.g., "Black people", "Asian people", "Elderly people")
            output_path: Path for final output video

        Returns:
            Path to the generated video
        """
        logger.info("Starting video localization pipeline")
        logger.info(f"Input video: {input_video_path}")
        logger.info(f"Transformation theme: {transformation_theme}")

        try:
            # Step 0: Detect and describe people in the video
            logger.info("Step 0: Detecting and describing people in video")
            person_registry = await detect_and_describe_people(
                input_video_path,
                work_dir=self.work_dir / "person_detection",
                config=self.config
            )
            logger.info(f"Detected {len(person_registry)} unique individuals")

            # Step 1: Segment video into clips
            logger.info("Step 1: Segmenting video into clips")
            clips = await segment_video(
                input_video_path,
                work_dir=self.work_dir / "clips",
                config=self.config
            )
            logger.info(f"Segmented into {len(clips)} clips")

            # Step 2: Extract conditioning frames
            logger.info("Step 2: Extracting conditioning frames")
            conditioning_frames = await extract_conditioning_frames(
                clips,
                work_dir=self.work_dir / "frames",
                config=self.config
            )
            logger.info(f"Extracted frames for {len(conditioning_frames)} clips")

            # Step 3: Edit demographic attributes with dynamic prompts
            logger.info("Step 3: Editing demographic attributes on frames")
            edited_frames = await edit_frames(
                conditioning_frames,
                person_registry,
                transformation_theme,
                work_dir=self.work_dir / "edited_frames",
                config=self.config
            )
            logger.info(f"Edited frames for {len(edited_frames)} clips")

            # Step 4: Generate video clips
            logger.info("Step 4: Generating video clips with Veo3.1")
            generated_clips = await generate_video_clips(
                edited_frames,
                work_dir=self.work_dir / "videos",
                config=self.config
            )
            logger.info(f"Generated {len(generated_clips)} video clips")

            # Step 5: Reassemble the final video
            logger.info("Step 5: Reassembling final video")
            final_video = await reassemble_video(
                generated_clips,
                output_path,
                config=self.config
            )
            logger.info(f"Final video created: {final_video}")

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
        output_path="output/localized_video.mp4"
    )

    print(f"Video localization complete: {result}")


if __name__ == "__main__":
    asyncio.run(main())
