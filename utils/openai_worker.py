import base64
import json
import os
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from utils.logger import setup_logger
from utils.config import Config

logger = setup_logger(__name__)


class OpenAIWorker:
    _instance: Optional["OpenAIWorker"] = None
    _initialized: bool = False

    def __init__(self, max_attempts: int = 3):
        # Only initialize once
        if self._initialized:
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = AsyncOpenAI(api_key=api_key)

        self.max_attempts = max_attempts
        self._initialized = True
        logger.info("OpenAIWorker initialized successfully")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls, max_attempts: int = 3) -> "OpenAIWorker":
        if cls._instance is None:
            cls._instance = OpenAIWorker(max_attempts)
        return cls._instance

    async def analyze_frame_for_people(
        self,
        image_path: str,
        frame_number: int,
        config: Config,
    ) -> Dict:
        """
        Analyze a frame to detect and describe people

        Args:
            image_path: Path to the frame image
            frame_number: Frame index in the video
            config: Pipeline configuration

        Returns:
            Dict with detected people and their descriptions
        """
        logger.info(f"Analyzing frame {frame_number} for people detection")

        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            prompt_template = config.get_prompt("analyse_frame_for_people")
            prompt = prompt_template.format(frame_number=frame_number)

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(f"Detected {len(result['people'])} people in frame {frame_number}")

            return result

        except Exception as e:
            logger.error(f"Failed to analyze frame {frame_number}: {str(e)}")
            return {"people": [], "frame_number": frame_number}

    async def consolidate_person_descriptions(
        self,
        frame_analyses: List[Dict]
    ) -> List[Dict]:
        """
        Consolidate person descriptions across frames to create consistent identities

        Args:
            frame_analyses: List of frame analysis results

        Returns:
            List of consolidated person profiles
        """
        logger.info("Consolidating person descriptions across frames")

        try:
            prompt = f"""You are analyzing a video and need to identify unique individuals across multiple frames.

Here is the detection data from {len(frame_analyses)} sampled frames:

{frame_analyses}

Your task:
1. Identify unique individuals that appear across frames (same person may have different temp_ids in different frames)
2. For each unique person, create a consolidated profile with their most consistent attributes
3. Assign each person a consistent ID (person_1, person_2, etc.)

Respond with valid JSON in this format:
{{
  "people": [
    {{
      "person_id": "person_1",
      "gender": "female",
      "age_group": "adult",
      "skin_tone": "medium",
      "hair": "consolidated hair description",
      "typical_clothing": "consolidated clothing description",
      "appearances": [1, 3, 5, 7]  // frame numbers where this person appears
    }}
  ]
}}"""

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )

            import json
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Consolidated into {len(result['people'])} unique individuals")

            return result["people"]

        except Exception as e:
            logger.error(f"Failed to consolidate person descriptions: {str(e)}")
            raise

    async def generate_transformation_prompt(
            self,
            people_in_frame: List[Dict],
            transformation_theme: str,
            person_registry: List[Dict]
    ) -> str:
        """
        Generate transformation prompt for a specific frame

        Args:
            people_in_frame: List of people detected in this specific frame
            transformation_theme: Overall transformation theme (e.g., "Black people")
            person_registry: Complete registry of all people in the video

        Returns:
            Transformation prompt for the image editing model
        """
        try:
            prompt = f"""Generate a detailed transformation prompt for an image editing AI model.

Context:
- Transformation theme: "{transformation_theme}"
- People in video: {len(person_registry)} unique individuals
- People visible in THIS frame: {len(people_in_frame)}

People registry (all individuals in video):
{person_registry}

People visible in THIS specific frame:
{people_in_frame if people_in_frame else "No people visible in this frame"}

Requirements:
1. If people ARE visible: Transform each person according to the theme while maintaining their clothing, pose, and other attributes
2. If NO people are visible: Return a prompt that preserves the scene as-is
3. Always maintain scene composition, lighting, background, and any text overlays
4. Be specific about each person's transformation based on their original attributes

Generate a single, detailed prompt suitable for an img2img model.
Respond with ONLY the prompt text, no JSON or extra formatting."""

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )

            generated_prompt = response.choices[0].message.content.strip()
            logger.info(f"Generated transformation prompt: {generated_prompt[:100]}...")

            return generated_prompt

        except Exception as e:
            logger.error(f"Failed to generate transformation prompt: {str(e)}")
            raise