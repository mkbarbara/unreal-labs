import base64
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from utils.logger import setup_logger
from utils.config import Config
from schemas import Person

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
        image_path: Path,
        config: Config,
    ) -> List[Person]:
        """
        Analyze a frame to detect and describe people

        Args:
            image_path: Path to the frame image
            config: Pipeline configuration

        Returns:
            A list of Person objects describing detected individuals.
        """
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            prompt = config.get_prompt("analyse_frame_for_people")

            response = await self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            raw_content = response.choices[0].message.content
            logger.info(f"Response: {raw_content}")

            # Remove Markdown code fences if they exist
            if raw_content.startswith("```"):
                raw_content = raw_content.strip().lstrip("`")
                if raw_content.startswith("json"):
                    raw_content = raw_content[len("json"):].lstrip()

                if raw_content.endswith("```"):
                    raw_content = raw_content[: -3].strip()

            result = json.loads(raw_content)
            logger.info(f"Detected {len(result["people"])} people in the frame.")

            people = [Person(**p) for p in result["people"]]
            return people

        except Exception as e:
            logger.error(f"Failed to analyze frame: {str(e)}")
            return []

    async def consolidate_person_descriptions(
        self,
        frame_analyses: List[Dict],
        config: Config,
    ) -> List[Person]:
        """
        Consolidate person descriptions across frames to create consistent identities

        Args:
            frame_analyses: List of frame analysis results
            config: Pipeline configuration

        Returns:
            List of consolidated person profiles
        """
        logger.info("Consolidating person descriptions across frames")

        try:
            prompt_template = config.get_prompt("persons_description")
            prompt = prompt_template.format(frame_analyses=frame_analyses)

            response = await self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )

            raw_content = response.choices[0].message.content

            # Remove Markdown code fences if they exist
            if raw_content.startswith("```"):
                raw_content = raw_content.strip().lstrip("`")
                if raw_content.startswith("json"):
                    raw_content = raw_content[len("json"):].lstrip()

                if raw_content.endswith("```"):
                    raw_content = raw_content[: -3].strip()

            result = json.loads(raw_content)
            logger.info(f"Consolidated into {len(result['people'])} unique individuals")

            people = [Person(**p) for p in result["people"]]
            return people

        except Exception as e:
            logger.error(f"Failed to consolidate person descriptions: {str(e)}")
            raise

    async def generate_new_people_descriptions(
        self,
        original_person_registry: List[Person],
        transformation_theme: str,
        config: Config,
    ) -> List[Person]:
        """
        Generate new person descriptions based on the transformation theme

        Args:
            original_person_registry: List of original people detected in video
            transformation_theme: Transformation theme (e.g., "Black people", "Asian couple")
            config: Pipeline configuration

        Returns:
            List of new person descriptions matching the theme
        """
        logger.info(f"Generating new people descriptions for theme: {transformation_theme}")

        try:
            prompt_template = config.get_prompt("generate_new_people")
            prompt = prompt_template.format(
                num_people=len(original_person_registry),
                transformation_theme=transformation_theme,
                original_people = json.dumps([p.model_dump() for p in original_person_registry], indent=2)
            )

            response = await self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )

            raw_content = response.choices[0].message.content

            # Remove Markdown code fences if they exist
            if raw_content.startswith("```"):
                raw_content = raw_content.strip().lstrip("`")
                if raw_content.startswith("json"):
                    raw_content = raw_content[len("json"):].lstrip()

                if raw_content.endswith("```"):
                    raw_content = raw_content[: -3].strip()

            result = json.loads(raw_content)
            new_people: List = result.get("people", [])

            logger.info(f"Generated {len(new_people)} new person descriptions")
            people = [Person(**p) for p in new_people]
            return people

        except Exception as e:
            logger.error(f"Failed to generate new people descriptions: {str(e)}")
            raise

    async def generate_transformation_prompt(
        self,
        people_in_frame: List[Person],
        new_people_in_frame: List[Person],
        config: Config,
    ) -> str:
        """
        Generate transformation prompt for a specific frame

        Args:
            people_in_frame: List of people detected in this specific frame
            new_people_in_frame: List of new people to be placed in this specific frame
            config: Pipeline configuration

        Returns:
            Transformation prompt for the image editing model
        """
        try:
            prompt_template = config.get_prompt("image_transformation")
            prompt = prompt_template.format(
                people_in_frame=json.dumps(
                    [person.model_dump(mode='json') for person in people_in_frame], indent=2
                ) if people_in_frame else "No people visible in this frame",
                # TODO: fix reference people may be empty
                reference_people=json.dumps(
                    [new_person.model_dump(mode='json') for new_person in new_people_in_frame], indent=2
                ),
            )

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )

            generated_prompt = response.choices[0].message.content
            logger.info(f"Generated transformation prompt: {generated_prompt[:100]}...")

            return generated_prompt

        except Exception as e:
            logger.error(f"Failed to generate transformation prompt: {str(e)}")
            raise