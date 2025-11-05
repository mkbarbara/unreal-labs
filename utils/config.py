from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Working directory
    work_dir: str = "work"

    # Model configurations
    reference_model: str = "fal-ai/nano-banana"
    img2img_model: str = "fal-ai/nano-banana/edit"
    video_model: str = "fal-ai/veo3.1/first-last-frame-to-video"

    # Video segmentation settings
    max_clip_duration: float = 8.0
    scene_threshold: float = 0.3  # For scene detection

    # Frame extraction settings
    frame_interval: int = 8
    frame_format: str = "jpg"
    frame_quality: int = 95

    # Video generation settings
    video_fps: int = 30
    video_resolution: tuple = (1080, 1920)  # Width x Height

    # Prompts directory
    prompts_dir: str = "prompts"

    def get_prompt(self, prompt_name: str) -> str:
        """Load a prompt template from the prompts directory"""
        prompt_path = Path(self.prompts_dir) / f"{prompt_name}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text()
