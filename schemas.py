from pydantic import BaseModel
from pathlib import Path


class VideoInterval(BaseModel):
    index: int
    start_frame_path: Path
    end_frame_path: Path
    start_time: float
    end_time: float
    duration: float
    fps: float


class Person(BaseModel):
    person_id: str
    gender: str
    age: str
    skin: str
    hair: str
    clothing: str

    @property
    def description(self) -> str:
        """Return a concise natural-language description of the person."""
        parts: list[str] = []

        if self.age:
            parts.append(self.age)

        if self.gender:
            parts.append(self.gender)

        if self.skin:
            parts.append(f"with {self.skin} skin tone")

        if self.hair:
            parts.append(f"with {self.hair}")

        description = " ".join(parts) if parts else "person"
        return description
