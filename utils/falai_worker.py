"""Wrapper client for fal.ai API to reduce code duplication"""

import asyncio
import fal_client
import os
from typing import Dict, Any, Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


class FalAIWorker:
    _instance: Optional["FalAIWorker"] = None
    _initialized: bool = False

    def __init__(self, max_attempts: int = 60, poll_interval: int = 5):
        # Only initialize once
        if self._initialized:
            return

        fal_key = os.getenv("FAL_KEY")
        if not fal_key:
            raise ValueError("FAL_KEY environment variable is not set")

        self.max_attempts = max_attempts
        self.poll_interval = poll_interval

        self._initialized = True
        logger.info("FalAIWorker initialized successfully")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls, max_attempts: int = 60, poll_interval: int = 5) -> "FalAIWorker":
        if cls._instance is None:
            cls._instance = FalAIWorker(max_attempts, poll_interval)
        return cls._instance

    @staticmethod
    async def upload_file(file_path: str) -> str:
        """
        Upload a file to fal.ai storage

        Args:
            file_path: Path to file to upload

        Returns:
            URL of the uploaded file
        """
        logger.debug(f"Uploading file: {file_path}")
        url = await fal_client.upload_file_async(file_path)
        logger.debug(f"File uploaded: {url}")
        return url

    @staticmethod
    async def _submit_request(model: str, arguments: Dict[str, Any]) -> str:
        """
        Submit a request to fal.ai model

        Args:
            model: Model identifier (e.g., "fal-ai/flux-lora", "fal-ai/veo")
            arguments: Model arguments (prompt, image_urls, etc.)

        Returns:
            Request ID for polling
        """
        logger.info(f"Submitting request to model: {model}")
        handler = await fal_client.submit_async(model, arguments=arguments)
        request_id = handler.request_id
        logger.info(f"Request submitted with ID: {request_id}")
        return request_id

    async def _poll_request(self, model: str, request_id: str) -> None:
        """
        Poll request status until completion

        Args:
            model: Model identifier
            request_id: Request ID to poll

        Raises:
            TimeoutError: If max attempts exceeded
            RuntimeError: If request failed
        """
        attempt = 0

        while attempt < self.max_attempts:
            status = await fal_client.status_async(
                model,
                request_id,
                with_logs=True
            )

            status_type = type(status).__name__
            logger.debug(f"Status for request {request_id}: {status_type} (attempt {attempt + 1}/{self.max_attempts})")

            if status_type == "Completed":
                logger.info(f"Request {request_id} completed successfully")
                return
            elif status_type == "Failed":
                error_msg = status.get('error', 'Unknown error') if isinstance(status, dict) else 'Unknown error'
                raise RuntimeError(f"Request {request_id} failed: {error_msg}")
            elif status_type in ["InProgress", "Queued"]:
                logger.debug(f"Request {request_id} status: {status_type}")

            await asyncio.sleep(self.poll_interval)
            attempt += 1

        raise TimeoutError(
            f"Request {request_id} timed out after {self.max_attempts * self.poll_interval} seconds"
        )

    @staticmethod
    async def _get_result(model: str, request_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed request

        Args:
            model: Model identifier
            request_id: Request ID

        Returns:
            Result dictionary from fal.ai
        """
        logger.info(f"Getting result for request {request_id}")
        result = await fal_client.result_async(model, request_id)
        logger.info(f"Successfully retrieved result for request {request_id}")
        return result

    async def generate(self, model: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content using fal.ai model (submits, polls, and retrieves result)

        Args:
            model: Model identifier (e.g., "fal-ai/flux-lora", "fal-ai/veo")
            arguments: Model arguments (prompt, image_urls, etc.)

        Returns:
            Result dictionary from fal.ai

        Raises:
            TimeoutError: If max attempts exceeded
            RuntimeError: If request failed
        """
        # Submit request
        request_id = await self._submit_request(model, arguments)

        # Poll until complete
        await self._poll_request(model, request_id)

        # Get and return result
        return await self._get_result(model, request_id)
