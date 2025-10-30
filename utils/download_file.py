import aiohttp
from utils.logger import setup_logger

logger = setup_logger(__name__)


async def download_file(url: str, output_path: str) -> str:
    """
    Download a file from URL to the local path

    Args:
        url: URL to download from
        output_path: Local path to save the file

    Returns:
        Path to the downloaded file

    Raises:
        RuntimeError: If download fails
    """
    logger.debug(f"Downloading file from {url} to {output_path}")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                with open(output_path, 'wb') as f:
                    f.write(await resp.read())
                logger.debug(f"File downloaded successfully: {output_path}")
                return output_path
            else:
                raise RuntimeError(f"Failed to download file: HTTP {resp.status}")