# Module: s3
# License: MIT (ARVTON project)
# Description: Stubbed S3 storage upload for future cloud deployment.
# Platform: Cloud
# Dependencies: boto3 (optional)

"""
S3 Storage — Stubbed implementation for cloud deployment.
Replace stubs with actual boto3 calls when deploying to AWS.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("arvton.storage.s3")

# Configuration from environment
S3_BUCKET = os.environ.get("ARVTON_S3_BUCKET", "")
S3_REGION = os.environ.get("ARVTON_S3_REGION", "us-east-1")
S3_PREFIX = os.environ.get("ARVTON_S3_PREFIX", "arvton-outputs/")
CDN_BASE_URL = os.environ.get("ARVTON_CDN_URL", "")


def is_s3_configured() -> bool:
    """Check if S3 credentials and bucket are configured."""
    return bool(S3_BUCKET) and bool(os.environ.get("AWS_ACCESS_KEY_ID", ""))


def upload_to_s3(
    local_path: str,
    job_id: str,
    filename: Optional[str] = None,
) -> Optional[str]:
    """
    Upload a file to S3 and return the public URL.

    Args:
        local_path: Path to the local file.
        job_id: Job identifier for S3 key prefix.
        filename: Override filename. Defaults to the local filename.

    Returns:
        Public URL to the uploaded file, or None on failure.
    """
    if not is_s3_configured():
        logger.warning("S3 not configured. Skipping upload.")
        return None

    if filename is None:
        filename = Path(local_path).name

    s3_key = f"{S3_PREFIX}{job_id}/{filename}"

    try:
        import boto3

        s3_client = boto3.client("s3", region_name=S3_REGION)

        # Determine content type
        content_type = "application/octet-stream"
        ext = Path(filename).suffix.lower()
        content_types = {
            ".glb": "model/gltf-binary",
            ".usdz": "model/vnd.usdz+zip",
            ".obj": "model/obj",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }
        content_type = content_types.get(ext, content_type)

        s3_client.upload_file(
            local_path,
            S3_BUCKET,
            s3_key,
            ExtraArgs={
                "ContentType": content_type,
                "CacheControl": "public, max-age=31536000",
            },
        )

        # Build URL
        if CDN_BASE_URL:
            url = f"{CDN_BASE_URL.rstrip('/')}/{s3_key}"
        else:
            url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"

        logger.info("Uploaded to S3: %s", url)
        return url

    except ImportError:
        logger.error("boto3 not installed. Run: pip install boto3")
        return None

    except Exception as e:
        logger.error("S3 upload failed: %s", str(e))
        return None


def delete_from_s3(job_id: str) -> bool:
    """
    Delete all objects for a job from S3.

    Args:
        job_id: Job identifier.

    Returns:
        True if deletion was successful.
    """
    if not is_s3_configured():
        return False

    try:
        import boto3

        s3_client = boto3.client("s3", region_name=S3_REGION)
        prefix = f"{S3_PREFIX}{job_id}/"

        # List and delete objects
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        objects = response.get("Contents", [])

        if objects:
            s3_client.delete_objects(
                Bucket=S3_BUCKET,
                Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
            )
            logger.info("Deleted %d S3 objects for job %s", len(objects), job_id[:8])

        return True

    except Exception as e:
        logger.error("S3 delete failed: %s", str(e))
        return False
