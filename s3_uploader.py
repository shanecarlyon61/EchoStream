
import boto3
from botocore.exceptions import ClientError
from typing import Optional, Dict
from datetime import datetime
import time

s3_client: Optional[boto3.client] = None

s3_bucket: Optional[str] = None
s3_region: str = "us-east-1"
s3_credentials: Optional[Dict[str, str]] = None

def init_s3(credentials: Dict[str, str], bucket: str, region: str = "us-east-1") -> bool:

    global s3_client, s3_bucket, s3_region, s3_credentials

    try:
        s3_bucket = bucket
        s3_region = region
        s3_credentials = credentials

        s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=credentials.get('aws_access_key_id'),
            aws_secret_access_key=credentials.get('aws_secret_access_key')
        )

        s3_client.head_bucket(Bucket=bucket)

        print(f"[S3] S3 client initialized (bucket: {bucket}, region: {region})")
        return True

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == '404':
            print(f"[S3] ERROR: S3 bucket {bucket} not found")
        else:
            print(f"[S3] ERROR: Failed to initialize S3: {e}")
        s3_client = None
        return False
    except Exception as e:
        print(f"[S3] ERROR: Exception initializing S3: {e}")
        s3_client = None
        return False

def upload_audio(audio_data: bytes, filename: Optional[str] = None) -> Optional[str]:

    global s3_client, s3_bucket

    if s3_client is None or s3_bucket is None:
        print("[S3] ERROR: S3 client not initialized")
        return None

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio/echostream_{timestamp}.opus"

    try:

        s3_client.put_object(
            Bucket=s3_bucket,
            Key=filename,
            Body=audio_data,
            ContentType='audio/opus'
        )

        print(f"[S3] Audio uploaded to S3: s3://{s3_bucket}/{filename} ({len(audio_data)} bytes)")
        return filename

    except ClientError as e:
        print(f"[S3] ERROR: Failed to upload audio to S3: {e}")
        return None
    except Exception as e:
        print(f"[S3] ERROR: Exception uploading audio: {e}")
        return None

def get_upload_url(filename: str, expiration: int = 3600) -> Optional[str]:

    global s3_client, s3_bucket

    if s3_client is None or s3_bucket is None:
        print("[S3] ERROR: S3 client not initialized")
        return None

    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': s3_bucket, 'Key': filename},
            ExpiresIn=expiration
        )

        print(f"[S3] Generated presigned URL for {filename} (expires in {expiration}s)")
        return url

    except ClientError as e:
        print(f"[S3] ERROR: Failed to generate presigned URL: {e}")
        return None
    except Exception as e:
        print(f"[S3] ERROR: Exception generating presigned URL: {e}")
        return None

def is_s3_initialized() -> bool:

    return s3_client is not None and s3_bucket is not None

