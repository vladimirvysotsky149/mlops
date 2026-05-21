import boto3

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
)

bucket_name = 'models'

try:
    s3.create_bucket(Bucket=bucket_name)
except:
    pass

s3.upload_file(
    'artifacts/model_dvc.pth',
    bucket_name,
    'model_dvc.pth'
)

print('Model uploaded to MinIO')