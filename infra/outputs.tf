output "s3_bucket" {
  description = "S3_BUCKET for the API."
  value       = aws_s3_bucket.media.bucket
}

output "s3_region" {
  description = "S3_REGION for the API."
  value       = var.region
}

output "s3_access_key_id" {
  description = "S3_ACCESS_KEY_ID for the API."
  value       = aws_iam_access_key.api.id
}

output "s3_secret_access_key" {
  description = "S3_SECRET_ACCESS_KEY for the API. Read with `terraform output -raw s3_secret_access_key`."
  value       = aws_iam_access_key.api.secret
  sensitive   = true
}
