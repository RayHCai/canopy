variable "region" {
  description = "AWS region for the bucket; becomes the API's S3_REGION."
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Globally unique bucket name; becomes the API's S3_BUCKET."
  type        = string
}

variable "cors_allowed_origins" {
  description = <<-EOT
    Origins allowed to PUT/GET objects via presigned URLs. The viewer page
    uploads photos/video straight to the bucket from the browser, so its
    origin (and the dashboard's) must be listed or the preflight fails.
  EOT
  type        = list(string)
  default     = ["http://localhost:3000", "http://localhost:4000"]
}
