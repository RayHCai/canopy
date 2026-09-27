# Object store for the review API (src/api). The API presigns PUTs for the
# viewer's photo/video uploads and GETs for the dashboard (see src/api/src/s3.ts),
# so the bucket stays fully private: every access goes through a presigned URL.

resource "aws_s3_bucket" "media" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_public_access_block" "media" {
  bucket = aws_s3_bucket.media.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "media" {
  bucket = aws_s3_bucket.media.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "media" {
  bucket = aws_s3_bucket.media.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Browsers PUT directly to presigned URLs, which is a cross-origin request.
resource "aws_s3_bucket_cors_configuration" "media" {
  bucket = aws_s3_bucket.media.id

  cors_rule {
    allowed_methods = ["PUT", "GET", "HEAD"]
    allowed_origins = var.cors_allowed_origins
    allowed_headers = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3600
  }
}

# Uploads are presigned for 15 minutes; clear out multipart debris from
# abandoned ones rather than paying for it indefinitely.
resource "aws_s3_bucket_lifecycle_configuration" "media" {
  bucket = aws_s3_bucket.media.id

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# The API authenticates with static keys (S3_ACCESS_KEY_ID /
# S3_SECRET_ACCESS_KEY), so it gets a dedicated user that can only read and
# write objects in this bucket. A presigned URL carries its signer's
# permissions, so this also bounds what a leaked URL can do.
resource "aws_iam_user" "api" {
  name = "${var.bucket_name}-api"
}

data "aws_iam_policy_document" "api" {
  statement {
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.media.arn}/*"]
  }
}

resource "aws_iam_user_policy" "api" {
  name   = "s3-objects"
  user   = aws_iam_user.api.name
  policy = data.aws_iam_policy_document.api.json
}

resource "aws_iam_access_key" "api" {
  user = aws_iam_user.api.name
}
