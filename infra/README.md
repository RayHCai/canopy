# infra

Terraform for the review API's AWS object store: a private S3 bucket
(encrypted, public access blocked, CORS for browser uploads via presigned
URLs) and a least-privilege IAM user whose keys the API signs with.

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars   # set bucket_name, origins
terraform init
terraform apply
```

Then point `src/api/.env` at it. Remove `S3_ENDPOINT` and set
`S3_FORCE_PATH_STYLE=false`:

```bash
S3_REGION=$(terraform output -raw s3_region)
S3_BUCKET=$(terraform output -raw s3_bucket)
S3_ACCESS_KEY_ID=$(terraform output -raw s3_access_key_id)
S3_SECRET_ACCESS_KEY=$(terraform output -raw s3_secret_access_key)
```

State is local and contains the secret key. Keep `terraform.tfstate` out of
git (the `.gitignore` here does that), and move to a remote backend before
anyone else applies.
