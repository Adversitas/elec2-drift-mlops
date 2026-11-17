# SageMaker real-time endpoint for the driftops container.
# Validated in CI with `terraform validate`; it has not been applied to a real AWS account.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "eu-west-3"
}

variable "project" {
  type    = string
  default = "driftops"
}

variable "image_tag" {
  type        = string
  description = "Tag of the image pushed to the ECR repository"
  default     = "latest"
}

variable "model_version" {
  type        = number
  description = "Registry version packaged as model.tar.gz under s3://<bucket>/models/v<N>/"
  default     = 1
}

variable "instance_type" {
  type    = string
  default = "ml.t2.medium"
}

resource "aws_ecr_repository" "inference" {
  name                 = "${var.project}-inference"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "${var.project}-artifacts-"
  force_destroy = false
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

data "aws_iam_policy_document" "assume_sagemaker" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "${var.project}-sagemaker"
  assume_role_policy = data.aws_iam_policy_document.assume_sagemaker.json
}

data "aws_iam_policy_document" "sagemaker_access" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
  }
  statement {
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "cloudwatch:PutMetricData",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "sagemaker" {
  role   = aws_iam_role.sagemaker.id
  policy = data.aws_iam_policy_document.sagemaker_access.json
}

resource "aws_sagemaker_model" "this" {
  name               = "${var.project}-v${var.model_version}"
  execution_role_arn = aws_iam_role.sagemaker.arn
  primary_container {
    image          = "${aws_ecr_repository.inference.repository_url}:${var.image_tag}"
    model_data_url = "s3://${aws_s3_bucket.artifacts.bucket}/models/v${var.model_version}/model.tar.gz"
  }
}

resource "aws_sagemaker_endpoint_configuration" "this" {
  name = "${var.project}-v${var.model_version}"
  production_variants {
    variant_name           = "champion"
    model_name             = aws_sagemaker_model.this.name
    initial_instance_count = 1
    instance_type          = var.instance_type
  }
  # Captured requests feed `driftops monitor` (PSI against the champion's reference sample).
  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = 100
    destination_s3_uri          = "s3://${aws_s3_bucket.artifacts.bucket}/capture"
    capture_options {
      capture_mode = "Input"
    }
    capture_options {
      capture_mode = "Output"
    }
  }
}

resource "aws_sagemaker_endpoint" "this" {
  name                 = var.project
  endpoint_config_name = aws_sagemaker_endpoint_configuration.this.name
}

output "endpoint_name" {
  value = aws_sagemaker_endpoint.this.name
}

output "ecr_repository_url" {
  value = aws_ecr_repository.inference.repository_url
}
