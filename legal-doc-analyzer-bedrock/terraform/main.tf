terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "legal-doc-analyzer-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "LegalDocAnalyzer"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "documents" {
  bucket = "${var.project_name}-documents-${var.environment}"

  tags = {
    Name = "Legal Documents Storage"
  }
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.documents.arn
    }
  }
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket = aws_s3_bucket.documents.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_kms_key" "documents" {
  description             = "KMS key for legal documents encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name = "Legal Documents Encryption Key"
  }
}

resource "aws_kms_alias" "documents" {
  name          = "alias/legal-docs-encryption"
  target_key_id = aws_kms_key.documents.key_id
}

resource "aws_sqs_queue" "processing_dlq" {
  name                       = "${var.project_name}-processing-dlq-${var.environment}"
  message_retention_seconds  = 1209600
  visibility_timeout_seconds = 30

  tags = {
    Name = "Document Processing DLQ"
  }
}

resource "aws_sqs_queue" "processing" {
  name                       = "${var.project_name}-processing-${var.environment}"
  visibility_timeout_seconds = 300
  message_retention_seconds  = 86400
  delay_seconds              = 0
  receive_wait_time_seconds  = 20

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.processing_dlq.arn
    maxReceiveCount     = 3
  })

  tags = {
    Name = "Document Processing Queue"
  }
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/${var.project_name}/api"
  retention_in_days = 30

  tags = {
    Name = "API Logs"
  }
}

resource "aws_cloudwatch_log_stream" "api" {
  name           = "api-logs"
  log_group_name = aws_cloudwatch_log_group.api.name
}

resource "aws_iam_role" "api_execution" {
  name = "${var.project_name}-api-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "api_bedrock" {
  name = "bedrock-access"
  role = aws_iam_role.api_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
      }
    ]
  })
}

resource "aws_iam_role_policy" "api_s3" {
  name = "s3-access"
  role = aws_iam_role.api_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey"
        ]
        Resource = aws_kms_key.documents.arn
      }
    ]
  })
}

resource "aws_iam_role_policy" "api_sqs" {
  name = "sqs-access"
  role = aws_iam_role.api_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:ChangeMessageVisibility"
        ]
        Resource = [
          aws_sqs_queue.processing.arn,
          aws_sqs_queue.processing_dlq.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "api_cloudwatch" {
  name = "cloudwatch-access"
  role = aws_iam_role.api_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "${aws_cloudwatch_log_group.api.arn}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "LegalDocAnalyzer"
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_metric_alarm" "bedrock_errors" {
  alarm_name          = "${var.project_name}-bedrock-errors-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "bedrock.request.error"
  namespace           = "LegalDocAnalyzer"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "This metric monitors Bedrock API errors"
  treat_missing_data  = "notBreaching"

  alarm_actions = []
}

resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "${var.project_name}-high-latency-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "api.request.duration"
  namespace           = "LegalDocAnalyzer"
  period              = "300"
  statistic           = "Average"
  threshold           = "3000"
  alarm_description   = "This metric monitors API latency"
  treat_missing_data  = "notBreaching"

  alarm_actions = []
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-${var.environment}"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["LegalDocAnalyzer", "api.request.count", { stat = "Sum" }],
            [".", "api.request.duration", { stat = "Average" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "API Metrics"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["LegalDocAnalyzer", "bedrock.request.success", { stat = "Sum" }],
            [".", "bedrock.request.error", { stat = "Sum" }]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "Bedrock Metrics"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["LegalDocAnalyzer", "bedrock.cost", { stat = "Sum" }]
          ]
          period = 3600
          stat   = "Sum"
          region = var.aws_region
          title  = "Bedrock Cost (USD)"
        }
      }
    ]
  })
}
