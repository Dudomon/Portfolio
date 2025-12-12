output "s3_bucket_name" {
  description = "Name of the S3 bucket for documents"
  value       = aws_s3_bucket.documents.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.documents.arn
}

output "kms_key_id" {
  description = "ID of the KMS key for encryption"
  value       = aws_kms_key.documents.id
}

output "kms_key_alias" {
  description = "Alias of the KMS key"
  value       = aws_kms_alias.documents.name
}

output "sqs_queue_url" {
  description = "URL of the processing queue"
  value       = aws_sqs_queue.processing.url
}

output "sqs_dlq_url" {
  description = "URL of the dead letter queue"
  value       = aws_sqs_queue.processing_dlq.url
}

output "cloudwatch_log_group" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.api.name
}

output "iam_role_arn" {
  description = "ARN of the IAM role for API execution"
  value       = aws_iam_role.api_execution.arn
}

output "dashboard_url" {
  description = "URL to the CloudWatch dashboard"
  value       = "https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
}
