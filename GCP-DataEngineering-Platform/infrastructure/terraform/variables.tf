variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for regional resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "bigquery_dataset_location" {
  description = "Location for BigQuery datasets"
  type        = string
  default     = "US"
}

variable "storage_bucket_location" {
  description = "Location for Cloud Storage buckets"
  type        = string
  default     = "US"
}

variable "dataflow_max_workers" {
  description = "Maximum number of Dataflow workers"
  type        = number
  default     = 10
}

variable "dataproc_worker_count" {
  description = "Number of Dataproc worker nodes"
  type        = number
  default     = 2
}

variable "dataproc_preemptible_count" {
  description = "Number of preemptible Dataproc workers"
  type        = number
  default     = 4
}

variable "enable_private_ip" {
  description = "Enable private IP for Dataproc and Composer"
  type        = bool
  default     = true
}

variable "composer_node_count" {
  description = "Number of nodes in Composer environment"
  type        = number
  default     = 3
}

variable "alert_email" {
  description = "Email address for alerting"
  type        = string
  default     = ""
}

variable "budget_amount" {
  description = "Monthly budget amount in USD"
  type        = number
  default     = 1000
}
