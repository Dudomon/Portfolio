output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP region"
  value       = var.region
}

output "bigquery_datasets" {
  description = "BigQuery dataset IDs"
  value = {
    raw_data       = google_bigquery_dataset.raw_data.dataset_id
    staging_data   = google_bigquery_dataset.staging_data.dataset_id
    analytics_data = google_bigquery_dataset.analytics_data.dataset_id
    data_quality   = google_bigquery_dataset.data_quality.dataset_id
  }
}

output "storage_buckets" {
  description = "Cloud Storage bucket names"
  value = {
    data_lake          = google_storage_bucket.data_lake.name
    dataflow_temp      = google_storage_bucket.dataflow_temp.name
    dataflow_staging   = google_storage_bucket.dataflow_staging.name
    dataproc_staging   = google_storage_bucket.dataproc_staging.name
    pipeline_artifacts = google_storage_bucket.pipeline_artifacts.name
  }
}

output "pubsub_topics" {
  description = "Pub/Sub topic names"
  value = {
    user_events     = google_pubsub_topic.user_events.name
    transactions    = google_pubsub_topic.transactions.name
    pipeline_errors = google_pubsub_topic.pipeline_errors.name
  }
}

output "pubsub_subscriptions" {
  description = "Pub/Sub subscription names"
  value = {
    user_events_dataflow       = google_pubsub_subscription.user_events_dataflow.name
    transactions_dataflow      = google_pubsub_subscription.transactions_dataflow.name
    pipeline_errors_monitoring = google_pubsub_subscription.pipeline_errors_monitoring.name
  }
}

output "service_accounts" {
  description = "Service account emails"
  value = {
    dataflow_sa = google_service_account.dataflow_sa.email
    dataproc_sa = google_service_account.dataproc_sa.email
    composer_sa = google_service_account.composer_sa.email
    api_sa      = google_service_account.api_sa.email
  }
}

output "dataproc_cluster_name" {
  description = "Dataproc cluster name"
  value       = google_dataproc_cluster.data_processing.name
}

output "monitoring_dashboards" {
  description = "Cloud Monitoring dashboard URLs"
  value = {
    pipeline_health = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.pipeline_health.id}?project=${var.project_id}"
    data_quality    = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.data_quality.id}?project=${var.project_id}"
  }
}
