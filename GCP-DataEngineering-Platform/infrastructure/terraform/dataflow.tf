resource "google_dataflow_flex_template_job" "streaming_pipeline" {
  count    = var.environment == "production" ? 1 : 0
  provider = google

  name = "streaming-user-events-pipeline"

  container_spec_gcs_path = "gs://${google_storage_bucket.pipeline_artifacts.name}/templates/streaming-pipeline-template.json"

  parameters = {
    project              = var.project_id
    region               = var.region
    subscription         = google_pubsub_subscription.user_events_dataflow.id
    output_table         = "${var.project_id}:${google_bigquery_dataset.raw_data.dataset_id}.${google_bigquery_table.user_events_raw.table_id}"
    temp_location        = "gs://${google_storage_bucket.dataflow_temp.name}/temp"
    staging_location     = "gs://${google_storage_bucket.dataflow_staging.name}/staging"
    max_num_workers      = var.dataflow_max_workers
    autoscaling_algorithm = "THROUGHPUT_BASED"
  }

  region                = var.region
  service_account_email = google_service_account.dataflow_sa.email

  labels = local.labels

  lifecycle {
    ignore_changes = [
      parameters["job_name"],
      parameters["update_timestamp"]
    ]
  }
}

resource "google_dataflow_job" "batch_pipeline" {
  count = 0

  name              = "batch-transaction-processor"
  template_gcs_path = "gs://${google_storage_bucket.pipeline_artifacts.name}/templates/batch-pipeline-template.json"

  parameters = {
    project          = var.project_id
    input_path       = "gs://${google_storage_bucket.data_lake.name}/transactions/*.parquet"
    output_table     = "${var.project_id}:${google_bigquery_dataset.raw_data.dataset_id}.${google_bigquery_table.transactions_raw.table_id}"
    temp_location    = "gs://${google_storage_bucket.dataflow_temp.name}/temp"
    staging_location = "gs://${google_storage_bucket.dataflow_staging.name}/staging"
  }

  region                = var.region
  service_account_email = google_service_account.dataflow_sa.email

  labels = local.labels
}
