resource "google_bigquery_dataset" "raw_data" {
  dataset_id                  = "raw_data"
  friendly_name               = "Raw Data Layer"
  description                 = "Unprocessed data from source systems"
  location                    = var.bigquery_dataset_location
  default_table_expiration_ms = null
  labels                      = local.labels

  access {
    role          = "OWNER"
    user_by_email = google_service_account.dataflow_sa.email
  }

  access {
    role          = "WRITER"
    user_by_email = google_service_account.dataproc_sa.email
  }
}

resource "google_bigquery_dataset" "staging_data" {
  dataset_id    = "staging_data"
  friendly_name = "Staging Data Layer"
  description   = "Cleaned and validated data ready for transformation"
  location      = var.bigquery_dataset_location
  labels        = local.labels

  access {
    role          = "OWNER"
    user_by_email = google_service_account.dataflow_sa.email
  }

  access {
    role          = "WRITER"
    user_by_email = google_service_account.dataproc_sa.email
  }
}

resource "google_bigquery_dataset" "analytics_data" {
  dataset_id    = "analytics_data"
  friendly_name = "Analytics Data Layer"
  description   = "Business-ready aggregations and metrics"
  location      = var.bigquery_dataset_location
  labels        = local.labels

  access {
    role          = "OWNER"
    user_by_email = google_service_account.dataflow_sa.email
  }

  access {
    role          = "READER"
    user_by_email = google_service_account.api_sa.email
  }
}

resource "google_bigquery_dataset" "data_quality" {
  dataset_id    = "data_quality"
  friendly_name = "Data Quality Metrics"
  description   = "Great Expectations validation results and quality metrics"
  location      = var.bigquery_dataset_location
  labels        = local.labels
}

resource "google_bigquery_table" "transactions_raw" {
  dataset_id = google_bigquery_dataset.raw_data.dataset_id
  table_id   = "transactions"

  time_partitioning {
    type  = "DAY"
    field = "ingestion_timestamp"
  }

  clustering = ["user_id", "transaction_type"]

  schema = jsonencode([
    {
      name        = "transaction_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unique transaction identifier"
    },
    {
      name        = "user_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "User identifier"
    },
    {
      name        = "transaction_type"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Type of transaction (purchase, refund, etc)"
    },
    {
      name        = "amount"
      type        = "NUMERIC"
      mode        = "REQUIRED"
      description = "Transaction amount"
    },
    {
      name        = "currency"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Currency code (USD, EUR, etc)"
    },
    {
      name        = "merchant_id"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Merchant identifier"
    },
    {
      name        = "metadata"
      type        = "JSON"
      mode        = "NULLABLE"
      description = "Additional transaction metadata"
    },
    {
      name        = "transaction_timestamp"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "When the transaction occurred"
    },
    {
      name        = "ingestion_timestamp"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "When the record was ingested into BigQuery"
    }
  ])

  labels = local.labels
}

resource "google_bigquery_table" "user_events_raw" {
  dataset_id = google_bigquery_dataset.raw_data.dataset_id
  table_id   = "user_events"

  time_partitioning {
    type  = "DAY"
    field = "event_timestamp"
  }

  clustering = ["event_type", "user_id"]

  schema = jsonencode([
    {
      name        = "event_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unique event identifier"
    },
    {
      name        = "user_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "User identifier"
    },
    {
      name        = "session_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Session identifier"
    },
    {
      name        = "event_type"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Type of event (page_view, click, etc)"
    },
    {
      name        = "event_properties"
      type        = "JSON"
      mode        = "NULLABLE"
      description = "Event-specific properties"
    },
    {
      name        = "device_type"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Device type (mobile, desktop, tablet)"
    },
    {
      name        = "user_agent"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Browser user agent string"
    },
    {
      name        = "ip_address"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "User IP address"
    },
    {
      name        = "event_timestamp"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "When the event occurred"
    }
  ])

  labels = local.labels
}

resource "google_bigquery_table" "data_quality_results" {
  dataset_id = google_bigquery_dataset.data_quality.dataset_id
  table_id   = "validation_results"

  time_partitioning {
    type  = "DAY"
    field = "validation_timestamp"
  }

  schema = jsonencode([
    {
      name        = "validation_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unique validation run identifier"
    },
    {
      name        = "expectation_suite_name"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Name of the Great Expectations suite"
    },
    {
      name        = "batch_identifier"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Identifier for the data batch validated"
    },
    {
      name        = "success"
      type        = "BOOLEAN"
      mode        = "REQUIRED"
      description = "Overall validation success status"
    },
    {
      name        = "validation_results"
      type        = "JSON"
      mode        = "REQUIRED"
      description = "Detailed validation results"
    },
    {
      name        = "validation_timestamp"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "When the validation was performed"
    }
  ])

  labels = local.labels
}

resource "google_bigquery_table" "pipeline_metrics" {
  dataset_id = google_bigquery_dataset.data_quality.dataset_id
  table_id   = "pipeline_metrics"

  time_partitioning {
    type  = "DAY"
    field = "metric_timestamp"
  }

  clustering = ["pipeline_name", "metric_type"]

  schema = jsonencode([
    {
      name        = "metric_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unique metric identifier"
    },
    {
      name        = "pipeline_name"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Name of the pipeline"
    },
    {
      name        = "metric_type"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Type of metric (latency, throughput, error_rate)"
    },
    {
      name        = "metric_value"
      type        = "NUMERIC"
      mode        = "REQUIRED"
      description = "Metric value"
    },
    {
      name        = "metric_unit"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unit of measurement"
    },
    {
      name        = "labels"
      type        = "JSON"
      mode        = "NULLABLE"
      description = "Additional metric labels"
    },
    {
      name        = "metric_timestamp"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "When the metric was recorded"
    }
  ])

  labels = local.labels
}
