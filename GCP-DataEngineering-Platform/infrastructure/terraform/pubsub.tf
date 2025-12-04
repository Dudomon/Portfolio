resource "google_pubsub_topic" "user_events" {
  name = "user-events"

  message_retention_duration = "86400s"

  labels = local.labels
}

resource "google_pubsub_topic" "transactions" {
  name = "transactions"

  message_retention_duration = "86400s"

  labels = local.labels
}

resource "google_pubsub_topic" "pipeline_errors" {
  name = "pipeline-errors"

  message_retention_duration = "604800s"

  labels = local.labels
}

resource "google_pubsub_subscription" "user_events_dataflow" {
  name  = "user-events-dataflow"
  topic = google_pubsub_topic.user_events.name

  ack_deadline_seconds = 300

  expiration_policy {
    ttl = ""
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.pipeline_errors.id
    max_delivery_attempts = 5
  }

  labels = local.labels
}

resource "google_pubsub_subscription" "transactions_dataflow" {
  name  = "transactions-dataflow"
  topic = google_pubsub_topic.transactions.name

  ack_deadline_seconds = 300

  expiration_policy {
    ttl = ""
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.pipeline_errors.id
    max_delivery_attempts = 5
  }

  labels = local.labels
}

resource "google_pubsub_subscription" "pipeline_errors_monitoring" {
  name  = "pipeline-errors-monitoring"
  topic = google_pubsub_topic.pipeline_errors.name

  ack_deadline_seconds = 60

  expiration_policy {
    ttl = ""
  }

  labels = local.labels
}

resource "google_pubsub_topic_iam_member" "dataflow_publisher" {
  topic  = google_pubsub_topic.user_events.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_pubsub_subscription_iam_member" "dataflow_subscriber" {
  subscription = google_pubsub_subscription.user_events_dataflow.name
  role         = "roles/pubsub.subscriber"
  member       = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_pubsub_subscription_iam_member" "transactions_subscriber" {
  subscription = google_pubsub_subscription.transactions_dataflow.name
  role         = "roles/pubsub.subscriber"
  member       = "serviceAccount:${google_service_account.dataflow_sa.email}"
}
