resource "google_service_account" "dataflow_sa" {
  account_id   = "dataflow-pipeline-sa"
  display_name = "Dataflow Pipeline Service Account"
  description  = "Service account for Dataflow pipeline execution"
}

resource "google_service_account" "dataproc_sa" {
  account_id   = "dataproc-cluster-sa"
  display_name = "Dataproc Cluster Service Account"
  description  = "Service account for Dataproc cluster operations"
}

resource "google_service_account" "composer_sa" {
  account_id   = "composer-environment-sa"
  display_name = "Cloud Composer Service Account"
  description  = "Service account for Cloud Composer environment"
}

resource "google_service_account" "api_sa" {
  account_id   = "api-server-sa"
  display_name = "API Server Service Account"
  description  = "Service account for API server operations"
}

resource "google_project_iam_member" "dataflow_worker" {
  project = var.project_id
  role    = "roles/dataflow.worker"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "dataflow_bigquery" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "dataflow_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "dataproc_worker" {
  project = var.project_id
  role    = "roles/dataproc.worker"
  member  = "serviceAccount:${google_service_account.dataproc_sa.email}"
}

resource "google_project_iam_member" "dataproc_bigquery" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.dataproc_sa.email}"
}

resource "google_project_iam_member" "composer_worker" {
  project = var.project_id
  role    = "roles/composer.worker"
  member  = "serviceAccount:${google_service_account.composer_sa.email}"
}

resource "google_project_iam_member" "composer_dataflow_developer" {
  project = var.project_id
  role    = "roles/dataflow.developer"
  member  = "serviceAccount:${google_service_account.composer_sa.email}"
}

resource "google_project_iam_member" "composer_dataproc_editor" {
  project = var.project_id
  role    = "roles/dataproc.editor"
  member  = "serviceAccount:${google_service_account.composer_sa.email}"
}

resource "google_project_iam_member" "api_bigquery_reader" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.api_sa.email}"
}

resource "google_project_iam_member" "api_bigquery_jobs" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.api_sa.email}"
}
