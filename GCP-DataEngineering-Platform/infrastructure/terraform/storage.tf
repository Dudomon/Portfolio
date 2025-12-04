resource "google_storage_bucket" "data_lake" {
  name          = "${var.project_id}-data-lake"
  location      = var.storage_bucket_location
  force_destroy = false

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  labels = local.labels
}

resource "google_storage_bucket" "dataflow_temp" {
  name          = "${var.project_id}-dataflow-temp"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }

  labels = local.labels
}

resource "google_storage_bucket" "dataflow_staging" {
  name          = "${var.project_id}-dataflow-staging"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }

  labels = local.labels
}

resource "google_storage_bucket" "dataproc_staging" {
  name          = "${var.project_id}-dataproc-staging"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }

  labels = local.labels
}

resource "google_storage_bucket" "pipeline_artifacts" {
  name          = "${var.project_id}-pipeline-artifacts"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true
  versioning {
    enabled = true
  }

  labels = local.labels
}

resource "google_storage_bucket_iam_member" "dataflow_temp_access" {
  bucket = google_storage_bucket.dataflow_temp.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_storage_bucket_iam_member" "dataflow_staging_access" {
  bucket = google_storage_bucket.dataflow_staging.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_storage_bucket_iam_member" "dataproc_staging_access" {
  bucket = google_storage_bucket.dataproc_staging.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.dataproc_sa.email}"
}

resource "google_storage_bucket_iam_member" "data_lake_dataflow" {
  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_storage_bucket_iam_member" "data_lake_dataproc" {
  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.dataproc_sa.email}"
}
