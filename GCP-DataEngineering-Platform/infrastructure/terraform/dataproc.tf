resource "google_dataproc_cluster" "data_processing" {
  name   = "data-processing-cluster"
  region = var.region

  cluster_config {
    staging_bucket = google_storage_bucket.dataproc_staging.name

    master_config {
      num_instances = 1
      machine_type  = "n1-standard-4"
      disk_config {
        boot_disk_type    = "pd-standard"
        boot_disk_size_gb = 100
      }
    }

    worker_config {
      num_instances = var.dataproc_worker_count
      machine_type  = "n1-standard-4"
      disk_config {
        boot_disk_type    = "pd-standard"
        boot_disk_size_gb = 100
      }
    }

    preemptible_worker_config {
      num_instances = var.dataproc_preemptible_count
      disk_config {
        boot_disk_type    = "pd-standard"
        boot_disk_size_gb = 100
      }
    }

    software_config {
      image_version = "2.1-debian11"
      override_properties = {
        "dataproc:dataproc.allow.zero.workers" = "false"
        "spark:spark.executor.memory"          = "4g"
        "spark:spark.driver.memory"            = "4g"
        "spark:spark.sql.adaptive.enabled"     = "true"
      }
      optional_components = ["JUPYTER", "ZEPPELIN"]
    }

    gce_cluster_config {
      service_account        = google_service_account.dataproc_sa.email
      service_account_scopes = ["cloud-platform"]

      dynamic "shielded_instance_config" {
        for_each = var.environment == "production" ? [1] : []
        content {
          enable_secure_boot          = true
          enable_vtpm                 = true
          enable_integrity_monitoring = true
        }
      }

      metadata = {
        "enable-oslogin" = "true"
      }

      tags = ["dataproc-cluster"]
    }

    initialization_action {
      script      = "gs://${google_storage_bucket.pipeline_artifacts.name}/scripts/dataproc-init.sh"
      timeout_sec = 300
    }

    lifecycle_config {
      idle_delete_ttl = "3600s"
    }
  }

  labels = local.labels
}

resource "google_dataproc_autoscaling_policy" "standard_policy" {
  policy_id = "standard-autoscaling-policy"
  location  = var.region

  worker_config {
    max_instances = 10
    min_instances = var.dataproc_worker_count
    weight        = 1
  }

  secondary_worker_config {
    max_instances = 20
    min_instances = var.dataproc_preemptible_count
    weight        = 1
  }

  basic_algorithm {
    cooldown_period = "120s"
    yarn_config {
      graceful_decommission_timeout = "300s"
      scale_up_factor               = 0.5
      scale_down_factor             = 0.5
    }
  }
}
