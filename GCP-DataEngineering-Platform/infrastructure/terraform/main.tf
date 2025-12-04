terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "terraform-state-bucket"
    prefix = "data-platform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  labels = {
    environment = var.environment
    managed_by  = "terraform"
    project     = "data-engineering-platform"
  }
}
