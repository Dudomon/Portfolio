resource "google_monitoring_notification_channel" "email" {
  count        = var.alert_email != "" ? 1 : 0
  display_name = "Data Engineering Team Email"
  type         = "email"
  labels = {
    email_address = var.alert_email
  }
}

resource "google_monitoring_alert_policy" "dataflow_pipeline_failure" {
  count        = var.alert_email != "" ? 1 : 0
  display_name = "Dataflow Pipeline Failure"
  combiner     = "OR"

  conditions {
    display_name = "Dataflow job failed"
    condition_threshold {
      filter          = "resource.type = \"dataflow_job\" AND metric.type = \"dataflow.googleapis.com/job/is_failed\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MAX"
      }
    }
  }

  documentation {
    content = <<-EOT
      A Dataflow pipeline has failed. This requires immediate attention.

      Troubleshooting steps:
      1. Check Cloud Logging for error details
      2. Review the pipeline metrics dashboard
      3. Verify Pub/Sub subscription status
      4. Check BigQuery table schemas
      5. Review resource quotas

      Runbook: docs/runbook.md
    EOT
  }

  notification_channels = [google_monitoring_notification_channel.email[0].id]

  alert_strategy {
    auto_close = "604800s"
  }
}

resource "google_monitoring_alert_policy" "bigquery_query_errors" {
  count        = var.alert_email != "" ? 1 : 0
  display_name = "BigQuery Query Error Rate High"
  combiner     = "OR"

  conditions {
    display_name = "Query error rate > 5%"
    condition_threshold {
      filter     = "resource.type = \"bigquery_project\" AND metric.type = \"bigquery.googleapis.com/query/count\" AND metric.label.job_type = \"query\""
      duration   = "300s"
      comparison = "COMPARISON_GT"
      threshold_value = 0.05
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.project_id"]
      }
    }
  }

  documentation {
    content = <<-EOT
      BigQuery query error rate has exceeded 5%.

      Common causes:
      - Schema changes breaking queries
      - Quota exhaustion
      - Table not found errors
      - Permission issues

      Check the BigQuery audit logs for specific error messages.
    EOT
  }

  notification_channels = [google_monitoring_notification_channel.email[0].id]
}

resource "google_monitoring_alert_policy" "pipeline_lag" {
  count        = var.alert_email != "" ? 1 : 0
  display_name = "Pipeline Processing Lag High"
  combiner     = "OR"

  conditions {
    display_name = "System lag > 5 minutes"
    condition_threshold {
      filter          = "resource.type = \"dataflow_job\" AND metric.type = \"dataflow.googleapis.com/job/system_lag\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 300
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  documentation {
    content = <<-EOT
      Streaming pipeline is experiencing high lag (>5 minutes).

      This indicates the pipeline cannot keep up with incoming data volume.

      Actions:
      1. Check current worker count and autoscaling behavior
      2. Review resource utilization metrics
      3. Consider increasing max_num_workers
      4. Check for bottlenecks in downstream systems (BigQuery)
      5. Review windowing configuration
    EOT
  }

  notification_channels = [google_monitoring_notification_channel.email[0].id]
}

resource "google_monitoring_alert_policy" "data_freshness" {
  count        = var.alert_email != "" ? 1 : 0
  display_name = "Data Freshness SLA Breach"
  combiner     = "OR"

  conditions {
    display_name = "No new data in 30 minutes"
    condition_threshold {
      filter = join(" AND ", [
        "resource.type = \"bigquery_table\"",
        "metric.type = \"bigquery.googleapis.com/storage/table/row_count\"",
        "resource.label.dataset_id = \"raw_data\""
      ])
      duration        = "1800s"
      comparison      = "COMPARISON_LT"
      threshold_value = 1
      aggregations {
        alignment_period   = "1800s"
        per_series_aligner = "ALIGN_DELTA"
      }
    }
  }

  documentation {
    content = <<-EOT
      No new data has been ingested into raw_data tables for 30 minutes.

      This indicates a potential upstream issue:
      1. Check Pub/Sub topics for message backlog
      2. Verify data sources are publishing
      3. Check Dataflow pipeline status
      4. Review network connectivity
      5. Check service account permissions
    EOT
  }

  notification_channels = [google_monitoring_notification_channel.email[0].id]
}

resource "google_monitoring_dashboard" "pipeline_health" {
  dashboard_json = jsonencode({
    displayName = "Data Pipeline Health Dashboard"
    mosaicLayout = {
      columns = 12
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Dataflow Job Status"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type = \"dataflow_job\" AND metric.type = \"dataflow.googleapis.com/job/is_failed\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_MAX"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Failed (1 = Yes)"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          width  = 6
          height = 4
          widget = {
            title = "Pipeline Throughput"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type = \"dataflow_job\" AND metric.type = \"dataflow.googleapis.com/job/elements_produced_count\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Elements/sec"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "System Lag"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type = \"dataflow_job\" AND metric.type = \"dataflow.googleapis.com/job/system_lag\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Lag (seconds)"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          xPos   = 6
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "BigQuery Slots Used"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type = \"bigquery_project\" AND metric.type = \"bigquery.googleapis.com/slots/total_allocated\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Slots"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          yPos   = 8
          width  = 12
          height = 4
          widget = {
            title = "Query Execution Time (p95)"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type = \"bigquery_project\" AND metric.type = \"bigquery.googleapis.com/query/execution_times\""
                    aggregation = {
                      alignmentPeriod    = "300s"
                      perSeriesAligner   = "ALIGN_DELTA"
                      crossSeriesReducer = "REDUCE_PERCENTILE_95"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Duration (ms)"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })
}

resource "google_monitoring_dashboard" "data_quality" {
  dashboard_json = jsonencode({
    displayName = "Data Quality Dashboard"
    mosaicLayout = {
      columns = 12
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Validation Success Rate"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type = \"bigquery_table\" AND metric.type = \"logging.googleapis.com/user/data_quality_success\""
                  aggregation = {
                    alignmentPeriod  = "3600s"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
            }
          }
        },
        {
          xPos   = 6
          width  = 6
          height = 4
          widget = {
            title = "Schema Validation Failures"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type = \"bigquery_table\" AND metric.type = \"logging.googleapis.com/user/schema_validation_failure\""
                    aggregation = {
                      alignmentPeriod  = "300s"
                      perSeriesAligner = "ALIGN_SUM"
                    }
                  }
                }
                plotType = "LINE"
              }]
            }
          }
        }
      ]
    }
  })
}
