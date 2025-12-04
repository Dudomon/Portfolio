# Looker Studio Dashboard Configurations

This directory contains configuration and documentation for Looker Studio dashboards used in the Data Engineering Platform.

## Available Dashboards

### 1. Executive Dashboard

**Purpose**: High-level business metrics for leadership

**Data Sources**:
- BigQuery: `analytics_data.daily_transaction_summary`
- BigQuery: `analytics_data.user_transaction_metrics`

**Key Metrics**:
- Daily Active Users (DAU)
- Monthly Active Users (MAU)
- Total Transaction Volume
- Average Transaction Value
- User Growth Rate
- Revenue by Segment

**Visualizations**:
- Line chart: Daily transaction volume (last 30 days)
- Scorecard: Key metrics with month-over-month comparison
- Pie chart: Revenue distribution by user segment
- Bar chart: Top 10 merchants by revenue
- Table: User cohort analysis

**Filters**:
- Date range selector
- User segment filter
- Currency filter

**Refresh Schedule**: Every 6 hours

### 2. Data Quality Dashboard

**Purpose**: Monitor data pipeline health and quality metrics

**Data Sources**:
- BigQuery: `data_quality.validation_results`
- BigQuery: `data_quality.pipeline_metrics`
- BigQuery: `data_quality.daily_quality_scores`

**Key Metrics**:
- Validation Success Rate (target: >95%)
- Failed Expectations Count
- Data Freshness by Table
- Schema Drift Incidents
- Pipeline SLA Compliance

**Visualizations**:
- Time series: Validation success rate over time
- Heatmap: Quality score by checkpoint and date
- Scorecard: Current validation status
- Table: Recent validation failures with details
- Bar chart: Failed expectations by category

**Alerts**:
- Email notification when quality score drops below 90%

**Refresh Schedule**: Every hour

### 3. Operations Dashboard

**Purpose**: Real-time monitoring for data engineering team

**Data Sources**:
- Cloud Monitoring API (Dataflow metrics)
- Cloud Monitoring API (BigQuery metrics)
- BigQuery: `data_quality.pipeline_metrics`

**Key Metrics**:
- Active Pipeline Count
- System Lag (streaming pipelines)
- Job Success Rate
- BigQuery Slot Utilization
- Query Execution Time (p95)
- Cost per Day by Service

**Visualizations**:
- Gauge: Current system lag
- Line chart: Throughput (elements/second)
- Stacked area chart: Cost breakdown by service
- Status table: Pipeline health status
- Line chart: BigQuery slot utilization

**Refresh Schedule**: Real-time (1 minute intervals)

## Creating Dashboards

### Step 1: Connect to BigQuery

1. Go to Looker Studio (https://lookerstudio.google.com)
2. Click "Create" > "Data Source"
3. Select "BigQuery"
4. Choose your project, dataset, and table
5. Click "Connect"

### Step 2: Import Dashboard Template

Unfortunately, Looker Studio doesn't support JSON export/import for dashboards. To recreate the dashboards:

1. Use the specifications above to manually create each visualization
2. Connect to the appropriate BigQuery tables
3. Configure filters and date ranges
4. Set up data refresh schedules

### Step 3: Configure Data Freshness

1. In Data Source settings, set "Data Freshness" to:
   - Executive Dashboard: 6 hours
   - Data Quality Dashboard: 1 hour
   - Operations Dashboard: 1 minute

### Step 4: Share with Stakeholders

1. Click "Share" button in top right
2. Add email addresses
3. Set permissions (Viewer or Editor)
4. Enable "Send email notification"

## Dashboard Access URLs

After creation, dashboards will be available at:

```
Executive Dashboard:
https://lookerstudio.google.com/reporting/[DASHBOARD_ID]

Data Quality Dashboard:
https://lookerstudio.google.com/reporting/[DASHBOARD_ID]

Operations Dashboard:
https://lookerstudio.google.com/reporting/[DASHBOARD_ID]
```

## Sample SQL Queries for Custom Fields

### Calculate Month-over-Month Growth

```sql
SELECT
  transaction_date,
  active_users,
  LAG(active_users) OVER (ORDER BY transaction_date) as prev_month_users,
  SAFE_DIVIDE(
    active_users - LAG(active_users) OVER (ORDER BY transaction_date),
    LAG(active_users) OVER (ORDER BY transaction_date)
  ) * 100 as growth_rate
FROM `project.analytics_data.daily_transaction_summary`
WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY)
```

### User Cohort Analysis

```sql
WITH cohorts AS (
  SELECT
    user_id,
    DATE_TRUNC(first_transaction_date, MONTH) as cohort_month,
    DATE_TRUNC(last_transaction_date, MONTH) as activity_month,
    total_amount
  FROM `project.analytics_data.user_transaction_metrics`
)
SELECT
  cohort_month,
  activity_month,
  COUNT(DISTINCT user_id) as active_users,
  SUM(total_amount) as revenue
FROM cohorts
GROUP BY cohort_month, activity_month
ORDER BY cohort_month, activity_month
```

## Best Practices

1. **Performance**:
   - Use aggregated tables instead of raw data when possible
   - Add date filters to all queries to limit scanned data
   - Use cached data sources for frequently accessed dashboards

2. **Data Freshness**:
   - Set appropriate refresh intervals based on use case
   - Don't refresh more frequently than source data updates
   - Use real-time queries only when necessary

3. **User Experience**:
   - Limit dashboard to 10-15 visualizations for fast loading
   - Provide clear labels and descriptions
   - Include data source information in footer
   - Add last refresh timestamp

4. **Access Control**:
   - Use Google Groups for team access management
   - Grant "Viewer" access by default
   - Only give "Editor" to dashboard owners
   - Use row-level security for sensitive data

## Troubleshooting

### Dashboard Not Loading

1. Check data source connection status
2. Verify BigQuery tables exist and contain data
3. Check query syntax in custom fields
4. Review Cloud Monitoring for API quota issues

### Incorrect Data

1. Verify date range filters are applied correctly
2. Check for timezone mismatches
3. Validate BigQuery query results directly
4. Confirm data refresh schedule is running

### Performance Issues

1. Reduce number of visualizations per page
2. Add more restrictive filters
3. Use extract data sources instead of live connections
4. Implement aggregation in BigQuery before visualization

## Support

For dashboard issues or feature requests:
- Data Engineering Team: data-engineering@example.com
- Slack Channel: #data-platform
- Documentation: [Internal Wiki]

## Maintenance Schedule

- Monthly: Review dashboard usage and optimize slow queries
- Quarterly: Update visualizations based on user feedback
- Annually: Archive unused dashboards
