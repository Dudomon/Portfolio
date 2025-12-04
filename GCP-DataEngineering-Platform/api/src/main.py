"""
FastAPI application for data collaboration and access.

Provides REST API endpoints for:
- Querying analytics data
- Accessing user and merchant metrics
- Data quality status
- Pipeline health monitoring
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
from datetime import datetime, date
from pydantic import BaseModel
from google.cloud import bigquery
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Engineering Platform API",
    description="REST API for accessing analytics data and monitoring pipeline health",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global BigQuery client
PROJECT_ID = "your-project-id"
bq_client = None


def get_bigquery_client() -> bigquery.Client:
    """
    Get or create BigQuery client instance.

    Returns:
        Initialized BigQuery client
    """
    global bq_client
    if bq_client is None:
        bq_client = bigquery.Client(project=PROJECT_ID)
    return bq_client


# Pydantic models for request/response
class UserMetrics(BaseModel):
    """User transaction metrics."""
    user_id: str
    total_transactions: int
    total_amount: float
    average_amount: float
    first_transaction_date: datetime
    last_transaction_date: datetime
    unique_merchants: int
    purchase_count: int
    refund_count: int
    refund_rate: float
    days_active: int


class MerchantMetrics(BaseModel):
    """Merchant transaction metrics."""
    merchant_id: str
    total_transactions: int
    unique_customers: int
    total_revenue: float
    average_transaction_value: float
    first_transaction_date: datetime
    last_transaction_date: datetime


class DataQualityStatus(BaseModel):
    """Data quality validation status."""
    checkpoint_name: str
    success: bool
    success_percent: float
    evaluated_expectations: int
    successful_expectations: int
    unsuccessful_expectations: int
    validation_timestamp: datetime


class PipelineMetric(BaseModel):
    """Pipeline performance metric."""
    pipeline_name: str
    metric_type: str
    metric_value: float
    metric_unit: str
    metric_timestamp: datetime


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "service": "Data Engineering Platform API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        client = get_bigquery_client()
        # Simple query to verify BigQuery connectivity
        query = "SELECT 1 as health_check"
        client.query(query).result()
        return {"status": "healthy", "bigquery": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/api/v1/users/{user_id}/metrics", response_model=UserMetrics)
def get_user_metrics(user_id: str, client: bigquery.Client = Depends(get_bigquery_client)):
    """
    Retrieve transaction metrics for a specific user.

    Args:
        user_id: User identifier
        client: BigQuery client instance

    Returns:
        User transaction metrics
    """
    query = f"""
        SELECT
            user_id,
            total_transactions,
            total_amount,
            average_amount,
            first_transaction_date,
            last_transaction_date,
            unique_merchants,
            purchase_count,
            refund_count,
            refund_rate,
            days_active
        FROM `{PROJECT_ID}.analytics_data.user_transaction_metrics`
        WHERE user_id = @user_id
        LIMIT 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
        ]
    )

    try:
        results = client.query(query, job_config=job_config).result()
        row = next(iter(results), None)

        if not row:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        return UserMetrics(**dict(row))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying user metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/users/top", response_model=List[UserMetrics])
def get_top_users(
    limit: int = Query(10, ge=1, le=100),
    order_by: str = Query("total_amount", regex="^(total_amount|total_transactions|average_amount)$"),
    client: bigquery.Client = Depends(get_bigquery_client)
):
    """
    Retrieve top users by specified metric.

    Args:
        limit: Number of users to return (1-100)
        order_by: Metric to order by (total_amount, total_transactions, average_amount)
        client: BigQuery client instance

    Returns:
        List of top user metrics
    """
    query = f"""
        SELECT
            user_id,
            total_transactions,
            total_amount,
            average_amount,
            first_transaction_date,
            last_transaction_date,
            unique_merchants,
            purchase_count,
            refund_count,
            refund_rate,
            days_active
        FROM `{PROJECT_ID}.analytics_data.user_transaction_metrics`
        ORDER BY {order_by} DESC
        LIMIT @limit
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("limit", "INT64", limit)
        ]
    )

    try:
        results = client.query(query, job_config=job_config).result()
        return [UserMetrics(**dict(row)) for row in results]

    except Exception as e:
        logger.error(f"Error querying top users: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/merchants/{merchant_id}/metrics", response_model=MerchantMetrics)
def get_merchant_metrics(merchant_id: str, client: bigquery.Client = Depends(get_bigquery_client)):
    """
    Retrieve transaction metrics for a specific merchant.

    Args:
        merchant_id: Merchant identifier
        client: BigQuery client instance

    Returns:
        Merchant transaction metrics
    """
    query = f"""
        SELECT
            merchant_id,
            total_transactions,
            unique_customers,
            total_revenue,
            average_transaction_value,
            first_transaction_date,
            last_transaction_date
        FROM `{PROJECT_ID}.analytics_data.merchant_transaction_metrics`
        WHERE merchant_id = @merchant_id
        LIMIT 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("merchant_id", "STRING", merchant_id)
        ]
    )

    try:
        results = client.query(query, job_config=job_config).result()
        row = next(iter(results), None)

        if not row:
            raise HTTPException(status_code=404, detail=f"Merchant {merchant_id} not found")

        return MerchantMetrics(**dict(row))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying merchant metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/data-quality/status", response_model=List[DataQualityStatus])
def get_data_quality_status(
    hours: int = Query(24, ge=1, le=168),
    client: bigquery.Client = Depends(get_bigquery_client)
):
    """
    Retrieve recent data quality validation results.

    Args:
        hours: Number of hours to look back (1-168)
        client: BigQuery client instance

    Returns:
        List of recent validation results
    """
    query = f"""
        SELECT
            checkpoint_name,
            success,
            success_percent,
            evaluated_expectations,
            successful_expectations,
            unsuccessful_expectations,
            validation_timestamp
        FROM `{PROJECT_ID}.data_quality.validation_results`
        WHERE validation_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @hours HOUR)
        ORDER BY validation_timestamp DESC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("hours", "INT64", hours)
        ]
    )

    try:
        results = client.query(query, job_config=job_config).result()
        return [DataQualityStatus(**dict(row)) for row in results]

    except Exception as e:
        logger.error(f"Error querying data quality status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/pipelines/metrics", response_model=List[PipelineMetric])
def get_pipeline_metrics(
    pipeline_name: Optional[str] = None,
    metric_type: Optional[str] = None,
    hours: int = Query(24, ge=1, le=168),
    client: bigquery.Client = Depends(get_bigquery_client)
):
    """
    Retrieve pipeline performance metrics.

    Args:
        pipeline_name: Filter by pipeline name (optional)
        metric_type: Filter by metric type (optional)
        hours: Number of hours to look back (1-168)
        client: BigQuery client instance

    Returns:
        List of pipeline metrics
    """
    where_clauses = [
        "metric_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @hours HOUR)"
    ]

    query_params = [
        bigquery.ScalarQueryParameter("hours", "INT64", hours)
    ]

    if pipeline_name:
        where_clauses.append("pipeline_name = @pipeline_name")
        query_params.append(
            bigquery.ScalarQueryParameter("pipeline_name", "STRING", pipeline_name)
        )

    if metric_type:
        where_clauses.append("metric_type = @metric_type")
        query_params.append(
            bigquery.ScalarQueryParameter("metric_type", "STRING", metric_type)
        )

    where_clause = " AND ".join(where_clauses)

    query = f"""
        SELECT
            pipeline_name,
            metric_type,
            metric_value,
            metric_unit,
            metric_timestamp
        FROM `{PROJECT_ID}.data_quality.pipeline_metrics`
        WHERE {where_clause}
        ORDER BY metric_timestamp DESC
        LIMIT 1000
    """

    job_config = bigquery.QueryJobConfig(query_parameters=query_params)

    try:
        results = client.query(query, job_config=job_config).result()
        return [PipelineMetric(**dict(row)) for row in results]

    except Exception as e:
        logger.error(f"Error querying pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/analytics/daily-summary")
def get_daily_summary(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    client: bigquery.Client = Depends(get_bigquery_client)
):
    """
    Retrieve daily transaction summary.

    Args:
        start_date: Start date for summary (optional, defaults to 30 days ago)
        end_date: End date for summary (optional, defaults to today)
        client: BigQuery client instance

    Returns:
        Daily transaction summary data
    """
    query = f"""
        SELECT
            transaction_date,
            active_users,
            total_transactions,
            total_volume,
            avg_transaction_value
        FROM `{PROJECT_ID}.analytics_data.daily_transaction_summary`
        WHERE transaction_date BETWEEN @start_date AND @end_date
        ORDER BY transaction_date DESC
    """

    if not start_date:
        start_date = date.today().replace(day=1)
    if not end_date:
        end_date = date.today()

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date)
        ]
    )

    try:
        results = client.query(query, job_config=job_config).result()
        return [dict(row) for row in results]

    except Exception as e:
        logger.error(f"Error querying daily summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
