{{
  config(
    materialized='incremental',
    unique_key='user_id',
    partition_by={
      'field': 'last_transaction_date',
      'data_type': 'date'
    },
    cluster_by=['user_id', 'user_segment'],
    description='Fact table with comprehensive user behavioral metrics and segmentation'
  )
}}

WITH user_transactions AS (
    SELECT
        user_id,
        COUNT(DISTINCT transaction_id) AS total_transactions,
        COUNT(DISTINCT DATE(transaction_timestamp)) AS transaction_days,
        SUM(CASE WHEN transaction_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,
        SUM(CASE WHEN transaction_type = 'refund' THEN 1 ELSE 0 END) AS refund_count,
        SUM(amount) AS total_amount,
        AVG(amount) AS avg_transaction_amount,
        MIN(transaction_timestamp) AS first_transaction_date,
        MAX(transaction_timestamp) AS last_transaction_date,
        COUNT(DISTINCT merchant_id) AS unique_merchants_count
    FROM {{ ref('stg_transactions') }}
    {% if is_incremental() %}
    WHERE DATE(transaction_timestamp) > (SELECT MAX(last_transaction_date) FROM {{ this }})
    {% endif %}
    GROUP BY user_id
),

user_segments AS (
    SELECT
        user_id,
        total_transactions,
        transaction_days,
        purchase_count,
        refund_count,
        total_amount,
        avg_transaction_amount,
        first_transaction_date,
        last_transaction_date,
        unique_merchants_count,
        SAFE_DIVIDE(refund_count, total_transactions) AS refund_rate,
        DATE_DIFF(CURRENT_DATE(), DATE(last_transaction_date), DAY) AS days_since_last_transaction,
        DATE_DIFF(DATE(last_transaction_date), DATE(first_transaction_date), DAY) AS customer_lifetime_days,
        CASE
            WHEN total_amount >= 10000 THEN 'VIP'
            WHEN total_amount >= 5000 THEN 'High Value'
            WHEN total_amount >= 1000 THEN 'Medium Value'
            ELSE 'Low Value'
        END AS user_segment,
        CASE
            WHEN DATE_DIFF(CURRENT_DATE(), DATE(last_transaction_date), DAY) <= 30 THEN 'Active'
            WHEN DATE_DIFF(CURRENT_DATE(), DATE(last_transaction_date), DAY) <= 90 THEN 'At Risk'
            ELSE 'Churned'
        END AS activity_status
    FROM user_transactions
)

SELECT
    user_id,
    total_transactions,
    transaction_days,
    purchase_count,
    refund_count,
    total_amount,
    avg_transaction_amount,
    first_transaction_date,
    DATE(last_transaction_date) AS last_transaction_date,
    unique_merchants_count,
    refund_rate,
    days_since_last_transaction,
    customer_lifetime_days,
    user_segment,
    activity_status,
    CURRENT_TIMESTAMP() AS calculation_timestamp
FROM user_segments
