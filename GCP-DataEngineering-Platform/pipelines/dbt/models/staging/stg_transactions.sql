{{
  config(
    materialized='view',
    description='Staging view for transactions with basic cleaning and type casting'
  )
}}

WITH source_data AS (
    SELECT
        transaction_id,
        user_id,
        transaction_type,
        CAST(amount AS NUMERIC) AS amount,
        UPPER(currency) AS currency,
        merchant_id,
        PARSE_JSON(metadata) AS metadata_json,
        TIMESTAMP(transaction_timestamp) AS transaction_timestamp,
        TIMESTAMP(ingestion_timestamp) AS ingestion_timestamp
    FROM {{ source('raw_data', 'transactions') }}
    WHERE DATE(ingestion_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {{ var('lookback_days') }} DAY)
),

cleaned AS (
    SELECT
        transaction_id,
        user_id,
        transaction_type,
        amount,
        currency,
        merchant_id,
        metadata_json,
        transaction_timestamp,
        ingestion_timestamp,
        DATE(transaction_timestamp) AS transaction_date,
        EXTRACT(HOUR FROM transaction_timestamp) AS transaction_hour,
        EXTRACT(DAYOFWEEK FROM transaction_timestamp) AS day_of_week
    FROM source_data
    WHERE
        transaction_id IS NOT NULL
        AND user_id IS NOT NULL
        AND amount > 0
        AND transaction_timestamp IS NOT NULL
)

SELECT * FROM cleaned
