-- ClickHouse initialization script for analytics database

-- Create database
CREATE DATABASE IF NOT EXISTS analytics;

USE analytics;

-- Events table (raw events from Kafka)
CREATE TABLE IF NOT EXISTS events (
    event_id String,
    event_type String,
    timestamp DateTime64(3),
    user_id String,
    session_id String,
    product_id String,
    product_name String,
    category String,
    price Decimal(10, 2),
    quantity UInt32,
    revenue Decimal(10, 2),
    country String,
    city String,
    device_type String,
    browser String,
    metadata String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, event_type, user_id)
TTL timestamp + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- Aggregated metrics by minute
CREATE TABLE IF NOT EXISTS metrics_1min (
    timestamp DateTime,
    event_type String,
    total_events UInt64,
    unique_users UInt64,
    total_revenue Decimal(18, 2),
    avg_order_value Decimal(10, 2),
    top_products Array(Tuple(String, UInt64)),
    countries Array(String)
) ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, event_type)
TTL timestamp + INTERVAL 30 DAY;

-- Aggregated metrics by hour
CREATE TABLE IF NOT EXISTS metrics_1hour (
    timestamp DateTime,
    event_type String,
    total_events UInt64,
    unique_users UInt64,
    total_revenue Decimal(18, 2),
    avg_order_value Decimal(10, 2),
    peak_transactions_per_second Float32
) ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, event_type)
TTL timestamp + INTERVAL 180 DAY;

-- Product analytics
CREATE TABLE IF NOT EXISTS product_stats (
    timestamp DateTime,
    product_id String,
    product_name String,
    category String,
    total_sales UInt64,
    total_revenue Decimal(18, 2),
    unique_buyers UInt64,
    avg_price Decimal(10, 2)
) ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, product_id);

-- Geographic analytics
CREATE TABLE IF NOT EXISTS geo_stats (
    timestamp DateTime,
    country String,
    city String,
    total_events UInt64,
    total_revenue Decimal(18, 2),
    unique_users UInt64
) ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, country, city);

-- User behavior analytics
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id String,
    user_id String,
    start_time DateTime,
    end_time DateTime,
    total_events UInt32,
    total_revenue Decimal(10, 2),
    device_type String,
    country String
) ENGINE = ReplacingMergeTree()
ORDER BY (session_id, user_id);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    alert_id String,
    alert_type String,
    severity Enum('low' = 1, 'medium' = 2, 'high' = 3, 'critical' = 4),
    timestamp DateTime,
    metric_name String,
    metric_value Float64,
    threshold Float64,
    message String,
    resolved Bool DEFAULT false
) ENGINE = MergeTree()
ORDER BY (timestamp, severity, alert_type)
TTL timestamp + INTERVAL 30 DAY;

-- Materialized view for real-time aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_realtime_metrics
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, event_type)
AS SELECT
    toStartOfMinute(timestamp) as timestamp,
    event_type,
    count() as total_events,
    uniq(user_id) as unique_users,
    sum(revenue) as total_revenue,
    avg(price) as avg_order_value
FROM events
GROUP BY timestamp, event_type;

-- Materialized view for product rankings
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_product_rankings
ENGINE = ReplacingMergeTree()
ORDER BY (timestamp, product_id)
AS SELECT
    toStartOfHour(timestamp) as timestamp,
    product_id,
    any(product_name) as product_name,
    any(category) as category,
    count() as total_sales,
    sum(revenue) as total_revenue,
    uniq(user_id) as unique_buyers,
    avg(price) as avg_price
FROM events
WHERE event_type = 'purchase'
GROUP BY timestamp, product_id;

-- Create indexes for faster queries
ALTER TABLE events ADD INDEX idx_user_id user_id TYPE bloom_filter GRANULARITY 1;
ALTER TABLE events ADD INDEX idx_product_id product_id TYPE bloom_filter GRANULARITY 1;
ALTER TABLE events ADD INDEX idx_country country TYPE set(100) GRANULARITY 1;

-- Insert sample data for testing
INSERT INTO events VALUES
    ('evt_001', 'purchase', now() - INTERVAL 5 MINUTE, 'user_001', 'sess_001', 'prod_001', 'Laptop Pro', 'Electronics', 1299.99, 1, 1299.99, 'USA', 'New York', 'desktop', 'Chrome', '{}'),
    ('evt_002', 'purchase', now() - INTERVAL 4 MINUTE, 'user_002', 'sess_002', 'prod_002', 'Wireless Mouse', 'Accessories', 29.99, 2, 59.98, 'Brazil', 'São Paulo', 'mobile', 'Safari', '{}'),
    ('evt_003', 'purchase', now() - INTERVAL 3 MINUTE, 'user_003', 'sess_003', 'prod_003', 'Mechanical Keyboard', 'Accessories', 149.99, 1, 149.99, 'Germany', 'Berlin', 'desktop', 'Firefox', '{}'),
    ('evt_004', 'view', now() - INTERVAL 2 MINUTE, 'user_004', 'sess_004', 'prod_001', 'Laptop Pro', 'Electronics', 1299.99, 0, 0, 'UK', 'London', 'tablet', 'Chrome', '{}'),
    ('evt_005', 'purchase', now() - INTERVAL 1 MINUTE, 'user_005', 'sess_005', 'prod_004', 'USB-C Cable', 'Accessories', 19.99, 3, 59.97, 'Canada', 'Toronto', 'mobile', 'Chrome', '{}');

-- Grant permissions
GRANT ALL ON analytics.* TO admin;
