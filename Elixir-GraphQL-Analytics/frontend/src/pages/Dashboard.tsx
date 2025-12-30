import { useQuery, useSubscription } from "@apollo/client";
import { useState, useEffect, useCallback } from "react";
import { MetricsCard } from "../components/MetricsCard";
import { TimeSeriesChart } from "../components/TimeSeriesChart";
import { gql } from "@apollo/client";

/**
 * Analytics dashboard with real time updates.
 *
 * Data flow:
 * 1. Initial query fetches current metrics
 * 2. Subscription receives metric updates
 * 3. Local state merges updates into displayed data
 * 4. React re renders affected components
 *
 * Why subscription over polling:
 * - Immediate updates when metrics change
 * - No wasted requests when nothing changes
 * - Single WebSocket connection for all metrics
 * - Battery efficient on mobile devices
 *
 * Subscription reconnection:
 * Apollo Client automatically reconnects on connection loss.
 * After reconnect, we refetch full state to ensure consistency.
 */

const DASHBOARD_QUERY = gql`
  query GetDashboardMetrics {
    dashboardMetrics {
      activeUsers
      activeUsersChange
      totalUsers
      newUsersToday
      ordersToday
      ordersChange
      revenueToday
      revenueChange
      ordersPerMinute
    }
    hourlyActivity(hours: 24) {
      hour
      count
    }
  }
`;

const METRICS_SUBSCRIPTION = gql`
  subscription OnMetricsUpdated {
    metricsUpdated {
      metric
      value
      timestamp
      change
    }
  }
`;

interface DashboardMetrics {
  activeUsers: number;
  activeUsersChange: number | null;
  totalUsers: number;
  newUsersToday: number;
  ordersToday: number;
  ordersChange: number | null;
  revenueToday: string;
  revenueChange: number | null;
  ordersPerMinute: number;
}

interface HourlyDataPoint {
  hour: number;
  count: number;
}

interface MetricUpdate {
  metric: string;
  value: string;
  timestamp: string;
  change: number | null;
}

export function Dashboard(): JSX.Element {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [hourlyData, setHourlyData] = useState<HourlyDataPoint[]>([]);

  const { data, loading, error, refetch } = useQuery(DASHBOARD_QUERY, {
    fetchPolicy: "cache-and-network",
  });

  // Handle subscription updates
  const { data: subscriptionData } = useSubscription(METRICS_SUBSCRIPTION, {
    onSubscriptionComplete: () => {
      // Refetch full state on reconnect to ensure consistency
      refetch();
    },
  });

  // Initialize from query data
  useEffect(() => {
    if (data) {
      setMetrics(data.dashboardMetrics);
      setHourlyData(data.hourlyActivity);
    }
  }, [data]);

  // Apply subscription updates
  useEffect(() => {
    if (subscriptionData?.metricsUpdated && metrics) {
      const update = subscriptionData.metricsUpdated as MetricUpdate;

      setMetrics((prev) => {
        if (!prev) return prev;

        const newMetrics = { ...prev };
        const value = parseMetricValue(update.metric, update.value);

        switch (update.metric) {
          case "ACTIVE_USERS":
            newMetrics.activeUsers = value as number;
            if (update.change !== null) newMetrics.activeUsersChange = update.change;
            break;
          case "ORDERS_TODAY":
            newMetrics.ordersToday = value as number;
            if (update.change !== null) newMetrics.ordersChange = update.change;
            break;
          case "REVENUE_TODAY":
            newMetrics.revenueToday = value as string;
            if (update.change !== null) newMetrics.revenueChange = update.change;
            break;
          case "ORDERS_PER_MINUTE":
            newMetrics.ordersPerMinute = value as number;
            break;
        }

        return newMetrics;
      });
    }
  }, [subscriptionData, metrics]);

  const handleRefresh = useCallback(() => {
    refetch();
  }, [refetch]);

  if (loading && !metrics) {
    return <DashboardSkeleton />;
  }

  if (error) {
    return (
      <div className="dashboard-error">
        <h2>Failed to load dashboard</h2>
        <p>{error.message}</p>
        <button onClick={handleRefresh}>Retry</button>
      </div>
    );
  }

  if (!metrics) {
    return <DashboardSkeleton />;
  }

  return (
    <div className="dashboard">
      <header className="dashboard__header">
        <h1>Analytics Dashboard</h1>
        <button className="refresh-button" onClick={handleRefresh}>
          Refresh
        </button>
      </header>

      <section className="dashboard__metrics">
        <MetricsCard
          title="Active Users"
          value={metrics.activeUsers}
          change={metrics.activeUsersChange}
          format="number"
        />
        <MetricsCard
          title="Total Users"
          value={metrics.totalUsers}
          format="number"
        />
        <MetricsCard
          title="New Today"
          value={metrics.newUsersToday}
          format="number"
        />
        <MetricsCard
          title="Orders Today"
          value={metrics.ordersToday}
          change={metrics.ordersChange}
          format="number"
        />
        <MetricsCard
          title="Revenue Today"
          value={metrics.revenueToday}
          change={metrics.revenueChange}
          format="currency"
        />
        <MetricsCard
          title="Orders/Min"
          value={metrics.ordersPerMinute}
          format="decimal"
        />
      </section>

      <section className="dashboard__charts">
        <div className="chart-container">
          <h2>Hourly Activity</h2>
          <TimeSeriesChart
            data={hourlyData}
            xKey="hour"
            yKey="count"
            xLabel="Hour"
            yLabel="Events"
          />
        </div>
      </section>
    </div>
  );
}

function DashboardSkeleton(): JSX.Element {
  return (
    <div className="dashboard dashboard--loading">
      <header className="dashboard__header">
        <div className="skeleton skeleton--text" style={{ width: 200 }} />
      </header>
      <section className="dashboard__metrics">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="metrics-card metrics-card--skeleton">
            <div className="skeleton skeleton--text" style={{ width: "60%" }} />
            <div className="skeleton skeleton--heading" style={{ width: "80%" }} />
          </div>
        ))}
      </section>
    </div>
  );
}

function parseMetricValue(metric: string, value: string): number | string {
  switch (metric) {
    case "ACTIVE_USERS":
    case "ORDERS_TODAY":
      return parseInt(value, 10);
    case "ORDERS_PER_MINUTE":
      return parseFloat(value);
    case "REVENUE_TODAY":
      return value;
    default:
      return value;
  }
}
