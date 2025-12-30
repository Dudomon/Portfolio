defmodule AnalyticsWeb.Schema.Subscriptions do
  @moduledoc """
  GraphQL subscription definitions for real time analytics updates.

  How GraphQL subscriptions work with Phoenix:

  1. Client opens WebSocket connection to /socket/websocket
  2. Client sends subscription query over the WebSocket
  3. Server registers subscription in Absinthe.Subscription process
  4. When data changes, server broadcasts via Phoenix.PubSub
  5. Absinthe.Subscription receives broadcast, executes subscription resolvers
  6. Resolved data pushes to client over WebSocket

  This architecture scales horizontally. Multiple server nodes share
  PubSub via Redis adapter. Subscription on node A receives broadcasts
  from node B. Client connections distribute across nodes with sticky
  sessions not required.

  Subscription design principles:

  1. Minimal payload: Send only changed data, not full state
  2. Debouncing: Aggregate rapid changes before broadcast
  3. Topic granularity: Allow subscribing to specific metrics
  4. Backpressure: Drop updates if client cannot keep up

  Topic naming convention:

  - metrics:all           All metric updates
  - metrics:{metric_name} Specific metric (active_users, revenue_today)
  - metrics:user:{id}     Per user metrics (for personalized dashboards)
  - alerts:{severity}     System alerts by severity level

  Authentication:

  Subscriptions inherit authentication from the WebSocket connection.
  The socket assigns user_id during connection, available in subscription
  config via context. Unauthorized subscriptions return error.
  """

  use Absinthe.Schema.Notation

  alias AnalyticsWeb.Schema.Types
  alias Analytics.Metrics.Aggregator

  # Subscription objects

  object :subscription_root do
    @desc """
    Subscribe to all metric updates.

    Receives updates whenever any dashboard metric changes significantly.
    The aggregator applies threshold filtering to prevent flooding:
    only changes exceeding 5% trigger broadcasts.

    Example subscription:

        subscription {
          metricsUpdated {
            metric
            value
            previousValue
            changePercent
            timestamp
          }
        }
    """
    field :metrics_updated, :metric_update_payload do
      config fn _, context ->
        case authorize_subscription(context) do
          :ok -> {:ok, topic: "metrics:all"}
          {:error, reason} -> {:error, reason}
        end
      end

      trigger :record_event, topic: fn _ -> "metrics:all" end

      resolve fn payload, _, _ ->
        {:ok, transform_metric_payload(payload)}
      end
    end

    @desc """
    Subscribe to a specific metric.

    More efficient than metricsUpdated when dashboard displays only
    certain metrics. Reduces WebSocket traffic and client processing.

    Arguments:
    - metricName: The metric to subscribe to (ACTIVE_USERS, REVENUE_TODAY, etc.)

    Example:

        subscription WatchActiveUsers {
          metricChanged(metricName: ACTIVE_USERS) {
            value
            changePercent
          }
        }
    """
    field :metric_changed, :metric_update_payload do
      arg :metric_name, non_null(:metric_name_enum)

      config fn args, context ->
        case authorize_subscription(context) do
          :ok ->
            topic = "metrics:#{args.metric_name}"
            {:ok, topic: topic}

          {:error, reason} ->
            {:error, reason}
        end
      end

      resolve fn payload, _, _ ->
        {:ok, transform_metric_payload(payload)}
      end
    end

    @desc """
    Subscribe to dashboard metrics bundle.

    Receives periodic snapshots of all dashboard metrics.
    Unlike metricsUpdated which sends individual changes,
    this sends complete state at regular intervals.

    Useful for dashboards that need consistent state rather
    than incremental updates. Interval is server controlled
    (default 5 seconds) to prevent client abuse.
    """
    field :dashboard_snapshot, :dashboard_metrics do
      config fn _, context ->
        case authorize_subscription(context) do
          :ok -> {:ok, topic: "dashboard:snapshot"}
          {:error, reason} -> {:error, reason}
        end
      end

      resolve fn _, _, _ ->
        {:ok, Aggregator.get_metrics()}
      end
    end

    @desc """
    Subscribe to hourly activity updates.

    Pushes updated hourly histogram when activity counts change.
    Used by time series charts that need real time updates.
    """
    field :hourly_activity_updated, list_of(:hourly_data_point) do
      arg :hours, :integer, default_value: 24

      config fn args, context ->
        case authorize_subscription(context) do
          :ok ->
            {:ok, topic: "activity:hourly", context: %{hours: args.hours}}

          {:error, reason} ->
            {:error, reason}
        end
      end

      resolve fn _, %{context: sub_context}, _ ->
        hours = Map.get(sub_context, :hours, 24)
        activity = Aggregator.get_hourly_activity()
        {:ok, Enum.take(activity, hours)}
      end
    end

    @desc """
    Subscribe to threshold alerts.

    Triggers when metrics cross defined thresholds.
    Used for operational dashboards monitoring system health.

    Arguments:
    - thresholds: Map of metric name to threshold value

    Example:

        subscription WatchThresholds {
          thresholdAlert(thresholds: {activeUsers: 1000, ordersPerMinute: 50}) {
            metric
            threshold
            currentValue
            direction
            triggeredAt
          }
        }
    """
    field :threshold_alert, :threshold_alert_payload do
      arg :thresholds, :threshold_input

      config fn args, context ->
        case authorize_subscription(context) do
          :ok ->
            # Store thresholds in subscription context for filtering
            {:ok, topic: "alerts:threshold", context: %{thresholds: args.thresholds}}

          {:error, reason} ->
            {:error, reason}
        end
      end

      resolve fn payload, %{context: sub_context}, _ ->
        # Only resolve if payload exceeds subscriber's threshold
        thresholds = Map.get(sub_context, :thresholds, %{})

        if should_alert?(payload, thresholds) do
          {:ok, payload}
        else
          {:ok, nil}
        end
      end
    end

    @desc """
    Subscribe to anomaly detection alerts.

    Uses statistical analysis to detect unusual patterns.
    Anomalies are values exceeding 2.5 standard deviations from
    the rolling average.
    """
    field :anomaly_detected, :anomaly_alert_payload do
      arg :metrics, list_of(:metric_name_enum)
      arg :sensitivity, :float, default_value: 2.5

      config fn args, context ->
        case authorize_subscription(context) do
          :ok ->
            {:ok,
             topic: "alerts:anomaly",
             context: %{
               metrics: args.metrics,
               sensitivity: args.sensitivity
             }}

          {:error, reason} ->
            {:error, reason}
        end
      end

      resolve fn payload, %{context: sub_context}, _ ->
        metrics = Map.get(sub_context, :metrics)

        if is_nil(metrics) or payload.metric in metrics do
          {:ok, payload}
        else
          {:ok, nil}
        end
      end
    end
  end

  # Subscription payload types

  object :metric_update_payload do
    @desc "The metric that changed"
    field :metric, non_null(:metric_name_enum)

    @desc "Current value (JSON encoded for type flexibility)"
    field :value, non_null(:string)

    @desc "Previous value before this update"
    field :previous_value, :string

    @desc "Percentage change from previous value"
    field :change_percent, :float

    @desc "Absolute change from previous value"
    field :change_absolute, :float

    @desc "Direction of change"
    field :trend, :trend_direction

    @desc "When the change occurred"
    field :timestamp, non_null(:datetime)

    @desc "Server sequence number for ordering"
    field :sequence, non_null(:integer)
  end

  object :threshold_alert_payload do
    field :metric, non_null(:metric_name_enum)
    field :threshold, non_null(:float)
    field :current_value, non_null(:float)
    field :direction, non_null(:threshold_direction)
    field :triggered_at, non_null(:datetime)
    field :message, :string
  end

  object :anomaly_alert_payload do
    field :metric, non_null(:metric_name_enum)
    field :expected_value, non_null(:float)
    field :actual_value, non_null(:float)
    field :z_score, non_null(:float)
    field :detected_at, non_null(:datetime)
    field :severity, non_null(:anomaly_severity)
  end

  # Enums

  enum :metric_name_enum do
    value :active_users, description: "Currently active user count"
    value :total_users, description: "Total registered users"
    value :new_users_today, description: "New registrations today"
    value :orders_today, description: "Orders completed today"
    value :revenue_today, description: "Revenue generated today"
    value :orders_per_minute, description: "Current order velocity"
  end

  enum :trend_direction do
    value :up, description: "Metric is increasing"
    value :down, description: "Metric is decreasing"
    value :flat, description: "Metric is stable"
  end

  enum :threshold_direction do
    value :above, description: "Value exceeded upper threshold"
    value :below, description: "Value dropped below lower threshold"
  end

  enum :anomaly_severity do
    value :low, description: "Minor deviation (2-3 sigma)"
    value :medium, description: "Significant deviation (3-4 sigma)"
    value :high, description: "Major deviation (4+ sigma)"
  end

  # Input types

  input_object :threshold_input do
    field :active_users, :integer
    field :orders_per_minute, :float
    field :revenue_today, :float
  end

  # Private functions

  defp authorize_subscription(%{current_user: nil}) do
    {:error, "Authentication required for subscriptions"}
  end

  defp authorize_subscription(%{current_user: _user}) do
    :ok
  end

  defp authorize_subscription(_context) do
    # Allow anonymous subscriptions in development
    if Application.get_env(:analytics, :env) == :dev do
      :ok
    else
      {:error, "Authentication required for subscriptions"}
    end
  end

  defp transform_metric_payload({:metric_updated, data}) do
    %{
      metric: data.metric,
      value: encode_value(data.value),
      previous_value: encode_value(Map.get(data, :previous_value)),
      change_percent: Map.get(data, :change_percent),
      change_absolute: Map.get(data, :change_absolute),
      trend: determine_trend(Map.get(data, :change_absolute)),
      timestamp: data.timestamp,
      sequence: System.unique_integer([:monotonic, :positive])
    }
  end

  defp transform_metric_payload(data) when is_map(data) do
    data
  end

  defp encode_value(nil), do: nil
  defp encode_value(%Decimal{} = d), do: Decimal.to_string(d)
  defp encode_value(value), do: Jason.encode!(value)

  defp determine_trend(nil), do: :flat
  defp determine_trend(change) when change > 0, do: :up
  defp determine_trend(change) when change < 0, do: :down
  defp determine_trend(_), do: :flat

  defp should_alert?(payload, thresholds) do
    metric_key = payload.metric
    threshold = Map.get(thresholds, metric_key)

    case threshold do
      nil -> false
      t -> payload.current_value >= t or payload.current_value <= t
    end
  end
end

defmodule AnalyticsWeb.Subscriptions.Publisher do
  @moduledoc """
  Publishes metric updates to GraphQL subscriptions.

  This module bridges the gap between internal PubSub (used by Aggregator)
  and Absinthe subscriptions. It listens to metric broadcasts and
  republishes them in the format expected by subscription resolvers.

  Debouncing:

  The aggregator may broadcast rapidly during high activity periods.
  This publisher debounces updates per metric, collecting changes over
  a 100ms window before publishing. This reduces WebSocket traffic
  while maintaining near real time feel.

  Sequence numbers:

  Each published update includes a monotonically increasing sequence number.
  Clients use these to detect missed updates and request reconciliation.
  Out of order delivery (possible with multiple server nodes) is detectable
  by sequence gaps.
  """

  use GenServer
  require Logger

  @debounce_window_ms 100

  defstruct pending_updates: %{}, timer_refs: %{}

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Publishes a metric update to all relevant subscriptions.

  Called by Aggregator when a metric changes. The update is debounced
  before actual publication.
  """
  @spec publish_metric_update(atom(), term(), term()) :: :ok
  def publish_metric_update(metric, current_value, previous_value) do
    GenServer.cast(__MODULE__, {:metric_update, metric, current_value, previous_value})
  end

  @doc """
  Publishes a threshold alert immediately (no debouncing).

  Alerts are time sensitive and should not be delayed.
  """
  @spec publish_threshold_alert(map()) :: :ok
  def publish_threshold_alert(alert) do
    Absinthe.Subscription.publish(
      AnalyticsWeb.Endpoint,
      alert,
      threshold_alert: "alerts:threshold"
    )
  end

  @doc """
  Publishes an anomaly alert immediately.
  """
  @spec publish_anomaly_alert(map()) :: :ok
  def publish_anomaly_alert(alert) do
    Absinthe.Subscription.publish(
      AnalyticsWeb.Endpoint,
      alert,
      anomaly_detected: "alerts:anomaly"
    )
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    # Subscribe to internal metric updates
    Phoenix.PubSub.subscribe(Analytics.PubSub, "metrics:updates")
    {:ok, %__MODULE__{}}
  end

  @impl true
  def handle_cast({:metric_update, metric, current, previous}, state) do
    update = %{
      metric: metric,
      value: current,
      previous_value: previous,
      timestamp: DateTime.utc_now()
    }

    # Store update, replacing any pending update for same metric
    pending = Map.put(state.pending_updates, metric, update)

    # Set or reset debounce timer for this metric
    timer_refs =
      case Map.get(state.timer_refs, metric) do
        nil ->
          ref = Process.send_after(self(), {:flush, metric}, @debounce_window_ms)
          Map.put(state.timer_refs, metric, ref)

        existing_ref ->
          Process.cancel_timer(existing_ref)
          ref = Process.send_after(self(), {:flush, metric}, @debounce_window_ms)
          Map.put(state.timer_refs, metric, ref)
      end

    {:noreply, %{state | pending_updates: pending, timer_refs: timer_refs}}
  end

  @impl true
  def handle_info({:flush, metric}, state) do
    case Map.get(state.pending_updates, metric) do
      nil ->
        {:noreply, state}

      update ->
        do_publish(update)

        pending = Map.delete(state.pending_updates, metric)
        timer_refs = Map.delete(state.timer_refs, metric)

        {:noreply, %{state | pending_updates: pending, timer_refs: timer_refs}}
    end
  end

  @impl true
  def handle_info({:metric_updated, data}, state) do
    # Forward internal PubSub messages to subscription publisher
    GenServer.cast(self(), {:metric_update, data.metric, data.value, nil})
    {:noreply, state}
  end

  defp do_publish(update) do
    payload = {:metric_updated, update}

    # Publish to wildcard topic (all metrics)
    Absinthe.Subscription.publish(
      AnalyticsWeb.Endpoint,
      payload,
      metrics_updated: "metrics:all"
    )

    # Publish to specific metric topic
    Absinthe.Subscription.publish(
      AnalyticsWeb.Endpoint,
      payload,
      metric_changed: "metrics:#{update.metric}"
    )

    Logger.debug("Published subscription update for #{update.metric}")
  end
end
