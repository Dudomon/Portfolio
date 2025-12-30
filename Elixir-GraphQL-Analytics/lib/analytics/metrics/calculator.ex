defmodule Analytics.Metrics.Calculator do
  @moduledoc """
  Pure functions for metric calculations.

  Design philosophy:

  Separating calculation logic from state management provides several benefits:

  1. Testability: Pure functions are trivial to unit test. No GenServer setup,
     no mocking, no process lifecycle concerns. Input in, output out.

  2. Reusability: Same calculations work for real time aggregation, historical
     queries, and data exports. One implementation, multiple contexts.

  3. Parallelization: Pure functions can run in parallel without coordination.
     Historical reports across date ranges parallelize trivially.

  4. Auditing: Financial metrics require audit trails. Pure functions produce
     deterministic results that can be verified against stored snapshots.

  Implementation notes:

  All monetary calculations use Decimal to avoid floating point errors.
  A naive float implementation accumulating $0.01 one million times produces
  $10000.000000000818 instead of $10000.00. Financial auditors notice.

  Statistical calculations (percentiles, moving averages) use streaming
  algorithms where possible to bound memory usage regardless of input size.
  """

  @type numeric :: integer() | float() | Decimal.t()

  # Time period calculations

  @doc """
  Calculates metrics change between two time periods.

  Returns a map with absolute change, percentage change, and trend direction.
  Handles edge cases: zero previous value, negative numbers, nil inputs.

  Example:
      calculate_change(150, 100)
      # => %{absolute: 50, percentage: 50.0, trend: :up}
  """
  @spec calculate_change(numeric() | nil, numeric() | nil) :: map()
  def calculate_change(nil, _previous), do: %{absolute: nil, percentage: nil, trend: :unknown}
  def calculate_change(_current, nil), do: %{absolute: nil, percentage: nil, trend: :unknown}

  def calculate_change(current, previous) do
    current = to_decimal(current)
    previous = to_decimal(previous)

    absolute = Decimal.sub(current, previous)

    percentage =
      if Decimal.eq?(previous, 0) do
        if Decimal.eq?(current, 0), do: Decimal.new(0), else: Decimal.new(100)
      else
        previous
        |> Decimal.div(Decimal.new(100))
        |> then(&Decimal.div(absolute, &1))
        |> Decimal.round(2)
      end

    trend =
      case Decimal.compare(absolute, 0) do
        :gt -> :up
        :lt -> :down
        :eq -> :flat
      end

    %{
      absolute: Decimal.to_float(absolute),
      percentage: Decimal.to_float(percentage),
      trend: trend
    }
  end

  @doc """
  Calculates compound growth rate over multiple periods.

  CAGR = (End Value / Start Value)^(1/periods) - 1

  Used for normalizing growth rates across different time spans.
  Monthly growth of 10% is not comparable to yearly growth of 10%.
  CAGR provides apples to apples comparison.
  """
  @spec compound_growth_rate(numeric(), numeric(), pos_integer()) :: float()
  def compound_growth_rate(end_value, start_value, periods) when periods > 0 do
    end_val = to_float(end_value)
    start_val = to_float(start_value)

    if start_val == 0 do
      0.0
    else
      ratio = end_val / start_val
      exponent = 1.0 / periods
      (:math.pow(ratio, exponent) - 1) * 100
      |> Float.round(4)
    end
  end

  # Revenue calculations

  @doc """
  Calculates average order value from total revenue and order count.

  Returns Decimal for precision. Division by zero returns zero, not error,
  as "no orders means no average" is a valid business interpretation.
  """
  @spec average_order_value(Decimal.t(), non_neg_integer()) :: Decimal.t()
  def average_order_value(_revenue, 0), do: Decimal.new(0)

  def average_order_value(revenue, order_count) do
    revenue
    |> Decimal.div(order_count)
    |> Decimal.round(2)
  end

  @doc """
  Calculates revenue per user (ARPU) for a period.

  ARPU is a key SaaS metric. Tracking ARPU trends reveals whether growth
  comes from more users (horizontal) or higher spending (vertical).
  """
  @spec revenue_per_user(Decimal.t(), non_neg_integer()) :: Decimal.t()
  def revenue_per_user(_revenue, 0), do: Decimal.new(0)

  def revenue_per_user(revenue, user_count) do
    revenue
    |> Decimal.div(user_count)
    |> Decimal.round(2)
  end

  @doc """
  Projects revenue to end of period based on current run rate.

  Used for "on track for $X this month" dashboard displays.
  Accounts for partial periods (e.g., 15 days into a 30 day month).
  """
  @spec project_revenue(Decimal.t(), Date.t(), Date.t()) :: Decimal.t()
  def project_revenue(current_revenue, period_start, period_end) do
    total_days = Date.diff(period_end, period_start) + 1
    elapsed_days = Date.diff(Date.utc_today(), period_start) + 1
    elapsed_days = min(elapsed_days, total_days)

    if elapsed_days <= 0 do
      current_revenue
    else
      daily_rate = Decimal.div(current_revenue, elapsed_days)
      Decimal.mult(daily_rate, total_days) |> Decimal.round(2)
    end
  end

  # Rate calculations

  @doc """
  Calculates events per time unit using sliding window.

  The window approach provides smoother rates than simple division.
  "Orders per minute" using total_orders / total_minutes produces a
  rate that barely moves. Using last 5 minutes shows current velocity.
  """
  @spec rate_per_minute(list(DateTime.t()), non_neg_integer()) :: float()
  def rate_per_minute(timestamps, window_minutes \\ 5) do
    now = DateTime.utc_now()
    window_start = DateTime.add(now, -window_minutes * 60, :second)

    recent_count =
      timestamps
      |> Enum.count(fn ts -> DateTime.compare(ts, window_start) != :lt end)

    (recent_count / window_minutes)
    |> Float.round(2)
  end

  @doc """
  Calculates conversion rate between funnel stages.

  Conversion rate = (conversions / impressions) * 100

  Handles the denominator zero case gracefully.
  """
  @spec conversion_rate(non_neg_integer(), non_neg_integer()) :: float()
  def conversion_rate(_conversions, 0), do: 0.0

  def conversion_rate(conversions, impressions) do
    (conversions / impressions * 100)
    |> Float.round(2)
  end

  # Statistical calculations

  @doc """
  Calculates percentile value from a sorted list.

  Uses linear interpolation between adjacent values for non integer indices.
  This matches the behavior of most statistical software.

  Percentile 50 = median, 95 = typical "high" threshold, 99 = outlier boundary.
  """
  @spec percentile(list(numeric()), number()) :: numeric() | nil
  def percentile([], _p), do: nil
  def percentile([single], _p), do: single

  def percentile(sorted_values, p) when p >= 0 and p <= 100 do
    n = length(sorted_values)
    rank = (p / 100) * (n - 1)
    lower_index = trunc(rank)
    upper_index = min(lower_index + 1, n - 1)
    fraction = rank - lower_index

    lower_value = Enum.at(sorted_values, lower_index) |> to_float()
    upper_value = Enum.at(sorted_values, upper_index) |> to_float()

    lower_value + fraction * (upper_value - lower_value)
  end

  @doc """
  Calculates exponential moving average for smoothed trend display.

  EMA weights recent values more heavily than older values.
  Smoothing factor alpha determines how quickly the average responds.
  Higher alpha = more responsive but noisier, lower alpha = smoother but laggier.

  For real time dashboards, alpha of 0.2 to 0.3 provides good balance.
  """
  @spec exponential_moving_average(list(numeric()), float()) :: float()
  def exponential_moving_average([], _alpha), do: 0.0
  def exponential_moving_average([first | rest], alpha) when alpha > 0 and alpha <= 1 do
    Enum.reduce(rest, to_float(first), fn value, ema ->
      alpha * to_float(value) + (1 - alpha) * ema
    end)
    |> Float.round(4)
  end

  @doc """
  Calculates standard deviation for variance analysis.

  Uses the two pass algorithm for numerical stability.
  First pass computes mean, second pass computes variance.
  """
  @spec standard_deviation(list(numeric())) :: float()
  def standard_deviation([]), do: 0.0
  def standard_deviation([_]), do: 0.0

  def standard_deviation(values) do
    n = length(values)
    float_values = Enum.map(values, &to_float/1)
    mean = Enum.sum(float_values) / n

    variance =
      float_values
      |> Enum.map(fn v -> :math.pow(v - mean, 2) end)
      |> Enum.sum()
      |> Kernel./(n)

    :math.sqrt(variance)
    |> Float.round(4)
  end

  # Time series aggregations

  @doc """
  Aggregates events into hourly buckets for chart display.

  Returns a map of hour => count for the specified number of hours.
  Missing hours are filled with zero for consistent chart rendering.
  """
  @spec aggregate_hourly(list(DateTime.t()), non_neg_integer()) :: list(map())
  def aggregate_hourly(timestamps, hours \\ 24) do
    now = DateTime.utc_now()

    # Initialize all hours with zero
    buckets =
      0..(hours - 1)
      |> Enum.map(fn offset ->
        hour_start = DateTime.add(now, -offset * 3600, :second)
        truncated = DateTime.truncate(hour_start, :second)
        hour_key = truncated.hour
        {hour_key, 0}
      end)
      |> Map.new()

    # Count events per hour
    filled_buckets =
      timestamps
      |> Enum.filter(fn ts ->
        DateTime.diff(now, ts, :second) < hours * 3600
      end)
      |> Enum.reduce(buckets, fn ts, acc ->
        hour_key = ts.hour
        Map.update(acc, hour_key, 1, &(&1 + 1))
      end)

    # Convert to sorted list for charts
    0..23
    |> Enum.map(fn hour ->
      %{
        hour: hour,
        count: Map.get(filled_buckets, hour, 0)
      }
    end)
  end

  @doc """
  Aggregates daily metrics from hourly data points.

  Used for rolling up granular data into daily summaries.
  Applies sum for counts, average for rates, last value for totals.
  """
  @spec aggregate_daily(list(map())) :: map()
  def aggregate_daily(hourly_data) do
    %{
      total_count: Enum.sum(Enum.map(hourly_data, & &1.count)),
      average_count: (Enum.sum(Enum.map(hourly_data, & &1.count)) / max(length(hourly_data), 1)) |> Float.round(2),
      peak_hour: Enum.max_by(hourly_data, & &1.count, fn -> %{hour: 0} end).hour,
      min_count: Enum.min(Enum.map(hourly_data, & &1.count)),
      max_count: Enum.max(Enum.map(hourly_data, & &1.count))
    }
  end

  # Anomaly detection

  @doc """
  Detects anomalies using z score threshold.

  Values more than `threshold` standard deviations from mean are anomalies.
  Default threshold of 2.5 catches approximately 1% of values in normal distribution.

  Returns list of {index, value, z_score} tuples for anomalous points.
  """
  @spec detect_anomalies(list(numeric()), float()) :: list(tuple())
  def detect_anomalies(values, threshold \\ 2.5) do
    float_values = Enum.map(values, &to_float/1)
    mean = Enum.sum(float_values) / max(length(float_values), 1)
    std_dev = standard_deviation(float_values)

    if std_dev == 0 do
      []
    else
      float_values
      |> Enum.with_index()
      |> Enum.filter(fn {value, _index} ->
        z_score = abs(value - mean) / std_dev
        z_score > threshold
      end)
      |> Enum.map(fn {value, index} ->
        z_score = abs(value - mean) / std_dev
        {index, value, Float.round(z_score, 2)}
      end)
    end
  end

  # Helper functions

  defp to_decimal(value) when is_integer(value), do: Decimal.new(value)
  defp to_decimal(value) when is_float(value), do: Decimal.from_float(value)
  defp to_decimal(%Decimal{} = value), do: value

  defp to_float(value) when is_integer(value), do: value / 1
  defp to_float(value) when is_float(value), do: value
  defp to_float(%Decimal{} = value), do: Decimal.to_float(value)
end
