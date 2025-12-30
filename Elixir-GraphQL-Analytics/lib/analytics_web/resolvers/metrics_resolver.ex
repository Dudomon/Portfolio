defmodule AnalyticsWeb.Resolvers.Metrics do
  @moduledoc """
  GraphQL resolvers for analytics metrics.

  Resolver design:
  - Thin layer between GraphQL and domain logic
  - No business logic here; delegate to contexts
  - Handle authentication/authorization checks
  - Transform context results to GraphQL response format

  Error handling:
  - Return {:error, message} for client errors (invalid input)
  - Let crashes propagate for server errors (they are caught by Absinthe)
  - Log warnings for unexpected but recoverable situations
  """

  alias Analytics.Metrics.Aggregator
  alias Analytics.Reports

  @doc """
  Returns current dashboard metrics from in memory aggregator.
  Fast path: no database queries, sub millisecond response.
  """
  def get_dashboard_metrics(_parent, _args, _resolution) do
    metrics = Aggregator.get_metrics()

    # Add computed change percentages
    {yesterday_metrics, _} = get_comparison_metrics()

    enriched = %{
      active_users: metrics.active_users,
      total_users: metrics.total_users,
      new_users_today: metrics.new_users_today,
      orders_today: metrics.orders_today,
      revenue_today: metrics.revenue_today,
      orders_per_minute: metrics.orders_per_minute,
      active_users_change: calculate_change(metrics.active_users, yesterday_metrics.active_users),
      orders_change: calculate_change(metrics.orders_today, yesterday_metrics.orders),
      revenue_change: calculate_change(metrics.revenue_today, yesterday_metrics.revenue)
    }

    {:ok, enriched}
  end

  @doc """
  Returns hourly activity data points for time series chart.
  """
  def get_hourly_activity(_parent, %{hours: hours}, _resolution) do
    # Get from aggregator for current data
    all_activity = Aggregator.get_hourly_activity()

    # Limit to requested hours
    current_hour = DateTime.utc_now().hour

    data_points =
      all_activity
      |> Enum.map(fn %{hour: hour, count: count} ->
        %{
          hour: hour,
          count: count,
          timestamp: hour_to_timestamp(hour)
        }
      end)
      |> rotate_to_current_hour(current_hour)
      |> Enum.take(-hours)

    {:ok, data_points}
  end

  @doc """
  Returns daily aggregated metrics for date range.
  Queries historical data from database.
  """
  def get_daily_metrics(_parent, %{start_date: start_date, end_date: end_date}, _resolution) do
    # Validate date range
    max_days = 365

    case Date.diff(end_date, start_date) do
      diff when diff < 0 ->
        {:error, "end_date must be after start_date"}

      diff when diff > max_days ->
        {:error, "date range cannot exceed #{max_days} days"}

      _ ->
        metrics = Reports.get_daily_metrics(start_date, end_date)
        {:ok, metrics}
    end
  end

  @doc """
  Returns revenue breakdown by category for the specified period.
  """
  def get_revenue_by_category(_parent, %{period: period}, _resolution) do
    {start_date, end_date} = period_to_date_range(period)
    breakdown = Reports.get_revenue_by_category(start_date, end_date)

    # Calculate percentages
    total = Enum.reduce(breakdown, Decimal.new(0), fn cat, acc ->
      Decimal.add(acc, cat.revenue)
    end)

    enriched = Enum.map(breakdown, fn cat ->
      percentage =
        if Decimal.compare(total, Decimal.new(0)) == :gt do
          cat.revenue
          |> Decimal.div(total)
          |> Decimal.mult(100)
          |> Decimal.to_float()
          |> Float.round(2)
        else
          0.0
        end

      Map.put(cat, :percentage, percentage)
    end)

    {:ok, enriched}
  end

  @doc """
  Returns top performing products by sales volume.
  """
  def get_top_products(_parent, %{limit: limit, period: period}, _resolution) do
    {start_date, end_date} = period_to_date_range(period)

    products =
      Reports.get_top_products(start_date, end_date, limit)
      |> Enum.map(fn product ->
        conversion_rate =
          if product.view_count > 0 do
            Float.round(product.total_sales / product.view_count * 100, 2)
          else
            0.0
          end

        Map.put(product, :conversion_rate, conversion_rate)
      end)

    {:ok, products}
  end

  @doc """
  Records a custom analytics event.
  """
  def record_event(_parent, %{event_type: event_type, properties: properties}, _resolution) do
    case Analytics.Events.record(event_type, properties || %{}) do
      {:ok, event_id} ->
        {:ok, %{success: true, event_id: event_id}}

      {:error, reason} ->
        {:ok, %{success: false, error: to_string(reason)}}
    end
  end

  # Private functions

  defp get_comparison_metrics do
    # In production, query yesterday's snapshot from database
    # For demo, return placeholder
    yesterday = %{
      active_users: 0,
      orders: 0,
      revenue: Decimal.new(0)
    }

    {yesterday, Date.add(Date.utc_today(), -1)}
  end

  defp calculate_change(current, previous) when is_number(current) and is_number(previous) do
    if previous > 0 do
      Float.round((current - previous) / previous * 100, 2)
    else
      nil
    end
  end

  defp calculate_change(%Decimal{} = current, %Decimal{} = previous) do
    if Decimal.compare(previous, Decimal.new(0)) == :gt do
      current
      |> Decimal.sub(previous)
      |> Decimal.div(previous)
      |> Decimal.mult(100)
      |> Decimal.to_float()
      |> Float.round(2)
    else
      nil
    end
  end

  defp calculate_change(_, _), do: nil

  defp period_to_date_range(period) do
    today = Date.utc_today()

    start_date =
      case period do
        :day -> today
        :week -> Date.add(today, -7)
        :month -> Date.add(today, -30)
        :quarter -> Date.add(today, -90)
        :year -> Date.add(today, -365)
      end

    {start_date, today}
  end

  defp hour_to_timestamp(hour) do
    today = Date.utc_today()
    {:ok, datetime} = NaiveDateTime.new(today, Time.new!(hour, 0, 0))
    DateTime.from_naive!(datetime, "Etc/UTC")
  end

  defp rotate_to_current_hour(data_points, current_hour) do
    # Rotate list so current hour is last
    {before_current, from_current} = Enum.split_while(data_points, fn %{hour: h} ->
      h <= current_hour
    end)

    from_current ++ before_current
  end
end
