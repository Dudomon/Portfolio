defmodule AnalyticsWeb.Schema.Types do
  @moduledoc """
  GraphQL type definitions for analytics dashboard.

  Type design follows these principles:

  1. Scalar types for atomic values (counts, amounts, timestamps)
  2. Object types group related scalars (dashboard metrics bundle)
  3. Enums for finite option sets (period: day/week/month/year)
  4. Input types for complex arguments (date range filters)

  Nullability is explicit:
  - Non null for required data that always exists
  - Nullable for optional or computed fields that might fail
  """

  use Absinthe.Schema.Notation

  # Enums

  enum :period do
    value :day, description: "Last 24 hours"
    value :week, description: "Last 7 days"
    value :month, description: "Last 30 days"
    value :quarter, description: "Last 90 days"
    value :year, description: "Last 365 days"
  end

  enum :metric_name do
    value :active_users
    value :total_users
    value :new_users_today
    value :orders_today
    value :revenue_today
    value :orders_per_minute
  end

  # Scalars

  scalar :decimal do
    parse fn
      %Absinthe.Blueprint.Input.String{value: value}, _ ->
        Decimal.parse(value)

      %Absinthe.Blueprint.Input.Float{value: value}, _ ->
        {:ok, Decimal.from_float(value)}

      %Absinthe.Blueprint.Input.Integer{value: value}, _ ->
        {:ok, Decimal.new(value)}

      _, _ ->
        :error
    end

    serialize fn decimal ->
      Decimal.to_string(decimal)
    end
  end

  scalar :json do
    parse fn
      %Absinthe.Blueprint.Input.String{value: value}, _ ->
        case Jason.decode(value) do
          {:ok, result} -> {:ok, result}
          _ -> :error
        end

      %Absinthe.Blueprint.Input.Null{}, _ ->
        {:ok, nil}

      _, _ ->
        :error
    end

    serialize fn value ->
      value
    end
  end

  # Object types

  @desc "Main dashboard metrics bundle"
  object :dashboard_metrics do
    field :active_users, non_null(:integer), description: "Currently active users"
    field :total_users, non_null(:integer), description: "Total registered users"
    field :new_users_today, non_null(:integer), description: "Users registered today"
    field :orders_today, non_null(:integer), description: "Orders completed today"
    field :revenue_today, non_null(:decimal), description: "Revenue generated today"
    field :orders_per_minute, non_null(:float), description: "Current order rate"

    # Computed comparison fields
    field :active_users_change, :float, description: "Percent change from yesterday"
    field :orders_change, :float, description: "Percent change from yesterday"
    field :revenue_change, :float, description: "Percent change from yesterday"
  end

  @desc "Single data point for time series charts"
  object :hourly_data_point do
    field :hour, non_null(:integer), description: "Hour of day (0-23)"
    field :count, non_null(:integer), description: "Event count for this hour"
    field :timestamp, :datetime, description: "Start of hour as timestamp"
  end

  @desc "Daily aggregated metrics"
  object :daily_metrics do
    field :date, non_null(:date)
    field :total_users, non_null(:integer)
    field :new_users, non_null(:integer)
    field :active_users, non_null(:integer)
    field :orders, non_null(:integer)
    field :revenue, non_null(:decimal)
    field :average_order_value, :decimal
  end

  @desc "Revenue breakdown by product category"
  object :category_revenue do
    field :category_id, non_null(:id)
    field :category_name, non_null(:string)
    field :revenue, non_null(:decimal)
    field :order_count, non_null(:integer)
    field :percentage, non_null(:float), description: "Percentage of total revenue"
  end

  @desc "Product performance statistics"
  object :product_stats do
    field :product_id, non_null(:id)
    field :product_name, non_null(:string)
    field :total_sales, non_null(:integer)
    field :total_revenue, non_null(:decimal)
    field :view_count, non_null(:integer)
    field :conversion_rate, :float, description: "Sales / Views percentage"
  end

  @desc "Result of recording a custom event"
  object :event_result do
    field :success, non_null(:boolean)
    field :event_id, :string
    field :error, :string
  end

  @desc "Real time metric update payload"
  object :metric_update do
    field :metric, non_null(:metric_name)
    field :value, non_null(:string), description: "JSON encoded value"
    field :timestamp, non_null(:datetime)
    field :change, :float, description: "Percent change from previous"
  end
end
