defmodule AnalyticsWeb.Schema do
  @moduledoc """
  GraphQL schema for analytics dashboard.

  Query complexity limits prevent expensive queries from overwhelming the server.
  Each field has a complexity cost; queries exceeding 200 total cost are rejected.

  Complexity calculation:
  - Scalar fields: 1
  - Object fields: 5 + child complexity
  - List fields: 10 * limit + child complexity

  Example: Fetching 100 products with 5 fields each = 10*100 + 100*5 = 1500
  This exceeds the limit, forcing client to paginate or reduce fields.

  Subscriptions broadcast only meaningful changes. The MetricsAggregator
  filters minor fluctuations to prevent subscription spam.
  """

  use Absinthe.Schema
  import_types AnalyticsWeb.Schema.Types
  import_types Absinthe.Type.Custom

  alias AnalyticsWeb.Resolvers

  @max_complexity 200

  query do
    @desc "Get current dashboard metrics"
    field :dashboard_metrics, :dashboard_metrics do
      complexity 10
      resolve &Resolvers.Metrics.get_dashboard_metrics/3
    end

    @desc "Get hourly activity for chart display"
    field :hourly_activity, list_of(:hourly_data_point) do
      arg :hours, :integer, default_value: 24
      complexity fn args, child_complexity ->
        args.hours * child_complexity
      end
      resolve &Resolvers.Metrics.get_hourly_activity/3
    end

    @desc "Get daily metrics for a date range"
    field :daily_metrics, list_of(:daily_metrics) do
      arg :start_date, non_null(:date)
      arg :end_date, non_null(:date)
      complexity fn args, child_complexity ->
        days = Date.diff(args.end_date, args.start_date)
        max(days, 1) * child_complexity
      end
      resolve &Resolvers.Metrics.get_daily_metrics/3
    end

    @desc "Get revenue breakdown by category"
    field :revenue_by_category, list_of(:category_revenue) do
      arg :period, :period, default_value: :month
      complexity 20
      resolve &Resolvers.Metrics.get_revenue_by_category/3
    end

    @desc "Get top performing products"
    field :top_products, list_of(:product_stats) do
      arg :limit, :integer, default_value: 10
      arg :period, :period, default_value: :week
      complexity fn args, child_complexity ->
        args.limit * child_complexity
      end
      resolve &Resolvers.Metrics.get_top_products/3
    end
  end

  mutation do
    @desc "Record a custom event for analytics"
    field :record_event, :event_result do
      arg :event_type, non_null(:string)
      arg :properties, :json
      resolve &Resolvers.Metrics.record_event/3
    end
  end

  subscription do
    @desc "Subscribe to real time metric updates"
    field :metrics_updated, :metric_update do
      config fn _, _ ->
        {:ok, topic: "metrics:updates"}
      end

      resolve fn payload, _, _ ->
        {:ok, payload}
      end
    end

    @desc "Subscribe to specific metric changes"
    field :metric_changed, :metric_update do
      arg :metric_name, non_null(:metric_name)

      config fn args, _ ->
        {:ok, topic: "metrics:#{args.metric_name}"}
      end
    end
  end

  # Middleware for query complexity
  def middleware(middleware, _field, %{identifier: :query}) do
    middleware ++ [Absinthe.Middleware.CheckComplexity]
  end

  def middleware(middleware, _field, _object) do
    middleware
  end

  def plugins do
    [Absinthe.Middleware.Dataloader] ++ Absinthe.Plugin.defaults()
  end

  def context(ctx) do
    loader =
      Dataloader.new()
      |> Dataloader.add_source(Analytics.Catalog, Analytics.Catalog.data())

    Map.put(ctx, :loader, loader)
  end

  # Complexity configuration
  def complexity_config do
    %{
      max_complexity: @max_complexity
    }
  end
end
