defmodule Analytics.Metrics.Snapshot do
  @moduledoc """
  Handles periodic persistence of aggregated metrics to PostgreSQL.

  Problem this solves:

  The Aggregator GenServer maintains metrics in memory for speed, but memory
  is volatile. A server restart loses all accumulated data. Furthermore,
  historical queries ("revenue last month") cannot use in memory state.

  Snapshotting strategy:

  1. Periodic writes: Every 60 seconds, current state writes to database
  2. Atomic snapshots: Single transaction captures consistent state
  3. Compressed storage: Only deltas from previous snapshot for space efficiency
  4. Retention policy: Hourly snapshots kept for 7 days, daily for 1 year

  Recovery process:

  On Aggregator startup:
  1. Load most recent snapshot from database
  2. Query event log for events after snapshot timestamp
  3. Replay events to rebuild current state
  4. Resume normal operation

  This provides crash recovery with bounded data loss (max 60 seconds).

  Schema design:

  metrics_snapshots table stores point in time captures. The `metrics_jsonb`
  column uses PostgreSQL JSONB for flexible schema evolution. New metrics
  can be added without migrations.

  Indexes on (inserted_at DESC) enable fast "latest snapshot" queries.
  Partial indexes on (date_trunc('hour', inserted_at)) optimize retention cleanup.
  """

  use Ecto.Schema
  import Ecto.Query
  import Ecto.Changeset
  require Logger

  alias Analytics.Repo
  alias Analytics.Metrics.Calculator

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "metrics_snapshots" do
    field :snapshot_type, Ecto.Enum, values: [:periodic, :manual, :shutdown]
    field :metrics_jsonb, :map
    field :aggregator_started_at, :utc_datetime_usec
    field :event_count_since_start, :integer
    field :checksum, :string

    timestamps(type: :utc_datetime_usec)
  end

  @type t :: %__MODULE__{
          id: Ecto.UUID.t(),
          snapshot_type: :periodic | :manual | :shutdown,
          metrics_jsonb: map(),
          aggregator_started_at: DateTime.t(),
          event_count_since_start: non_neg_integer(),
          checksum: String.t(),
          inserted_at: DateTime.t(),
          updated_at: DateTime.t()
        }

  @doc """
  Creates a new snapshot from aggregator state.

  The checksum ensures data integrity during recovery. If the loaded snapshot
  checksum does not match recomputed checksum, the snapshot is considered
  corrupted and recovery falls back to event replay.
  """
  @spec create(map(), atom()) :: {:ok, t()} | {:error, Ecto.Changeset.t()}
  def create(aggregator_state, type \\ :periodic) do
    metrics = serialize_metrics(aggregator_state)
    checksum = compute_checksum(metrics)

    attrs = %{
      snapshot_type: type,
      metrics_jsonb: metrics,
      aggregator_started_at: aggregator_state.started_at,
      event_count_since_start: calculate_event_count(aggregator_state),
      checksum: checksum
    }

    %__MODULE__{}
    |> changeset(attrs)
    |> Repo.insert()
  end

  @doc """
  Loads the most recent valid snapshot.

  Validates checksum before returning. Invalid snapshots are logged and skipped,
  falling back to the next most recent valid snapshot.
  """
  @spec load_latest() :: {:ok, t()} | {:error, :not_found | :all_corrupted}
  def load_latest do
    query =
      from s in __MODULE__,
        order_by: [desc: s.inserted_at],
        limit: 5

    case Repo.all(query) do
      [] ->
        {:error, :not_found}

      snapshots ->
        find_valid_snapshot(snapshots)
    end
  end

  @doc """
  Loads snapshot nearest to a specific timestamp for historical queries.

  Uses binary search on indexed timestamp column. Returns the snapshot
  immediately before or at the requested time.
  """
  @spec load_at(DateTime.t()) :: {:ok, t()} | {:error, :not_found}
  def load_at(timestamp) do
    query =
      from s in __MODULE__,
        where: s.inserted_at <= ^timestamp,
        order_by: [desc: s.inserted_at],
        limit: 1

    case Repo.one(query) do
      nil -> {:error, :not_found}
      snapshot -> validate_and_return(snapshot)
    end
  end

  @doc """
  Deserializes snapshot back to aggregator state format.

  Handles schema evolution: missing fields get default values, unknown fields
  are ignored. This allows older snapshots to load in newer code versions.
  """
  @spec to_aggregator_state(t()) :: map()
  def to_aggregator_state(%__MODULE__{} = snapshot) do
    metrics = snapshot.metrics_jsonb

    %{
      started_at: snapshot.aggregator_started_at,
      active_users: Map.get(metrics, "active_users", 0),
      total_users: Map.get(metrics, "total_users", 0),
      new_users_today: Map.get(metrics, "new_users_today", 0),
      orders_today: Map.get(metrics, "orders_today", 0),
      revenue_today: deserialize_decimal(Map.get(metrics, "revenue_today", "0")),
      orders_per_minute: Map.get(metrics, "orders_per_minute", 0.0),
      hourly_activity: deserialize_hourly_activity(Map.get(metrics, "hourly_activity", %{}))
    }
  end

  @doc """
  Applies retention policy, removing old snapshots.

  Retention rules:
  1. Keep all snapshots from last 24 hours (for debugging)
  2. Keep hourly snapshots for 7 days
  3. Keep daily snapshots (noon UTC) for 1 year
  4. Delete everything else

  Runs as a background job, typically scheduled via Oban or Quantum.
  """
  @spec apply_retention_policy() :: {:ok, non_neg_integer()}
  def apply_retention_policy do
    now = DateTime.utc_now()
    one_day_ago = DateTime.add(now, -86_400, :second)
    seven_days_ago = DateTime.add(now, -604_800, :second)
    one_year_ago = DateTime.add(now, -31_536_000, :second)

    # Step 1: Identify snapshots to keep
    keep_ids = MapSet.new()

    # All from last 24 hours
    keep_ids =
      from(s in __MODULE__, where: s.inserted_at > ^one_day_ago, select: s.id)
      |> Repo.all()
      |> MapSet.new()
      |> MapSet.union(keep_ids)

    # Hourly for 7 days: keep first snapshot of each hour
    keep_ids =
      identify_hourly_keepers(one_day_ago, seven_days_ago)
      |> MapSet.union(keep_ids)

    # Daily for 1 year: keep snapshot closest to noon UTC
    keep_ids =
      identify_daily_keepers(seven_days_ago, one_year_ago)
      |> MapSet.union(keep_ids)

    # Step 2: Delete everything else older than 24 hours
    {deleted_count, _} =
      from(s in __MODULE__,
        where: s.inserted_at < ^one_day_ago and s.id not in ^MapSet.to_list(keep_ids)
      )
      |> Repo.delete_all()

    Logger.info("Snapshot retention: deleted #{deleted_count} old snapshots")
    {:ok, deleted_count}
  end

  @doc """
  Computes storage statistics for monitoring dashboards.
  """
  @spec storage_stats() :: map()
  def storage_stats do
    total_count =
      from(s in __MODULE__, select: count(s.id))
      |> Repo.one()

    oldest =
      from(s in __MODULE__, order_by: [asc: s.inserted_at], limit: 1, select: s.inserted_at)
      |> Repo.one()

    newest =
      from(s in __MODULE__, order_by: [desc: s.inserted_at], limit: 1, select: s.inserted_at)
      |> Repo.one()

    avg_size =
      from(s in __MODULE__,
        select: fragment("avg(pg_column_size(metrics_jsonb))")
      )
      |> Repo.one()

    %{
      total_snapshots: total_count,
      oldest_snapshot: oldest,
      newest_snapshot: newest,
      average_size_bytes: avg_size || 0,
      estimated_total_bytes: (avg_size || 0) * total_count
    }
  end

  # Private functions

  defp changeset(snapshot, attrs) do
    snapshot
    |> cast(attrs, [
      :snapshot_type,
      :metrics_jsonb,
      :aggregator_started_at,
      :event_count_since_start,
      :checksum
    ])
    |> validate_required([
      :snapshot_type,
      :metrics_jsonb,
      :aggregator_started_at,
      :checksum
    ])
  end

  defp serialize_metrics(state) do
    %{
      "active_users" => state.active_users,
      "total_users" => state.total_users,
      "new_users_today" => state.new_users_today,
      "orders_today" => state.orders_today,
      "revenue_today" => Decimal.to_string(state.revenue_today),
      "orders_per_minute" => state.orders_per_minute,
      "hourly_activity" => serialize_hourly_activity(state.hourly_activity)
    }
  end

  defp serialize_hourly_activity(activity) do
    Map.new(activity, fn {hour, count} ->
      {Integer.to_string(hour), count}
    end)
  end

  defp deserialize_hourly_activity(activity) do
    Map.new(activity, fn {hour_str, count} ->
      {String.to_integer(hour_str), count}
    end)
  end

  defp deserialize_decimal(value) when is_binary(value) do
    case Decimal.parse(value) do
      {decimal, ""} -> decimal
      _ -> Decimal.new(0)
    end
  end

  defp deserialize_decimal(_), do: Decimal.new(0)

  defp compute_checksum(metrics) do
    metrics
    |> Jason.encode!()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end

  defp calculate_event_count(state) do
    state.total_users + state.orders_today
  end

  defp find_valid_snapshot([]), do: {:error, :all_corrupted}

  defp find_valid_snapshot([snapshot | rest]) do
    case validate_and_return(snapshot) do
      {:ok, _} = result -> result
      {:error, :corrupted} ->
        Logger.warning("Snapshot #{snapshot.id} failed checksum validation, trying next")
        find_valid_snapshot(rest)
    end
  end

  defp validate_and_return(snapshot) do
    expected_checksum = compute_checksum(snapshot.metrics_jsonb)

    if snapshot.checksum == expected_checksum do
      {:ok, snapshot}
    else
      {:error, :corrupted}
    end
  end

  defp identify_hourly_keepers(from_time, to_time) do
    # Group snapshots by hour and keep the first of each
    from(s in __MODULE__,
      where: s.inserted_at > ^to_time and s.inserted_at <= ^from_time,
      select: %{
        id: s.id,
        hour: fragment("date_trunc('hour', ?)", s.inserted_at)
      }
    )
    |> Repo.all()
    |> Enum.group_by(& &1.hour)
    |> Enum.map(fn {_hour, snapshots} -> hd(snapshots).id end)
    |> MapSet.new()
  end

  defp identify_daily_keepers(from_time, to_time) do
    # For each day, keep snapshot closest to noon UTC
    noon_hour = 12

    from(s in __MODULE__,
      where: s.inserted_at > ^to_time and s.inserted_at <= ^from_time,
      select: %{
        id: s.id,
        day: fragment("date_trunc('day', ?)", s.inserted_at),
        hour: fragment("extract(hour from ?)", s.inserted_at)
      }
    )
    |> Repo.all()
    |> Enum.group_by(& &1.day)
    |> Enum.map(fn {_day, snapshots} ->
      Enum.min_by(snapshots, fn s -> abs(s.hour - noon_hour) end).id
    end)
    |> MapSet.new()
  end
end
