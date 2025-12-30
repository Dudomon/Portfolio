defmodule Chat.Rooms.RoomSupervisor do
  @moduledoc """
  DynamicSupervisor for chat room processes.

  Why DynamicSupervisor instead of regular Supervisor:
  Regular Supervisor starts children at boot from a static list.
  Chat rooms are created dynamically when users request them.
  DynamicSupervisor allows starting children at runtime.

  Restart strategy is :transient because rooms should restart if they
  crash abnormally (error in message handling), but not if they terminate
  normally (room empty for extended period, manual shutdown).

  If we used :permanent, a room that deliberately stops would restart
  immediately, which is wasteful for inactive rooms.
  """

  use DynamicSupervisor
  alias Chat.Rooms.Room

  def start_link(init_arg) do
    DynamicSupervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    DynamicSupervisor.init(strategy: :one_for_one)
  end

  @doc """
  Creates a new room or returns existing room.

  Idempotent: calling create_room("general") twice returns the same
  room process. This simplifies client code; they can always call
  create_room before joining without checking existence.
  """
  def create_room(room_name) do
    case Room.exists?(room_name) do
      true ->
        {:ok, room_name}

      false ->
        child_spec = %{
          id: Room,
          start: {Room, :start_link, [room_name]},
          restart: :transient
        }

        case DynamicSupervisor.start_child(__MODULE__, child_spec) do
          {:ok, _pid} -> {:ok, room_name}
          {:error, {:already_started, _pid}} -> {:ok, room_name}
          error -> error
        end
    end
  end

  @doc """
  Lists all active room names.
  """
  def list_rooms do
    Registry.select(Chat.Rooms.Registry, [{{:"$1", :_, :_}, [], [:"$1"]}])
  end

  @doc """
  Returns count of active rooms.
  """
  def room_count do
    DynamicSupervisor.count_children(__MODULE__).active
  end
end
