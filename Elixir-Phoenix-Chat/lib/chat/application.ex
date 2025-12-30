defmodule Chat.Application do
  @moduledoc """
  OTP Application entry point.

  Supervision tree defines process hierarchy and restart behavior:

  1. Registry starts first (others depend on it for process lookup)
  2. PubSub starts (channels depend on it for broadcasting)
  3. RoomSupervisor starts (manages room processes)
  4. Endpoint starts last (accepts connections once everything is ready)

  Order matters. If Endpoint started first, incoming connections would
  fail because RoomSupervisor is not ready to create rooms.

  Strategy is :one_for_one because children are independent:
  - Registry crash does not affect PubSub
  - RoomSupervisor crash does not require Endpoint restart

  For tightly coupled processes, :one_for_all or :rest_for_one would
  be appropriate. Not the case here.
  """

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Process registry for room lookup by name
      # Uses ETS table for O(1) lookup performance
      {Registry, keys: :unique, name: Chat.Rooms.Registry},

      # PubSub for channel broadcasting
      # Phoenix.PubSub.PG2 uses Erlang pg module for distributed pubsub
      {Phoenix.PubSub, name: Chat.PubSub},

      # Presence tracker
      ChatWeb.Presence,

      # Dynamic supervisor for room processes
      Chat.Rooms.RoomSupervisor,

      # HTTP/WebSocket endpoint
      ChatWeb.Endpoint
    ]

    opts = [strategy: :one_for_one, name: Chat.Supervisor]
    Supervisor.start_link(children, opts)
  end

  @impl true
  def config_change(changed, _new, removed) do
    ChatWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
