defmodule ChatWeb.Presence do
  @moduledoc """
  Presence tracking for online users in chat rooms.

  Phoenix.Presence uses CRDTs (Conflict free Replicated Data Types) for
  distributed presence tracking. This means:

  1. No single point of failure: presence works even if some nodes are down
  2. Eventual consistency: all nodes converge to same state without coordination
  3. Automatic cleanup: when a process dies, its presence is removed

  How it works internally:
  - Each node maintains local presence state
  - Nodes gossip state changes via PubSub
  - CRDT merge function combines states deterministically
  - Heartbeat detects dead nodes and removes their presence

  Presence diffs are sent to clients on join/leave events:
  - "presence_diff" with %{joins: %{}, leaves: %{}}

  This is more efficient than sending full presence list on every change.
  For a room with 1000 users, a join sends 1 user diff, not 1001 user list.
  """

  use Phoenix.Presence,
    otp_app: :chat,
    pubsub_server: Chat.PubSub

  @doc """
  Fetches presence data and transforms for client consumption.

  Default implementation returns raw presence data. Override to add
  computed fields or filter sensitive data.
  """
  def fetch(_topic, presences) do
    # Add computed fields to each presence
    for {key, %{metas: metas}} <- presences, into: %{} do
      {key,
       %{
         metas:
           Enum.map(metas, fn meta ->
             Map.put(meta, :status, compute_status(meta))
           end)
       }}
    end
  end

  defp compute_status(meta) do
    seconds_since_join = System.system_time(:second) - meta.online_at

    cond do
      seconds_since_join < 300 -> "active"
      true -> "idle"
    end
  end
end
