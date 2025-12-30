defmodule ChatWeb.RoomChannel do
  @moduledoc """
  Phoenix Channel for chat room communication.

  Channel lifecycle:
  1. Client calls socket.channel("room:general") creating Channel process
  2. join/3 callback authenticates and initializes state
  3. Client sends events via push(), handled by handle_in/3
  4. Server broadcasts via broadcast!/3, received by handle_out/3
  5. Client disconnects, terminate/2 cleans up

  Each channel subscription creates a separate process linked to the socket.
  If channel process crashes, the socket survives and client can rejoin.
  If socket process crashes, all its channels terminate.

  Presence tracking uses Phoenix.Presence CRDT for distributed consistency.
  When user joins on node A and another on node B, both nodes converge
  to the same presence state without central coordination.
  """

  use Phoenix.Channel
  alias Chat.Rooms.{Room, RoomSupervisor}
  alias ChatWeb.Presence

  @impl true
  def join("room:" <> room_name, _params, socket) do
    # Ensure room exists (creates if needed)
    case RoomSupervisor.create_room(room_name) do
      {:ok, _} ->
        # Track presence after successful join
        send(self(), :after_join)

        socket = assign(socket, :room_name, room_name)

        # Return recent messages with join response
        messages = Room.get_messages(room_name)
        {:ok, %{messages: messages}, socket}

      {:error, reason} ->
        {:error, %{reason: reason}}
    end
  end

  @impl true
  def handle_info(:after_join, socket) do
    # Track user presence in room
    {:ok, _} =
      Presence.track(socket, socket.assigns.user_id, %{
        username: socket.assigns.username,
        online_at: System.system_time(:second)
      })

    # Push current presence state to joining user
    push(socket, "presence_state", Presence.list(socket))

    {:noreply, socket}
  end

  @impl true
  def handle_in("new_message", %{"content" => content}, socket) do
    room_name = socket.assigns.room_name
    user_id = socket.assigns.user_id
    username = socket.assigns.username

    case validate_message(content) do
      :ok ->
        {:ok, message} = Room.send_message(room_name, user_id, username, content)

        # Broadcast to all subscribers including sender
        broadcast!(socket, "new_message", message)

        {:noreply, socket}

      {:error, reason} ->
        {:reply, {:error, %{reason: reason}}, socket}
    end
  end

  @impl true
  def handle_in("typing", _params, socket) do
    Room.user_typing(
      socket.assigns.room_name,
      socket.assigns.user_id,
      socket.assigns.username
    )

    {:noreply, socket}
  end

  @impl true
  def handle_in("stop_typing", _params, socket) do
    Room.user_stopped_typing(
      socket.assigns.room_name,
      socket.assigns.user_id
    )

    {:noreply, socket}
  end

  @impl true
  def handle_in("get_typing", _params, socket) do
    users = Room.get_typing_users(socket.assigns.room_name)
    {:reply, {:ok, %{typing: users}}, socket}
  end

  # Intercept outgoing events for transformation or filtering
  @impl true
  def handle_out("new_message", payload, socket) do
    push(socket, "new_message", payload)
    {:noreply, socket}
  end

  @impl true
  def terminate(_reason, socket) do
    # Presence automatically cleans up on process termination
    # Room typing indicator cleanup
    if room_name = socket.assigns[:room_name] do
      Room.user_stopped_typing(room_name, socket.assigns.user_id)
    end

    :ok
  end

  # Private functions

  defp validate_message(content) when is_binary(content) do
    trimmed = String.trim(content)

    cond do
      byte_size(trimmed) == 0 ->
        {:error, "message cannot be empty"}

      byte_size(trimmed) > 2000 ->
        {:error, "message too long (max 2000 characters)"}

      true ->
        :ok
    end
  end

  defp validate_message(_), do: {:error, "invalid message format"}
end
