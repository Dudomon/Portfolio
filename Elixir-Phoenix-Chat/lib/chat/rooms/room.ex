defmodule Chat.Rooms.Room do
  @moduledoc """
  GenServer managing state for a single chat room.

  Each room runs as an isolated process. This isolation provides:

  1. Fault containment: If message processing crashes (malformed data,
     unexpected pattern), only this room's process dies. The supervisor
     restarts it with fresh state. Other rooms continue unaffected.

  2. Concurrency: Room processes run in parallel across CPU cores.
     A message in "general" room does not block message processing
     in "elixir" room, even on a single core machine.

  3. State encapsulation: Room state (messages, typing users) is private
     to this process. No global state, no race conditions, no locks.

  Message history is kept in process memory for simplicity. Production
  would persist to database and load recent messages on room start.
  The message limit prevents unbounded memory growth.
  """

  use GenServer
  require Logger

  @max_messages 100
  @typing_timeout_ms 3000

  defstruct [
    :name,
    messages: [],
    typing: %{},
    created_at: nil
  ]

  @type message :: %{
          id: String.t(),
          user_id: String.t(),
          username: String.t(),
          content: String.t(),
          timestamp: DateTime.t()
        }

  @type t :: %__MODULE__{
          name: String.t(),
          messages: [message()],
          typing: %{String.t() => reference()},
          created_at: DateTime.t()
        }

  # Client API

  @doc """
  Starts a room process with the given name.
  Rooms are registered via Chat.Rooms.Registry for lookup by name.
  """
  def start_link(name) do
    GenServer.start_link(__MODULE__, name, name: via_tuple(name))
  end

  @doc """
  Sends a message to the room. Returns the message with generated ID.
  """
  def send_message(room_name, user_id, username, content) do
    GenServer.call(via_tuple(room_name), {:send_message, user_id, username, content})
  end

  @doc """
  Retrieves recent messages. Returns up to @max_messages in chronological order.
  """
  def get_messages(room_name) do
    GenServer.call(via_tuple(room_name), :get_messages)
  end

  @doc """
  Marks user as typing. Automatically clears after timeout.
  Called on each keystroke; GenServer handles debouncing internally.
  """
  def user_typing(room_name, user_id, username) do
    GenServer.cast(via_tuple(room_name), {:user_typing, user_id, username})
  end

  @doc """
  Explicitly clears typing indicator when user sends message or stops typing.
  """
  def user_stopped_typing(room_name, user_id) do
    GenServer.cast(via_tuple(room_name), {:user_stopped_typing, user_id})
  end

  @doc """
  Returns list of currently typing users.
  """
  def get_typing_users(room_name) do
    GenServer.call(via_tuple(room_name), :get_typing_users)
  end

  @doc """
  Checks if room process exists.
  """
  def exists?(room_name) do
    case Registry.lookup(Chat.Rooms.Registry, room_name) do
      [{_pid, _}] -> true
      [] -> false
    end
  end

  # Server callbacks

  @impl true
  def init(name) do
    Logger.info("Room #{name} started")

    state = %__MODULE__{
      name: name,
      created_at: DateTime.utc_now()
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:send_message, user_id, username, content}, _from, state) do
    message = %{
      id: generate_id(),
      user_id: user_id,
      username: username,
      content: content,
      timestamp: DateTime.utc_now()
    }

    # Add message and trim to max
    messages =
      [message | state.messages]
      |> Enum.take(@max_messages)

    # Clear typing indicator since user sent a message
    state = clear_typing(state, user_id)

    # Broadcast to all subscribers via PubSub
    broadcast_message(state.name, message)

    {:reply, {:ok, message}, %{state | messages: messages}}
  end

  @impl true
  def handle_call(:get_messages, _from, state) do
    # Return in chronological order (oldest first)
    messages = Enum.reverse(state.messages)
    {:reply, messages, state}
  end

  @impl true
  def handle_call(:get_typing_users, _from, state) do
    users = Map.keys(state.typing)
    {:reply, users, state}
  end

  @impl true
  def handle_cast({:user_typing, user_id, username}, state) do
    # Cancel existing timer for this user
    state = clear_typing(state, user_id)

    # Set new timer
    timer_ref = Process.send_after(self(), {:typing_timeout, user_id}, @typing_timeout_ms)

    typing = Map.put(state.typing, user_id, {username, timer_ref})

    broadcast_typing(state.name, user_id, username, true)

    {:noreply, %{state | typing: typing}}
  end

  @impl true
  def handle_cast({:user_stopped_typing, user_id}, state) do
    if Map.has_key?(state.typing, user_id) do
      {username, _} = Map.get(state.typing, user_id)
      state = clear_typing(state, user_id)
      broadcast_typing(state.name, user_id, username, false)
      {:noreply, state}
    else
      {:noreply, state}
    end
  end

  @impl true
  def handle_info({:typing_timeout, user_id}, state) do
    if Map.has_key?(state.typing, user_id) do
      {username, _} = Map.get(state.typing, user_id)
      typing = Map.delete(state.typing, user_id)
      broadcast_typing(state.name, user_id, username, false)
      {:noreply, %{state | typing: typing}}
    else
      {:noreply, state}
    end
  end

  @impl true
  def terminate(reason, state) do
    Logger.info("Room #{state.name} terminating: #{inspect(reason)}")
    :ok
  end

  # Private functions

  defp via_tuple(room_name) do
    {:via, Registry, {Chat.Rooms.Registry, room_name}}
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp clear_typing(state, user_id) do
    case Map.get(state.typing, user_id) do
      {_username, timer_ref} ->
        Process.cancel_timer(timer_ref)
        %{state | typing: Map.delete(state.typing, user_id)}

      nil ->
        state
    end
  end

  defp broadcast_message(room_name, message) do
    Phoenix.PubSub.broadcast(
      Chat.PubSub,
      "room:#{room_name}",
      {:new_message, message}
    )
  end

  defp broadcast_typing(room_name, user_id, username, is_typing) do
    Phoenix.PubSub.broadcast(
      Chat.PubSub,
      "room:#{room_name}",
      {:typing, %{user_id: user_id, username: username, is_typing: is_typing}}
    )
  end
end
