defmodule ChatWeb.UserSocket do
  @moduledoc """
  WebSocket entry point for chat connections.

  Each browser tab opens one WebSocket connection managed by this socket.
  The socket can subscribe to multiple channels (rooms) over the same
  connection, avoiding the overhead of multiple TCP handshakes.

  Authentication happens during socket connection via token in params.
  The token is verified once at connection time; subsequent channel
  operations trust the socket assigns without re verifying.

  Token format uses Phoenix.Token for signed, expiring tokens.
  Unlike JWT, Phoenix.Token uses server side secret and can be invalidated
  by changing the secret. Simpler for single server deployments.
  """

  use Phoenix.Socket

  channel "room:*", ChatWeb.RoomChannel

  @max_age 86400 # 24 hours

  @impl true
  def connect(%{"token" => token}, socket, _connect_info) do
    case verify_token(token) do
      {:ok, user_data} ->
        socket =
          socket
          |> assign(:user_id, user_data.user_id)
          |> assign(:username, user_data.username)

        {:ok, socket}

      {:error, _reason} ->
        :error
    end
  end

  def connect(_params, _socket, _connect_info) do
    # No token provided
    :error
  end

  @impl true
  def id(socket), do: "user_socket:#{socket.assigns.user_id}"

  defp verify_token(token) do
    case Phoenix.Token.verify(ChatWeb.Endpoint, "user socket", token, max_age: @max_age) do
      {:ok, user_data} ->
        {:ok, user_data}

      {:error, :expired} ->
        {:error, :token_expired}

      {:error, :invalid} ->
        {:error, :invalid_token}
    end
  end
end
