defmodule MarketplaceWeb.Resolvers.Accounts do
  @moduledoc """
  GraphQL resolvers for account operations.

  Resolvers are thin wrappers around context functions. Business logic stays
  in contexts; resolvers handle GraphQL specific concerns like extracting
  current_user from context and formatting errors for GraphQL responses.

  Authentication uses Guardian tokens. Token is extracted from Authorization
  header in plug and placed in Absinthe context. Resolvers that require
  authentication check for current_user presence.
  """

  alias Marketplace.Accounts

  @doc """
  Returns current authenticated user or nil.
  """
  def me(_, _, %{context: %{current_user: user}}) when not is_nil(user) do
    {:ok, user}
  end

  def me(_, _, _), do: {:ok, nil}

  @doc """
  Fetches user by ID. Public endpoint for viewing seller profiles.
  Hides email field for non self queries.
  """
  def get_user(_, %{id: id}, %{context: context}) do
    case Accounts.get_user(id) do
      nil ->
        {:error, "User not found"}

      user ->
        # Only expose email to the user themselves
        user =
          if context[:current_user] && context[:current_user].id == user.id do
            user
          else
            %{user | email: nil}
          end

        {:ok, user}
    end
  end

  @doc """
  Registers new user and returns authentication token.
  """
  def register(_, args, _) do
    case Accounts.register_user(args) do
      {:ok, user} ->
        token = generate_token(user)
        {:ok, %{token: token, user: user}}

      {:error, changeset} ->
        {:error, format_errors(changeset)}
    end
  end

  @doc """
  Authenticates user and returns token.
  Error messages intentionally vague to prevent account enumeration.
  """
  def login(_, %{email: email, password: password}, _) do
    case Accounts.authenticate(email, password) do
      {:ok, user} ->
        token = generate_token(user)
        {:ok, %{token: token, user: user}}

      {:error, :invalid_credentials} ->
        {:error, "Invalid email or password"}
    end
  end

  @doc """
  Updates profile for authenticated user.
  """
  def update_profile(_, args, %{context: %{current_user: user}}) when not is_nil(user) do
    case Accounts.update_profile(user, args) do
      {:ok, updated_user} ->
        {:ok, updated_user}

      {:error, changeset} ->
        {:error, format_errors(changeset)}
    end
  end

  def update_profile(_, _, _), do: {:error, "Authentication required"}

  # Token generation using Guardian.
  # In production, this would use actual Guardian configuration.
  defp generate_token(user) do
    # Placeholder for Guardian.encode_and_sign
    "token_#{user.id}_#{:os.system_time(:second)}"
  end

  # Formats changeset errors into user friendly messages
  defp format_errors(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {msg, opts} ->
      Enum.reduce(opts, msg, fn {key, value}, acc ->
        String.replace(acc, "%{#{key}}", to_string(value))
      end)
    end)
    |> Enum.map(fn {field, messages} ->
      "#{field}: #{Enum.join(messages, ", ")}"
    end)
    |> Enum.join("; ")
  end
end
