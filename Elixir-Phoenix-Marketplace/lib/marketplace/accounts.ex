defmodule Marketplace.Accounts do
  @moduledoc """
  Public API for user account operations.

  This context owns all user data and authentication logic. Other contexts
  must use these functions rather than querying the users table directly.

  Authentication uses Guardian for JWT token generation. Tokens expire after
  24 hours and contain minimal claims (user_id only). User permissions are
  loaded fresh on each request to ensure immediate revocation capability.
  """

  import Ecto.Query
  alias Marketplace.Repo
  alias Marketplace.Accounts.User

  @doc """
  Registers a new user account.

  Returns {:ok, user} on success or {:error, changeset} with validation errors.
  Passwords are hashed before storage; plaintext is never persisted.
  """
  def register_user(attrs) do
    %User{}
    |> User.registration_changeset(attrs)
    |> Repo.insert()
  end

  @doc """
  Authenticates user by email and password.

  Performs constant time comparison even for non existent emails to prevent
  timing attacks that could enumerate valid accounts.
  """
  def authenticate(email, password) do
    user = Repo.get_by(User, email: String.downcase(email))

    case user do
      nil ->
        # Perform dummy check to maintain constant time
        Bcrypt.no_user_verify()
        {:error, :invalid_credentials}

      user ->
        if Bcrypt.verify_pass(password, user.password_hash) do
          {:ok, user}
        else
          {:error, :invalid_credentials}
        end
    end
  end

  @doc """
  Retrieves user by ID. Returns nil if not found.
  """
  def get_user(id), do: Repo.get(User, id)

  @doc """
  Retrieves user by ID. Raises if not found.
  Use when caller expects user to exist (e.g., from authenticated session).
  """
  def get_user!(id), do: Repo.get!(User, id)

  @doc """
  Retrieves user by email. Used for password reset flows.
  """
  def get_user_by_email(email) do
    Repo.get_by(User, email: String.downcase(email))
  end

  @doc """
  Updates user profile fields (display name, bio, avatar).
  Does not allow email or password changes; those have separate secure flows.
  """
  def update_profile(user, attrs) do
    user
    |> User.profile_changeset(attrs)
    |> Repo.update()
  end

  @doc """
  Updates seller rating after buyer leaves review.
  Uses incremental calculation to avoid full recalculation.
  """
  def update_seller_rating(seller, rating) when is_integer(rating) and rating in 1..5 do
    seller
    |> User.update_rating_changeset(Decimal.new(rating))
    |> Repo.update()
  end

  @doc """
  Marks user as verified seller. Called by admin after identity verification.
  """
  def verify_seller(user) do
    user
    |> Ecto.Changeset.change(verified_seller: true)
    |> Repo.update()
  end

  @doc """
  Lists users with pagination. Admin only function.
  """
  def list_users(opts \\ []) do
    page = Keyword.get(opts, :page, 1)
    per_page = Keyword.get(opts, :per_page, 20)
    offset = (page - 1) * per_page

    User
    |> order_by(desc: :inserted_at)
    |> limit(^per_page)
    |> offset(^offset)
    |> Repo.all()
  end

  @doc """
  Counts total users for pagination metadata.
  """
  def count_users do
    Repo.aggregate(User, :count, :id)
  end

  @doc """
  Dataloader source for batched user loading in GraphQL resolvers.
  Prevents N+1 queries when loading seller info for product lists.
  """
  def data do
    Dataloader.Ecto.new(Repo, query: &query/2)
  end

  def query(queryable, _params) do
    queryable
  end
end
