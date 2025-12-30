defmodule Marketplace.Accounts.User do
  @moduledoc """
  User schema representing marketplace participants.

  Users can act as buyers, sellers, or both. Role enforcement happens at the
  resolver level, not here. A user becomes a seller by listing their first
  product, not by toggling a role flag.

  Password handling uses bcrypt with cost factor 12. Higher costs provide
  diminishing security returns while linearly increasing login latency.
  At cost 12, hashing takes ~300ms which is acceptable for login but
  would be problematic if we hashed on every request.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @type t :: %__MODULE__{
          id: Ecto.UUID.t(),
          email: String.t(),
          password_hash: String.t(),
          display_name: String.t(),
          avatar_url: String.t() | nil,
          bio: String.t() | nil,
          verified_seller: boolean(),
          rating: Decimal.t(),
          rating_count: integer(),
          inserted_at: NaiveDateTime.t(),
          updated_at: NaiveDateTime.t()
        }

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "users" do
    field :email, :string
    field :password, :string, virtual: true, redact: true
    field :password_hash, :string, redact: true
    field :display_name, :string
    field :avatar_url, :string
    field :bio, :string
    field :verified_seller, :boolean, default: false
    field :rating, :decimal, default: Decimal.new("0.0")
    field :rating_count, :integer, default: 0

    has_many :products, Marketplace.Catalog.Product, foreign_key: :seller_id
    has_many :orders, Marketplace.Orders.Order, foreign_key: :buyer_id

    timestamps()
  end

  @doc """
  Registration changeset validates email format and password strength.

  Password requirements: minimum 8 characters. No complexity rules.
  Research shows length matters more than complexity, and complexity rules
  lead users to write passwords on sticky notes.
  """
  def registration_changeset(user, attrs) do
    user
    |> cast(attrs, [:email, :password, :display_name])
    |> validate_required([:email, :password, :display_name])
    |> validate_email()
    |> validate_password()
    |> hash_password()
  end

  @doc """
  Profile changeset for non sensitive field updates.
  Email and password changes go through separate flows with verification.
  """
  def profile_changeset(user, attrs) do
    user
    |> cast(attrs, [:display_name, :avatar_url, :bio])
    |> validate_length(:display_name, min: 2, max: 50)
    |> validate_length(:bio, max: 500)
    |> validate_url(:avatar_url)
  end

  @doc """
  Updates seller rating using incremental average formula.
  Avoids recalculating from all reviews which would be O(n) on review count.
  """
  def update_rating_changeset(user, new_rating) do
    new_count = user.rating_count + 1

    # Incremental average: new_avg = old_avg + (new_value - old_avg) / new_count
    new_avg =
      user.rating
      |> Decimal.add(Decimal.sub(new_rating, user.rating) |> Decimal.div(new_count))
      |> Decimal.round(2)

    change(user, rating: new_avg, rating_count: new_count)
  end

  defp validate_email(changeset) do
    changeset
    |> validate_format(:email, ~r/^[^\s]+@[^\s]+\.[^\s]+$/, message: "must be a valid email")
    |> validate_length(:email, max: 254)
    |> unsafe_validate_unique(:email, Marketplace.Repo)
    |> unique_constraint(:email)
  end

  defp validate_password(changeset) do
    changeset
    |> validate_length(:password, min: 8, max: 72)
  end

  defp hash_password(changeset) do
    case get_change(changeset, :password) do
      nil ->
        changeset

      password ->
        put_change(changeset, :password_hash, Bcrypt.hash_pwd_salt(password))
    end
  end

  defp validate_url(changeset, field) do
    validate_change(changeset, field, fn _, value ->
      case URI.parse(value) do
        %URI{scheme: scheme, host: host}
        when scheme in ["http", "https"] and not is_nil(host) ->
          []

        _ ->
          [{field, "must be a valid URL"}]
      end
    end)
  end
end
