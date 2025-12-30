defmodule Marketplace.Catalog.Product do
  @moduledoc """
  Product listing for consignment marketplace.

  Each product represents a single physical item. Unlike retail e-commerce,
  there is no inventory count; quantity is always 1 or 0 (sold).

  Condition grades follow industry standard for luxury resale:
  - new_with_tags: Never used, original tags attached
  - new_without_tags: Never used, tags removed
  - excellent: Minimal signs of wear, no visible flaws
  - very_good: Light wear, minor imperfections
  - good: Moderate wear, visible but not major flaws
  - fair: Significant wear, may need repair

  Authentication status tracks whether item has been verified by platform
  authenticators. High value items require authentication before listing
  goes live.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @type condition ::
          :new_with_tags
          | :new_without_tags
          | :excellent
          | :very_good
          | :good
          | :fair

  @type status :: :draft | :pending_auth | :active | :sold | :withdrawn

  @type t :: %__MODULE__{
          id: Ecto.UUID.t(),
          title: String.t(),
          description: String.t(),
          price: Decimal.t(),
          original_price: Decimal.t() | nil,
          condition: condition(),
          status: status(),
          brand: String.t(),
          images: [String.t()],
          attributes: map(),
          deleted_at: NaiveDateTime.t() | nil
        }

  @conditions [:new_with_tags, :new_without_tags, :excellent, :very_good, :good, :fair]
  @statuses [:draft, :pending_auth, :active, :sold, :withdrawn]

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "products" do
    field :title, :string
    field :description, :string
    field :price, :decimal
    field :original_price, :decimal
    field :condition, Ecto.Enum, values: @conditions
    field :status, Ecto.Enum, values: @statuses, default: :draft
    field :brand, :string
    field :images, {:array, :string}, default: []
    field :attributes, :map, default: %{}
    field :view_count, :integer, default: 0
    field :favorite_count, :integer, default: 0
    field :deleted_at, :naive_datetime

    belongs_to :seller, Marketplace.Accounts.User
    belongs_to :category, Marketplace.Catalog.Category
    has_many :order_items, Marketplace.Orders.OrderItem

    timestamps()
  end

  @doc """
  Initial product creation changeset.
  Validates required fields and business rules for listing.
  """
  def create_changeset(product, attrs) do
    product
    |> cast(attrs, [
      :title,
      :description,
      :price,
      :original_price,
      :condition,
      :brand,
      :images,
      :attributes,
      :category_id,
      :seller_id
    ])
    |> validate_required([:title, :description, :price, :condition, :brand, :seller_id, :category_id])
    |> validate_length(:title, min: 10, max: 200)
    |> validate_length(:description, min: 50, max: 5000)
    |> validate_price()
    |> validate_images()
    |> foreign_key_constraint(:seller_id)
    |> foreign_key_constraint(:category_id)
  end

  @doc """
  Status transition changeset enforces valid state machine transitions.

  Valid transitions:
    draft -> pending_auth (high value items)
    draft -> active (items below auth threshold)
    pending_auth -> active (after authentication)
    pending_auth -> withdrawn (failed authentication)
    active -> sold (purchase completed)
    active -> withdrawn (seller removes listing)
  """
  def status_changeset(product, new_status) do
    if valid_transition?(product.status, new_status) do
      change(product, status: new_status)
    else
      product
      |> change()
      |> add_error(:status, "cannot transition from #{product.status} to #{new_status}")
    end
  end

  @doc """
  Soft delete preserves data for audit trail and dispute resolution.
  """
  def soft_delete_changeset(product) do
    change(product, deleted_at: NaiveDateTime.utc_now())
  end

  @doc """
  Query scope for non deleted products. Use in all public facing queries.
  """
  def not_deleted(query \\ __MODULE__) do
    from p in query, where: is_nil(p.deleted_at)
  end

  @doc """
  Query scope for products available for purchase.
  """
  def available(query \\ __MODULE__) do
    from p in query, where: p.status == :active
  end

  defp validate_price(changeset) do
    changeset
    |> validate_number(:price, greater_than: 0)
    |> validate_original_price()
  end

  # Original price, if provided, must exceed current price.
  # This prevents misleading "discounts" on items that were never sold at
  # the claimed original price.
  defp validate_original_price(changeset) do
    validate_change(changeset, :original_price, fn :original_price, original ->
      price = get_field(changeset, :price)

      if price && Decimal.compare(original, price) == :gt do
        []
      else
        [original_price: "must be greater than current price"]
      end
    end)
  end

  defp validate_images(changeset) do
    validate_change(changeset, :images, fn :images, images ->
      cond do
        length(images) < 3 ->
          [images: "minimum 3 images required"]

        length(images) > 12 ->
          [images: "maximum 12 images allowed"]

        not Enum.all?(images, &valid_image_url?/1) ->
          [images: "all images must be valid URLs"]

        true ->
          []
      end
    end)
  end

  defp valid_image_url?(url) do
    case URI.parse(url) do
      %URI{scheme: scheme, host: host} when scheme in ["http", "https"] and not is_nil(host) ->
        String.match?(url, ~r/\.(jpg|jpeg|png|webp)$/i)

      _ ->
        false
    end
  end

  defp valid_transition?(from, to) do
    transitions = %{
      draft: [:pending_auth, :active],
      pending_auth: [:active, :withdrawn],
      active: [:sold, :withdrawn],
      sold: [],
      withdrawn: [:draft]
    }

    to in Map.get(transitions, from, [])
  end
end
