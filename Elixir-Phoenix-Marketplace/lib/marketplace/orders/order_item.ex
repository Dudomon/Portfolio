defmodule Marketplace.Orders.OrderItem do
  @moduledoc """
  Snapshot of product data at time of purchase.

  Denormalizes product and seller information to preserve historical accuracy.
  When a seller changes their display name or a product title is edited,
  order history must reflect what the buyer actually purchased.

  This pattern adds storage overhead but eliminates an entire class of bugs
  where historical reports show current data instead of point in time data.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @type t :: %__MODULE__{
          id: Ecto.UUID.t(),
          price: Decimal.t(),
          product_snapshot: map()
        }

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "order_items" do
    field :price, :decimal

    # Snapshot of product data at purchase time
    field :product_snapshot, :map

    belongs_to :order, Marketplace.Orders.Order
    belongs_to :product, Marketplace.Catalog.Product
    belongs_to :seller, Marketplace.Accounts.User

    timestamps()
  end

  @doc """
  Creates order item from product, capturing point in time snapshot.
  """
  def create_changeset(order_item, product, order_id) do
    snapshot = build_snapshot(product)

    order_item
    |> cast(%{}, [])
    |> put_change(:order_id, order_id)
    |> put_change(:product_id, product.id)
    |> put_change(:seller_id, product.seller_id)
    |> put_change(:price, product.price)
    |> put_change(:product_snapshot, snapshot)
    |> foreign_key_constraint(:order_id)
    |> foreign_key_constraint(:product_id)
    |> foreign_key_constraint(:seller_id)
  end

  defp build_snapshot(product) do
    %{
      title: product.title,
      description: product.description,
      condition: product.condition,
      brand: product.brand,
      images: product.images,
      category_id: product.category_id,
      original_price: product.original_price && Decimal.to_string(product.original_price)
    }
  end
end
