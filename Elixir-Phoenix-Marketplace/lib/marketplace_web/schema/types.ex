defmodule MarketplaceWeb.Schema.Types do
  @moduledoc """
  GraphQL type definitions.

  Types mirror Ecto schemas but expose only fields appropriate for the API.
  Internal fields (password_hash, deleted_at) are never exposed. Computed
  fields (savings_percent, formatted_price) exist only in GraphQL layer.
  """

  use Absinthe.Schema.Notation
  import Absinthe.Resolution.Helpers, only: [dataloader: 1]

  # Custom scalars

  scalar :decimal do
    parse fn
      %Absinthe.Blueprint.Input.String{value: value}, _ ->
        Decimal.parse(value)

      %Absinthe.Blueprint.Input.Float{value: value}, _ ->
        {:ok, Decimal.from_float(value)}

      %Absinthe.Blueprint.Input.Integer{value: value}, _ ->
        {:ok, Decimal.new(value)}

      _, _ ->
        :error
    end

    serialize fn decimal ->
      Decimal.to_string(decimal)
    end
  end

  # Enums

  enum :condition do
    value :new_with_tags, description: "Never used, original tags attached"
    value :new_without_tags, description: "Never used, tags removed"
    value :excellent, description: "Minimal signs of wear"
    value :very_good, description: "Light wear, minor imperfections"
    value :good, description: "Moderate wear, visible flaws"
    value :fair, description: "Significant wear, may need repair"
  end

  enum :product_status do
    value :draft
    value :pending_auth
    value :active
    value :sold
    value :withdrawn
  end

  enum :order_status do
    value :pending
    value :processing
    value :paid
    value :shipped
    value :completed
    value :failed
    value :disputed
    value :refunded
  end

  enum :product_sort do
    value :newest
    value :price_asc
    value :price_desc
  end

  enum :order_role do
    value :buyer
    value :seller
  end

  # Object types

  object :user do
    field :id, non_null(:id)
    field :email, :string  # Only visible to self
    field :display_name, non_null(:string)
    field :avatar_url, :string
    field :bio, :string
    field :verified_seller, non_null(:boolean)
    field :rating, :decimal
    field :rating_count, non_null(:integer)
    field :inserted_at, non_null(:datetime)

    field :products, list_of(:product) do
      arg :page, :integer, default_value: 1
      arg :per_page, :integer, default_value: 12
      resolve dataloader(Marketplace.Catalog)
    end
  end

  object :product do
    field :id, non_null(:id)
    field :title, non_null(:string)
    field :description, non_null(:string)
    field :price, non_null(:decimal)
    field :original_price, :decimal
    field :condition, non_null(:condition)
    field :status, non_null(:product_status)
    field :brand, non_null(:string)
    field :images, non_null(list_of(non_null(:string)))
    field :attributes, :json
    field :view_count, non_null(:integer)
    field :favorite_count, non_null(:integer)
    field :inserted_at, non_null(:datetime)

    field :seller, non_null(:user), resolve: dataloader(Marketplace.Accounts)
    field :category, non_null(:category), resolve: dataloader(Marketplace.Catalog)

    # Computed field: percentage savings from original price
    field :savings_percent, :integer do
      resolve fn product, _, _ ->
        case product.original_price do
          nil ->
            {:ok, nil}

          original ->
            savings =
              original
              |> Decimal.sub(product.price)
              |> Decimal.div(original)
              |> Decimal.mult(100)
              |> Decimal.round(0)
              |> Decimal.to_integer()

            {:ok, savings}
        end
      end
    end
  end

  object :category do
    field :id, non_null(:id)
    field :name, non_null(:string)
    field :slug, non_null(:string)
    field :path, non_null(:string)
    field :depth, non_null(:integer)
    field :product_count, non_null(:integer)

    field :parent, :category, resolve: dataloader(Marketplace.Catalog)
    field :children, list_of(:category), resolve: dataloader(Marketplace.Catalog)
  end

  object :order do
    field :id, non_null(:id)
    field :status, non_null(:order_status)
    field :subtotal, non_null(:decimal)
    field :shipping_cost, non_null(:decimal)
    field :tax_amount, non_null(:decimal)
    field :total, non_null(:decimal)
    field :currency, non_null(:string)
    field :shipping_address, non_null(:address)
    field :paid_at, :datetime
    field :shipped_at, :datetime
    field :completed_at, :datetime
    field :inserted_at, non_null(:datetime)

    field :buyer, non_null(:user), resolve: dataloader(Marketplace.Accounts)
    field :items, non_null(list_of(non_null(:order_item)))
  end

  object :order_item do
    field :id, non_null(:id)
    field :price, non_null(:decimal)
    field :product_snapshot, non_null(:json)
    field :product, :product, resolve: dataloader(Marketplace.Catalog)
    field :seller, non_null(:user), resolve: dataloader(Marketplace.Accounts)
  end

  object :address do
    field :street, non_null(:string)
    field :city, non_null(:string)
    field :state, non_null(:string)
    field :postal_code, non_null(:string)
    field :country, non_null(:string)
  end

  object :auth_payload do
    field :token, non_null(:string)
    field :user, non_null(:user)
  end

  object :payment_result do
    field :client_secret, non_null(:string)
    field :order, non_null(:order)
  end

  # Pagination wrapper
  object :product_connection do
    field :items, non_null(list_of(non_null(:product)))
    field :total_count, non_null(:integer)
    field :page, non_null(:integer)
    field :per_page, non_null(:integer)
    field :total_pages, non_null(:integer)
  end

  # Input types

  input_object :product_input do
    field :title, non_null(:string)
    field :description, non_null(:string)
    field :price, non_null(:decimal)
    field :original_price, :decimal
    field :condition, non_null(:condition)
    field :brand, non_null(:string)
    field :images, non_null(list_of(non_null(:string)))
    field :attributes, :json
    field :category_id, non_null(:id)
  end

  input_object :address_input do
    field :street, non_null(:string)
    field :city, non_null(:string)
    field :state, non_null(:string)
    field :postal_code, non_null(:string)
    field :country, non_null(:string)
  end

  # JSON scalar for flexible attribute storage
  scalar :json do
    parse fn
      %Absinthe.Blueprint.Input.String{value: value}, _ ->
        case Jason.decode(value) do
          {:ok, result} -> {:ok, result}
          _ -> :error
        end

      %Absinthe.Blueprint.Input.Null{}, _ ->
        {:ok, nil}

      _, _ ->
        :error
    end

    serialize fn value ->
      value
    end
  end
end
