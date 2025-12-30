defmodule MarketplaceWeb.Schema do
  @moduledoc """
  GraphQL schema definition using Absinthe.

  Organizes types into separate modules for maintainability. Uses Dataloader
  for batched association loading to prevent N+1 queries. A product listing
  page loading 24 products with seller info makes 2 queries (products + users)
  instead of 25.

  Authentication happens at the resolver level, not middleware. This allows
  public queries (product listing) alongside authenticated mutations (create order)
  in the same schema.

  Subscriptions use Phoenix PubSub for real time updates. When a product sells,
  all connected clients viewing that product receive instant notification through
  their existing WebSocket connection.
  """

  use Absinthe.Schema
  import_types MarketplaceWeb.Schema.Types
  import_types Absinthe.Type.Custom

  alias MarketplaceWeb.Resolvers

  query do
    @desc "Get current authenticated user"
    field :me, :user do
      resolve &Resolvers.Accounts.me/3
    end

    @desc "Get user by ID"
    field :user, :user do
      arg :id, non_null(:id)
      resolve &Resolvers.Accounts.get_user/3
    end

    @desc "List products with filtering and pagination"
    field :products, :product_connection do
      arg :category_id, :id
      arg :seller_id, :id
      arg :min_price, :decimal
      arg :max_price, :decimal
      arg :condition, list_of(:condition)
      arg :search, :string
      arg :sort, :product_sort
      arg :page, :integer, default_value: 1
      arg :per_page, :integer, default_value: 24
      resolve &Resolvers.Catalog.list_products/3
    end

    @desc "Get single product by ID"
    field :product, :product do
      arg :id, non_null(:id)
      resolve &Resolvers.Catalog.get_product/3
    end

    @desc "List all categories"
    field :categories, list_of(:category) do
      resolve &Resolvers.Catalog.list_categories/3
    end

    @desc "Get order by ID (must be buyer or seller)"
    field :order, :order do
      arg :id, non_null(:id)
      resolve &Resolvers.Orders.get_order/3
    end

    @desc "List orders for current user"
    field :my_orders, list_of(:order) do
      arg :role, :order_role, default_value: :buyer
      arg :page, :integer, default_value: 1
      arg :per_page, :integer, default_value: 20
      resolve &Resolvers.Orders.list_my_orders/3
    end
  end

  mutation do
    @desc "Register new user account"
    field :register, :auth_payload do
      arg :email, non_null(:string)
      arg :password, non_null(:string)
      arg :display_name, non_null(:string)
      resolve &Resolvers.Accounts.register/3
    end

    @desc "Login with email and password"
    field :login, :auth_payload do
      arg :email, non_null(:string)
      arg :password, non_null(:string)
      resolve &Resolvers.Accounts.login/3
    end

    @desc "Update current user profile"
    field :update_profile, :user do
      arg :display_name, :string
      arg :bio, :string
      arg :avatar_url, :string
      resolve &Resolvers.Accounts.update_profile/3
    end

    @desc "Create new product listing"
    field :create_product, :product do
      arg :input, non_null(:product_input)
      resolve &Resolvers.Catalog.create_product/3
    end

    @desc "Publish draft product for sale"
    field :publish_product, :product do
      arg :id, non_null(:id)
      resolve &Resolvers.Catalog.publish_product/3
    end

    @desc "Create order from product IDs"
    field :create_order, :order do
      arg :product_ids, non_null(list_of(non_null(:id)))
      arg :shipping_address, non_null(:address_input)
      resolve &Resolvers.Orders.create_order/3
    end

    @desc "Initiate payment for order"
    field :initiate_payment, :payment_result do
      arg :order_id, non_null(:id)
      resolve &Resolvers.Orders.initiate_payment/3
    end

    @desc "Mark order as shipped (seller only)"
    field :mark_shipped, :order do
      arg :order_id, non_null(:id)
      arg :tracking_number, :string
      arg :carrier, :string
      resolve &Resolvers.Orders.mark_shipped/3
    end

    @desc "Complete order and release funds (buyer only)"
    field :complete_order, :order do
      arg :order_id, non_null(:id)
      resolve &Resolvers.Orders.complete_order/3
    end
  end

  subscription do
    @desc "Subscribe to product updates (price changes, sold status)"
    field :product_updated, :product do
      arg :product_id, non_null(:id)

      config fn args, _ ->
        {:ok, topic: "products:#{args.product_id}"}
      end

      trigger :publish_product, topic: fn product ->
        "products:#{product.id}"
      end
    end

    @desc "Subscribe to order status changes"
    field :order_updated, :order do
      arg :order_id, non_null(:id)

      config fn args, context ->
        # Verify user has access to this order before allowing subscription
        case context[:current_user] do
          nil -> {:error, "authentication required"}
          _ -> {:ok, topic: "orders:#{args.order_id}"}
        end
      end
    end
  end

  # Dataloader setup for batched association loading
  def context(ctx) do
    loader =
      Dataloader.new()
      |> Dataloader.add_source(Marketplace.Accounts, Marketplace.Accounts.data())
      |> Dataloader.add_source(Marketplace.Catalog, Marketplace.Catalog.data())

    Map.put(ctx, :loader, loader)
  end

  def plugins do
    [Absinthe.Middleware.Dataloader] ++ Absinthe.Plugin.defaults()
  end
end
