defmodule Marketplace.Repo.Migrations.CreateTables do
  @moduledoc """
  Initial database schema for marketplace.

  Index strategy:
  - B-tree on foreign keys for join performance
  - GIN on products.title for full text search
  - Partial index on products for active only queries (most common)
  - Composite index on orders(buyer_id, status) for user order history

  UUID primary keys chosen over auto increment integers:
  - No sequential enumeration (security)
  - Safe for distributed systems if we shard later
  - Tradeoff: 16 bytes vs 4 bytes, negligible at our scale
  """

  use Ecto.Migration

  def change do
    # Enable UUID generation
    execute "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"", ""

    create table(:users, primary_key: false) do
      add :id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()")
      add :email, :string, null: false
      add :password_hash, :string, null: false
      add :display_name, :string, null: false
      add :avatar_url, :string
      add :bio, :text
      add :verified_seller, :boolean, default: false, null: false
      add :rating, :decimal, precision: 3, scale: 2, default: 0
      add :rating_count, :integer, default: 0, null: false

      timestamps()
    end

    create unique_index(:users, [:email])
    # Lowercase index for case insensitive email lookup
    create index(:users, ["lower(email)"], name: :users_email_lower_index)

    create table(:categories, primary_key: false) do
      add :id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()")
      add :name, :string, null: false
      add :slug, :string, null: false
      add :path, :string, null: false
      add :depth, :integer, default: 0, null: false
      add :product_count, :integer, default: 0, null: false
      add :parent_id, references(:categories, type: :binary_id, on_delete: :nilify_all)

      timestamps()
    end

    create unique_index(:categories, [:slug])
    # Prefix index for tree traversal queries: WHERE path LIKE 'fashion/%'
    create index(:categories, [:path], using: "btree", where: "path IS NOT NULL")

    create table(:products, primary_key: false) do
      add :id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()")
      add :title, :string, null: false
      add :description, :text, null: false
      add :price, :decimal, precision: 12, scale: 2, null: false
      add :original_price, :decimal, precision: 12, scale: 2
      add :condition, :string, null: false
      add :status, :string, default: "draft", null: false
      add :brand, :string, null: false
      add :images, {:array, :string}, default: [], null: false
      add :attributes, :map, default: %{}
      add :view_count, :integer, default: 0, null: false
      add :favorite_count, :integer, default: 0, null: false
      add :deleted_at, :naive_datetime

      add :seller_id, references(:users, type: :binary_id, on_delete: :nothing), null: false
      add :category_id, references(:categories, type: :binary_id, on_delete: :nothing), null: false

      timestamps()
    end

    create index(:products, [:seller_id])
    create index(:products, [:category_id])
    # Partial index: only active products for public queries
    create index(:products, [:status], where: "status = 'active' AND deleted_at IS NULL", name: :products_active_index)
    # Price range queries
    create index(:products, [:price])
    # Full text search on title and brand
    execute """
    CREATE INDEX products_search_index ON products
    USING gin(to_tsvector('english', title || ' ' || brand))
    WHERE deleted_at IS NULL
    """, "DROP INDEX products_search_index"

    create table(:orders, primary_key: false) do
      add :id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()")
      add :status, :string, default: "pending", null: false
      add :subtotal, :decimal, precision: 12, scale: 2, null: false
      add :shipping_cost, :decimal, precision: 8, scale: 2, null: false
      add :tax_amount, :decimal, precision: 10, scale: 2, null: false
      add :total, :decimal, precision: 12, scale: 2, null: false
      add :currency, :string, default: "USD", null: false
      add :shipping_address, :map, null: false
      add :shipping_tracking, :map, default: %{}
      add :payment_intent_id, :string
      add :dispute_reason, :text
      add :paid_at, :naive_datetime
      add :shipped_at, :naive_datetime
      add :completed_at, :naive_datetime

      add :buyer_id, references(:users, type: :binary_id, on_delete: :nothing), null: false

      timestamps()
    end

    create index(:orders, [:buyer_id])
    # Composite index for user order history filtered by status
    create index(:orders, [:buyer_id, :status])
    create index(:orders, [:payment_intent_id], where: "payment_intent_id IS NOT NULL")
    # For auto complete job: find shipped orders older than N days
    create index(:orders, [:shipped_at], where: "status = 'shipped'")

    create table(:order_items, primary_key: false) do
      add :id, :binary_id, primary_key: true, default: fragment("uuid_generate_v4()")
      add :price, :decimal, precision: 12, scale: 2, null: false
      add :product_snapshot, :map, null: false

      add :order_id, references(:orders, type: :binary_id, on_delete: :delete_all), null: false
      add :product_id, references(:products, type: :binary_id, on_delete: :nothing), null: false
      add :seller_id, references(:users, type: :binary_id, on_delete: :nothing), null: false

      timestamps()
    end

    create index(:order_items, [:order_id])
    create index(:order_items, [:seller_id])
    # Prevent duplicate product in same order
    create unique_index(:order_items, [:order_id, :product_id])
  end
end
