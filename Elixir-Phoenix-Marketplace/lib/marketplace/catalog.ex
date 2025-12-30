defmodule Marketplace.Catalog do
  @moduledoc """
  Public API for product catalog operations.

  Handles product listings, search, and category management. Products go through
  a state machine from draft to active to sold. High value items require
  authentication before becoming publicly visible.

  Search uses PostgreSQL full text search rather than Elasticsearch. For a
  marketplace with under 1 million products, Postgres FTS provides adequate
  performance without operational complexity of a separate search cluster.
  When scale demands it, search can be extracted without changing this API.
  """

  import Ecto.Query
  alias Marketplace.Repo
  alias Marketplace.Catalog.{Product, Category}

  # Products above this threshold require authentication before listing
  @auth_threshold Decimal.new(1000)

  @doc """
  Creates a new product listing in draft status.

  Products are not visible to buyers until published. Seller can edit
  draft products freely before submission.
  """
  def create_product(seller, attrs) do
    %Product{}
    |> Product.create_changeset(Map.put(attrs, :seller_id, seller.id))
    |> Repo.insert()
  end

  @doc """
  Publishes a draft product, making it available for purchase.

  High value items (over $1000) enter pending_auth status and require
  platform authentication before becoming active. Lower value items
  go directly to active status.
  """
  def publish_product(product) do
    target_status =
      if Decimal.compare(product.price, @auth_threshold) == :gt do
        :pending_auth
      else
        :active
      end

    product
    |> Product.status_changeset(target_status)
    |> Repo.update()
    |> tap(&broadcast_product_update/1)
  end

  @doc """
  Marks product as sold. Called after successful order completion.
  This is idempotent; calling on already sold product returns success.
  """
  def mark_as_sold(product) do
    if product.status == :sold do
      {:ok, product}
    else
      product
      |> Product.status_changeset(:sold)
      |> Repo.update()
      |> tap(&broadcast_product_update/1)
    end
  end

  @doc """
  Retrieves single product by ID. Returns nil for deleted or non existent products.
  """
  def get_product(id) do
    Product
    |> Product.not_deleted()
    |> Repo.get(id)
    |> Repo.preload([:seller, :category])
  end

  @doc """
  Retrieves product for purchase. Returns error if not available for sale.
  Used in checkout flow to validate product can be purchased.
  """
  def get_available_product(id) do
    product =
      Product
      |> Product.not_deleted()
      |> Product.available()
      |> Repo.get(id)

    case product do
      nil -> {:error, :not_available}
      product -> {:ok, Repo.preload(product, [:seller, :category])}
    end
  end

  @doc """
  Lists products with filtering, sorting, and pagination.

  Options:
    - category_id: Filter by category
    - seller_id: Filter by seller
    - min_price / max_price: Price range
    - condition: Filter by condition
    - search: Full text search on title and description
    - sort: :newest, :price_asc, :price_desc
    - page / per_page: Pagination
  """
  def list_products(opts \\ []) do
    page = Keyword.get(opts, :page, 1)
    per_page = Keyword.get(opts, :per_page, 24)
    offset = (page - 1) * per_page

    Product
    |> Product.not_deleted()
    |> Product.available()
    |> filter_by_category(opts[:category_id])
    |> filter_by_seller(opts[:seller_id])
    |> filter_by_price_range(opts[:min_price], opts[:max_price])
    |> filter_by_condition(opts[:condition])
    |> filter_by_search(opts[:search])
    |> apply_sort(opts[:sort])
    |> limit(^per_page)
    |> offset(^offset)
    |> Repo.all()
    |> Repo.preload([:seller, :category])
  end

  @doc """
  Counts products matching filter criteria. Used for pagination metadata.
  """
  def count_products(opts \\ []) do
    Product
    |> Product.not_deleted()
    |> Product.available()
    |> filter_by_category(opts[:category_id])
    |> filter_by_seller(opts[:seller_id])
    |> filter_by_price_range(opts[:min_price], opts[:max_price])
    |> filter_by_condition(opts[:condition])
    |> filter_by_search(opts[:search])
    |> Repo.aggregate(:count, :id)
  end

  @doc """
  Increments view count. Fire and forget; failure does not affect user experience.
  Uses update_all for atomic increment without read then write race condition.
  """
  def increment_view_count(product_id) do
    Product
    |> where(id: ^product_id)
    |> Repo.update_all(inc: [view_count: 1])
  end

  @doc """
  Lists all categories in tree order for navigation display.
  """
  def list_categories do
    Category
    |> order_by(:path)
    |> Repo.all()
  end

  @doc """
  Gets category by slug for URL friendly lookups.
  """
  def get_category_by_slug(slug) do
    Repo.get_by(Category, slug: slug)
  end

  @doc """
  Dataloader source for batched loading in GraphQL.
  """
  def data do
    Dataloader.Ecto.new(Repo, query: &query/2)
  end

  def query(queryable, _params) do
    queryable
  end

  # Private filter functions compose query based on provided options

  defp filter_by_category(query, nil), do: query

  defp filter_by_category(query, category_id) do
    from p in query, where: p.category_id == ^category_id
  end

  defp filter_by_seller(query, nil), do: query

  defp filter_by_seller(query, seller_id) do
    from p in query, where: p.seller_id == ^seller_id
  end

  defp filter_by_price_range(query, nil, nil), do: query

  defp filter_by_price_range(query, min, nil) do
    from p in query, where: p.price >= ^min
  end

  defp filter_by_price_range(query, nil, max) do
    from p in query, where: p.price <= ^max
  end

  defp filter_by_price_range(query, min, max) do
    from p in query, where: p.price >= ^min and p.price <= ^max
  end

  defp filter_by_condition(query, nil), do: query

  defp filter_by_condition(query, conditions) when is_list(conditions) do
    from p in query, where: p.condition in ^conditions
  end

  defp filter_by_condition(query, condition) do
    from p in query, where: p.condition == ^condition
  end

  defp filter_by_search(query, nil), do: query
  defp filter_by_search(query, ""), do: query

  defp filter_by_search(query, search_term) do
    # PostgreSQL full text search with websearch_to_tsquery for natural language
    search = "%#{search_term}%"

    from p in query,
      where: ilike(p.title, ^search) or ilike(p.brand, ^search)
  end

  defp apply_sort(query, nil), do: order_by(query, desc: :inserted_at)
  defp apply_sort(query, :newest), do: order_by(query, desc: :inserted_at)
  defp apply_sort(query, :price_asc), do: order_by(query, asc: :price)
  defp apply_sort(query, :price_desc), do: order_by(query, desc: :price)

  defp broadcast_product_update({:ok, product}) do
    Phoenix.PubSub.broadcast(
      Marketplace.PubSub,
      "products:#{product.id}",
      {:product_updated, product}
    )
  end

  defp broadcast_product_update(error), do: error
end
