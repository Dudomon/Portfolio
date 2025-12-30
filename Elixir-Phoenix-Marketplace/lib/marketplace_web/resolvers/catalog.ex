defmodule MarketplaceWeb.Resolvers.Catalog do
  @moduledoc """
  GraphQL resolvers for catalog operations.

  Product listing is public. Product creation and management requires
  authentication. Resolvers handle authorization checks and delegate
  to Catalog context for business logic.
  """

  alias Marketplace.Catalog

  @doc """
  Lists products with filtering, sorting, and pagination.
  Returns connection object with items and pagination metadata.
  """
  def list_products(_, args, _) do
    opts =
      args
      |> Map.take([:category_id, :seller_id, :min_price, :max_price, :condition, :search, :sort])
      |> Map.to_list()
      |> Keyword.merge(page: args[:page], per_page: args[:per_page])

    products = Catalog.list_products(opts)
    total_count = Catalog.count_products(opts)
    total_pages = ceil(total_count / args[:per_page])

    {:ok,
     %{
       items: products,
       total_count: total_count,
       page: args[:page],
       per_page: args[:per_page],
       total_pages: total_pages
     }}
  end

  @doc """
  Fetches single product by ID.
  Increments view count as side effect for analytics.
  """
  def get_product(_, %{id: id}, _) do
    case Catalog.get_product(id) do
      nil ->
        {:error, "Product not found"}

      product ->
        # Fire and forget view tracking
        Task.start(fn -> Catalog.increment_view_count(id) end)
        {:ok, product}
    end
  end

  @doc """
  Lists all categories for navigation.
  """
  def list_categories(_, _, _) do
    {:ok, Catalog.list_categories()}
  end

  @doc """
  Creates new product listing for authenticated seller.
  Product starts in draft status.
  """
  def create_product(_, %{input: input}, %{context: %{current_user: user}}) when not is_nil(user) do
    case Catalog.create_product(user, input) do
      {:ok, product} ->
        {:ok, product}

      {:error, changeset} ->
        {:error, format_errors(changeset)}
    end
  end

  def create_product(_, _, _), do: {:error, "Authentication required"}

  @doc """
  Publishes draft product for sale.
  Only product owner can publish.
  """
  def publish_product(_, %{id: id}, %{context: %{current_user: user}}) when not is_nil(user) do
    with product when not is_nil(product) <- Catalog.get_product(id),
         :ok <- verify_ownership(product, user),
         {:ok, published} <- Catalog.publish_product(product) do
      {:ok, published}
    else
      nil -> {:error, "Product not found"}
      {:error, :not_owner} -> {:error, "Not authorized to publish this product"}
      {:error, changeset} -> {:error, format_errors(changeset)}
    end
  end

  def publish_product(_, _, _), do: {:error, "Authentication required"}

  defp verify_ownership(product, user) do
    if product.seller_id == user.id do
      :ok
    else
      {:error, :not_owner}
    end
  end

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
