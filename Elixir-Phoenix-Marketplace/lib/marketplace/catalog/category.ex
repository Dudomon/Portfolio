defmodule Marketplace.Catalog.Category do
  @moduledoc """
  Product categories with hierarchical structure.

  Uses materialized path pattern for tree traversal. Each category stores
  its full ancestry as a string path (e.g., "fashion/handbags/clutches").

  Why materialized path over nested sets or adjacency list:
  1. Reads are simple string prefix queries: WHERE path LIKE 'fashion/%'
  2. Writes update one row, not rebalancing entire subtree
  3. Human readable in database, aids debugging

  Tradeoff: Moving a subtree requires updating all descendants. Acceptable
  because category restructuring happens rarely (yearly) while reads happen
  constantly.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @type t :: %__MODULE__{
          id: Ecto.UUID.t(),
          name: String.t(),
          slug: String.t(),
          path: String.t(),
          depth: integer(),
          product_count: integer()
        }

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "categories" do
    field :name, :string
    field :slug, :string
    field :path, :string
    field :depth, :integer, default: 0
    field :product_count, :integer, default: 0

    belongs_to :parent, __MODULE__
    has_many :children, __MODULE__, foreign_key: :parent_id
    has_many :products, Marketplace.Catalog.Product

    timestamps()
  end

  def changeset(category, attrs) do
    category
    |> cast(attrs, [:name, :slug, :parent_id])
    |> validate_required([:name, :slug])
    |> validate_format(:slug, ~r/^[a-z0-9]+(?:-[a-z0-9]+)*$/, message: "must be lowercase with hyphens")
    |> unique_constraint(:slug)
    |> compute_path()
  end

  # Path computation happens after parent association is loaded.
  # Called by context function, not changeset, because it requires database reads.
  defp compute_path(changeset) do
    changeset
  end

  @doc """
  Generates URL safe slug from category name.
  """
  def generate_slug(name) do
    name
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9\s]/, "")
    |> String.replace(~r/\s+/, "-")
    |> String.trim("-")
  end
end
