defmodule Marketplace.Orders.Order do
  @moduledoc """
  Order representing a completed purchase transaction.

  Orders are immutable after creation. Price changes, product edits, or user
  profile updates do not affect historical orders. This is enforced by copying
  relevant data into order_items at purchase time rather than relying on
  foreign key joins.

  Payment flow:
  1. Order created with status :pending
  2. Payment initiated, status -> :processing
  3. Payment confirmed by webhook, status -> :paid
  4. Seller ships item, status -> :shipped
  5. Buyer confirms receipt or 14 days pass, status -> :completed
  6. Funds released to seller

  If payment fails: status -> :failed
  If disputed: status -> :disputed, enters manual review queue
  """

  use Ecto.Schema
  import Ecto.Changeset

  @type status ::
          :pending
          | :processing
          | :paid
          | :shipped
          | :completed
          | :failed
          | :disputed
          | :refunded

  @type t :: %__MODULE__{
          id: Ecto.UUID.t(),
          status: status(),
          subtotal: Decimal.t(),
          shipping_cost: Decimal.t(),
          tax_amount: Decimal.t(),
          total: Decimal.t(),
          shipping_address: map(),
          payment_intent_id: String.t() | nil
        }

  @statuses [:pending, :processing, :paid, :shipped, :completed, :failed, :disputed, :refunded]

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "orders" do
    field :status, Ecto.Enum, values: @statuses, default: :pending
    field :subtotal, :decimal
    field :shipping_cost, :decimal
    field :tax_amount, :decimal
    field :total, :decimal
    field :currency, :string, default: "USD"
    field :shipping_address, :map
    field :payment_intent_id, :string
    field :paid_at, :naive_datetime
    field :shipped_at, :naive_datetime
    field :completed_at, :naive_datetime

    belongs_to :buyer, Marketplace.Accounts.User
    has_many :items, Marketplace.Orders.OrderItem

    timestamps()
  end

  @doc """
  Creates order from validated cart data.
  Calculates totals server side; never trust client calculations.
  """
  def create_changeset(order, attrs, items) do
    subtotal = calculate_subtotal(items)
    shipping = calculate_shipping(items, attrs[:shipping_address])
    tax = calculate_tax(subtotal, attrs[:shipping_address])
    total = Decimal.add(subtotal, shipping) |> Decimal.add(tax)

    order
    |> cast(attrs, [:buyer_id, :shipping_address])
    |> validate_required([:buyer_id, :shipping_address])
    |> validate_shipping_address()
    |> put_change(:subtotal, subtotal)
    |> put_change(:shipping_cost, shipping)
    |> put_change(:tax_amount, tax)
    |> put_change(:total, total)
    |> foreign_key_constraint(:buyer_id)
  end

  @doc """
  Status transitions with validation and side effects tracking.
  Returns changeset that can be used in transaction with associated operations.
  """
  def transition_changeset(order, new_status, metadata \\ %{}) do
    cond do
      not valid_transition?(order.status, new_status) ->
        order
        |> change()
        |> add_error(:status, "invalid transition from #{order.status} to #{new_status}")

      new_status == :paid ->
        order
        |> change(status: new_status, paid_at: NaiveDateTime.utc_now())
        |> put_change(:payment_intent_id, metadata[:payment_intent_id])

      new_status == :shipped ->
        order
        |> change(status: new_status, shipped_at: NaiveDateTime.utc_now())

      new_status == :completed ->
        order
        |> change(status: new_status, completed_at: NaiveDateTime.utc_now())

      true ->
        change(order, status: new_status)
    end
  end

  defp calculate_subtotal(items) do
    Enum.reduce(items, Decimal.new(0), fn item, acc ->
      Decimal.add(acc, item.price)
    end)
  end

  # Shipping calculation based on item value and destination.
  # High value items get complimentary shipping (built into platform fees).
  defp calculate_shipping(items, address) do
    subtotal = calculate_subtotal(items)

    cond do
      Decimal.compare(subtotal, Decimal.new(500)) == :gt ->
        Decimal.new(0)

      domestic?(address) ->
        Decimal.new("12.99")

      true ->
        Decimal.new("29.99")
    end
  end

  # Tax calculation placeholder. Production would integrate with tax service
  # like Avalara or TaxJar based on nexus requirements.
  defp calculate_tax(subtotal, address) do
    rate = tax_rate_for_state(address[:state])
    Decimal.mult(subtotal, rate) |> Decimal.round(2)
  end

  defp domestic?(address), do: address[:country] == "US"

  defp tax_rate_for_state(state) do
    # Simplified; real implementation queries tax service
    case state do
      "CA" -> Decimal.new("0.0725")
      "NY" -> Decimal.new("0.08")
      "TX" -> Decimal.new("0.0625")
      _ -> Decimal.new(0)
    end
  end

  defp validate_shipping_address(changeset) do
    validate_change(changeset, :shipping_address, fn :shipping_address, address ->
      required = [:street, :city, :state, :postal_code, :country]
      missing = Enum.filter(required, fn key -> is_nil(address[key]) end)

      if Enum.empty?(missing) do
        []
      else
        [shipping_address: "missing required fields: #{Enum.join(missing, ", ")}"]
      end
    end)
  end

  defp valid_transition?(from, to) do
    transitions = %{
      pending: [:processing, :failed],
      processing: [:paid, :failed],
      paid: [:shipped, :disputed, :refunded],
      shipped: [:completed, :disputed],
      completed: [:disputed],
      failed: [],
      disputed: [:refunded, :completed],
      refunded: []
    }

    to in Map.get(transitions, from, [])
  end
end
