defmodule Marketplace.Orders do
  @moduledoc """
  Public API for order processing.

  Handles the complete purchase flow from cart to completed order. Uses database
  transactions to ensure atomicity: either the order is created and product
  marked as sold, or neither happens.

  Payment processing delegates to external payment gateway (Stripe). This context
  handles the local state machine and coordinates with Catalog context to update
  product availability.

  Funds are held in escrow until buyer confirms receipt or auto release period
  (14 days) expires. This protects both parties in disputed transactions.
  """

  import Ecto.Query
  alias Marketplace.Repo
  alias Marketplace.Orders.{Order, OrderItem}
  alias Marketplace.Catalog

  @auto_complete_days 14

  @doc """
  Creates order from list of product IDs.

  Validates all products are available, calculates totals, and creates order
  with items in a single transaction. If any product is unavailable, the entire
  operation fails atomically.

  Returns {:ok, order} or {:error, reason}.
  """
  def create_order(buyer, product_ids, shipping_address) do
    Repo.transaction(fn ->
      with {:ok, products} <- fetch_available_products(product_ids),
           {:ok, order} <- insert_order(buyer, products, shipping_address),
           :ok <- insert_order_items(order, products),
           :ok <- mark_products_sold(products) do
        Repo.preload(order, [:items, :buyer])
      else
        {:error, reason} -> Repo.rollback(reason)
      end
    end)
  end

  @doc """
  Initiates payment for pending order.

  Creates payment intent with external gateway and transitions order to
  processing status. Payment confirmation happens asynchronously via webhook.
  """
  def initiate_payment(order) do
    # In production, this calls Stripe API
    payment_intent_id = "pi_#{generate_id()}"

    order
    |> Order.transition_changeset(:processing, %{payment_intent_id: payment_intent_id})
    |> Repo.update()
  end

  @doc """
  Confirms payment received. Called by payment gateway webhook handler.

  Transitions order to paid status and notifies seller to ship item.
  Idempotent: calling on already paid order returns success without side effects.
  """
  def confirm_payment(order, payment_intent_id) do
    cond do
      order.status == :paid ->
        {:ok, order}

      order.payment_intent_id != payment_intent_id ->
        {:error, :payment_mismatch}

      true ->
        order
        |> Order.transition_changeset(:paid, %{payment_intent_id: payment_intent_id})
        |> Repo.update()
        |> tap(&notify_seller_to_ship/1)
    end
  end

  @doc """
  Records shipment with tracking information.
  Transitions order to shipped status.
  """
  def mark_shipped(order, tracking_info \\ %{}) do
    order
    |> Order.transition_changeset(:shipped)
    |> Ecto.Changeset.put_change(:shipping_tracking, tracking_info)
    |> Repo.update()
    |> tap(&notify_buyer_shipped/1)
  end

  @doc """
  Completes order and releases funds to seller.
  Can be called by buyer to confirm receipt early, or by scheduled job
  after auto complete period expires.
  """
  def complete_order(order) do
    order
    |> Order.transition_changeset(:completed)
    |> Repo.update()
    |> tap(&release_funds_to_seller/1)
  end

  @doc """
  Opens dispute on order. Freezes funds pending resolution.
  """
  def open_dispute(order, reason) do
    order
    |> Order.transition_changeset(:disputed)
    |> Ecto.Changeset.put_change(:dispute_reason, reason)
    |> Repo.update()
    |> tap(&notify_support_team/1)
  end

  @doc """
  Retrieves order by ID for owner (buyer or item seller).
  Returns nil if order does not exist or user lacks permission.
  """
  def get_order(id, user) do
    order =
      Order
      |> Repo.get(id)
      |> Repo.preload([:items, :buyer, items: :seller])

    cond do
      is_nil(order) -> nil
      order.buyer_id == user.id -> order
      seller_of_order?(order, user) -> order
      true -> nil
    end
  end

  @doc """
  Lists orders for user (as buyer or seller) with pagination.
  """
  def list_orders_for_user(user, opts \\ []) do
    page = Keyword.get(opts, :page, 1)
    per_page = Keyword.get(opts, :per_page, 20)
    role = Keyword.get(opts, :role, :buyer)
    offset = (page - 1) * per_page

    query =
      case role do
        :buyer ->
          from o in Order, where: o.buyer_id == ^user.id

        :seller ->
          from o in Order,
            join: i in OrderItem,
            on: i.order_id == o.id,
            where: i.seller_id == ^user.id,
            distinct: true
      end

    query
    |> order_by(desc: :inserted_at)
    |> limit(^per_page)
    |> offset(^offset)
    |> Repo.all()
    |> Repo.preload([:items, :buyer])
  end

  @doc """
  Finds orders ready for auto completion.
  Called by scheduled job to release funds after waiting period.
  """
  def orders_ready_for_auto_complete do
    cutoff = NaiveDateTime.add(NaiveDateTime.utc_now(), -@auto_complete_days * 24 * 60 * 60)

    Order
    |> where([o], o.status == :shipped)
    |> where([o], o.shipped_at < ^cutoff)
    |> Repo.all()
  end

  # Private functions

  defp fetch_available_products(product_ids) do
    products =
      Enum.reduce_while(product_ids, {:ok, []}, fn id, {:ok, acc} ->
        case Catalog.get_available_product(id) do
          {:ok, product} -> {:cont, {:ok, [product | acc]}}
          {:error, _} -> {:halt, {:error, {:product_unavailable, id}}}
        end
      end)

    case products do
      {:ok, list} -> {:ok, Enum.reverse(list)}
      error -> error
    end
  end

  defp insert_order(buyer, products, shipping_address) do
    items = Enum.map(products, &%{price: &1.price})

    %Order{}
    |> Order.create_changeset(%{buyer_id: buyer.id, shipping_address: shipping_address}, items)
    |> Repo.insert()
  end

  defp insert_order_items(order, products) do
    items =
      Enum.map(products, fn product ->
        %OrderItem{}
        |> OrderItem.create_changeset(product, order.id)
        |> Repo.insert!()
      end)

    if length(items) == length(products), do: :ok, else: {:error, :item_creation_failed}
  end

  defp mark_products_sold(products) do
    Enum.each(products, &Catalog.mark_as_sold/1)
    :ok
  end

  defp seller_of_order?(order, user) do
    Enum.any?(order.items, &(&1.seller_id == user.id))
  end

  defp generate_id, do: :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)

  # Notification stubs; production integrates with email/push service
  defp notify_seller_to_ship({:ok, _order}), do: :ok
  defp notify_seller_to_ship(_), do: :ok

  defp notify_buyer_shipped({:ok, _order}), do: :ok
  defp notify_buyer_shipped(_), do: :ok

  defp release_funds_to_seller({:ok, _order}), do: :ok
  defp release_funds_to_seller(_), do: :ok

  defp notify_support_team({:ok, _order}), do: :ok
  defp notify_support_team(_), do: :ok
end
