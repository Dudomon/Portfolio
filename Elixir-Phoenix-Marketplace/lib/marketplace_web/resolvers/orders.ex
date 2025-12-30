defmodule MarketplaceWeb.Resolvers.Orders do
  @moduledoc """
  GraphQL resolvers for order operations.

  All order operations require authentication. Authorization varies by action:
  buyers can create orders and confirm receipt; sellers can mark orders shipped.
  """

  alias Marketplace.Orders

  @doc """
  Fetches order by ID. User must be buyer or seller of order items.
  """
  def get_order(_, %{id: id}, %{context: %{current_user: user}}) when not is_nil(user) do
    case Orders.get_order(id, user) do
      nil -> {:error, "Order not found or access denied"}
      order -> {:ok, order}
    end
  end

  def get_order(_, _, _), do: {:error, "Authentication required"}

  @doc """
  Lists orders for current user with role filter.
  """
  def list_my_orders(_, args, %{context: %{current_user: user}}) when not is_nil(user) do
    opts = [
      role: args[:role],
      page: args[:page],
      per_page: args[:per_page]
    ]

    orders = Orders.list_orders_for_user(user, opts)
    {:ok, orders}
  end

  def list_my_orders(_, _, _), do: {:error, "Authentication required"}

  @doc """
  Creates order from product IDs.
  Validates product availability atomically.
  """
  def create_order(_, args, %{context: %{current_user: user}}) when not is_nil(user) do
    shipping_address = Map.from_struct_if_needed(args.shipping_address)

    case Orders.create_order(user, args.product_ids, shipping_address) do
      {:ok, order} ->
        broadcast_order_created(order)
        {:ok, order}

      {:error, {:product_unavailable, product_id}} ->
        {:error, "Product #{product_id} is no longer available"}

      {:error, reason} ->
        {:error, format_error(reason)}
    end
  end

  def create_order(_, _, _), do: {:error, "Authentication required"}

  @doc """
  Initiates payment for pending order.
  Returns client secret for frontend payment form.
  """
  def initiate_payment(_, %{order_id: order_id}, %{context: %{current_user: user}})
      when not is_nil(user) do
    with order when not is_nil(order) <- Orders.get_order(order_id, user),
         :ok <- verify_buyer(order, user),
         {:ok, updated_order} <- Orders.initiate_payment(order) do
      # In production, client_secret comes from Stripe payment intent
      {:ok,
       %{
         client_secret: "cs_#{updated_order.payment_intent_id}",
         order: updated_order
       }}
    else
      nil -> {:error, "Order not found"}
      {:error, :not_buyer} -> {:error, "Only buyer can initiate payment"}
      {:error, changeset} -> {:error, format_error(changeset)}
    end
  end

  def initiate_payment(_, _, _), do: {:error, "Authentication required"}

  @doc """
  Marks order as shipped. Seller only.
  """
  def mark_shipped(_, args, %{context: %{current_user: user}}) when not is_nil(user) do
    with order when not is_nil(order) <- Orders.get_order(args.order_id, user),
         :ok <- verify_seller(order, user),
         tracking_info <- build_tracking_info(args),
         {:ok, updated} <- Orders.mark_shipped(order, tracking_info) do
      broadcast_order_updated(updated)
      {:ok, updated}
    else
      nil -> {:error, "Order not found"}
      {:error, :not_seller} -> {:error, "Only seller can mark order as shipped"}
      {:error, changeset} -> {:error, format_error(changeset)}
    end
  end

  def mark_shipped(_, _, _), do: {:error, "Authentication required"}

  @doc """
  Completes order and releases funds to seller. Buyer only.
  """
  def complete_order(_, %{order_id: order_id}, %{context: %{current_user: user}})
      when not is_nil(user) do
    with order when not is_nil(order) <- Orders.get_order(order_id, user),
         :ok <- verify_buyer(order, user),
         {:ok, completed} <- Orders.complete_order(order) do
      broadcast_order_updated(completed)
      {:ok, completed}
    else
      nil -> {:error, "Order not found"}
      {:error, :not_buyer} -> {:error, "Only buyer can complete order"}
      {:error, changeset} -> {:error, format_error(changeset)}
    end
  end

  def complete_order(_, _, _), do: {:error, "Authentication required"}

  # Private helpers

  defp verify_buyer(order, user) do
    if order.buyer_id == user.id, do: :ok, else: {:error, :not_buyer}
  end

  defp verify_seller(order, user) do
    is_seller = Enum.any?(order.items, &(&1.seller_id == user.id))
    if is_seller, do: :ok, else: {:error, :not_seller}
  end

  defp build_tracking_info(args) do
    %{
      tracking_number: args[:tracking_number],
      carrier: args[:carrier]
    }
    |> Enum.reject(fn {_, v} -> is_nil(v) end)
    |> Map.new()
  end

  defp broadcast_order_created(order) do
    Phoenix.PubSub.broadcast(
      Marketplace.PubSub,
      "orders:#{order.id}",
      {:order_created, order}
    )
  end

  defp broadcast_order_updated(order) do
    Phoenix.PubSub.broadcast(
      Marketplace.PubSub,
      "orders:#{order.id}",
      {:order_updated, order}
    )
  end

  defp format_error(%Ecto.Changeset{} = changeset) do
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

  defp format_error(reason) when is_atom(reason), do: to_string(reason)
  defp format_error(reason), do: inspect(reason)

  defp Map.from_struct_if_needed(%{__struct__: _} = struct), do: Map.from_struct(struct)
  defp Map.from_struct_if_needed(map) when is_map(map), do: map
end
