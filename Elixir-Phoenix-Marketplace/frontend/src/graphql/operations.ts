import { gql } from "@apollo/client";

/**
 * GraphQL operations for the marketplace.
 *
 * Fragment usage:
 * - ProductCard: Minimal fields for list views. Keeps payload small.
 * - ProductDetail: Full fields for single product view.
 * - UserSummary: Public seller info without sensitive fields.
 *
 * Fragments prevent field duplication and ensure consistency.
 * When product card design changes, update one fragment instead of
 * hunting through multiple queries.
 */

// Fragments define reusable field selections

export const USER_SUMMARY_FRAGMENT = gql`
  fragment UserSummary on User {
    id
    displayName
    avatarUrl
    verifiedSeller
    rating
    ratingCount
  }
`;

export const PRODUCT_CARD_FRAGMENT = gql`
  fragment ProductCard on Product {
    id
    title
    price
    originalPrice
    savingsPercent
    condition
    brand
    images
    seller {
      ...UserSummary
    }
  }
  ${USER_SUMMARY_FRAGMENT}
`;

export const PRODUCT_DETAIL_FRAGMENT = gql`
  fragment ProductDetail on Product {
    id
    title
    description
    price
    originalPrice
    savingsPercent
    condition
    status
    brand
    images
    attributes
    viewCount
    favoriteCount
    insertedAt
    seller {
      ...UserSummary
      bio
    }
    category {
      id
      name
      slug
      path
    }
  }
  ${USER_SUMMARY_FRAGMENT}
`;

export const ORDER_FRAGMENT = gql`
  fragment OrderFields on Order {
    id
    status
    subtotal
    shippingCost
    taxAmount
    total
    currency
    shippingAddress {
      street
      city
      state
      postalCode
      country
    }
    paidAt
    shippedAt
    completedAt
    insertedAt
    items {
      id
      price
      productSnapshot
      seller {
        ...UserSummary
      }
    }
  }
  ${USER_SUMMARY_FRAGMENT}
`;

// Queries

export const GET_PRODUCTS = gql`
  query GetProducts(
    $categoryId: ID
    $sellerId: ID
    $minPrice: Decimal
    $maxPrice: Decimal
    $condition: [Condition!]
    $search: String
    $sort: ProductSort
    $page: Int
    $perPage: Int
  ) {
    products(
      categoryId: $categoryId
      sellerId: $sellerId
      minPrice: $minPrice
      maxPrice: $maxPrice
      condition: $condition
      search: $search
      sort: $sort
      page: $page
      perPage: $perPage
    ) {
      items {
        ...ProductCard
      }
      totalCount
      page
      perPage
      totalPages
    }
  }
  ${PRODUCT_CARD_FRAGMENT}
`;

export const GET_PRODUCT = gql`
  query GetProduct($id: ID!) {
    product(id: $id) {
      ...ProductDetail
    }
  }
  ${PRODUCT_DETAIL_FRAGMENT}
`;

export const GET_CATEGORIES = gql`
  query GetCategories {
    categories {
      id
      name
      slug
      path
      depth
      productCount
    }
  }
`;

export const GET_ME = gql`
  query GetMe {
    me {
      id
      email
      displayName
      avatarUrl
      bio
      verifiedSeller
      rating
      ratingCount
    }
  }
`;

export const GET_MY_ORDERS = gql`
  query GetMyOrders($role: OrderRole, $page: Int, $perPage: Int) {
    myOrders(role: $role, page: $page, perPage: $perPage) {
      ...OrderFields
    }
  }
  ${ORDER_FRAGMENT}
`;

export const GET_ORDER = gql`
  query GetOrder($id: ID!) {
    order(id: $id) {
      ...OrderFields
      buyer {
        ...UserSummary
      }
    }
  }
  ${ORDER_FRAGMENT}
  ${USER_SUMMARY_FRAGMENT}
`;

// Mutations

export const LOGIN = gql`
  mutation Login($email: String!, $password: String!) {
    login(email: $email, password: $password) {
      token
      user {
        id
        email
        displayName
        avatarUrl
        verifiedSeller
      }
    }
  }
`;

export const REGISTER = gql`
  mutation Register($email: String!, $password: String!, $displayName: String!) {
    register(email: $email, password: $password, displayName: $displayName) {
      token
      user {
        id
        email
        displayName
      }
    }
  }
`;

export const CREATE_PRODUCT = gql`
  mutation CreateProduct($input: ProductInput!) {
    createProduct(input: $input) {
      ...ProductDetail
    }
  }
  ${PRODUCT_DETAIL_FRAGMENT}
`;

export const PUBLISH_PRODUCT = gql`
  mutation PublishProduct($id: ID!) {
    publishProduct(id: $id) {
      id
      status
    }
  }
`;

export const CREATE_ORDER = gql`
  mutation CreateOrder($productIds: [ID!]!, $shippingAddress: AddressInput!) {
    createOrder(productIds: $productIds, shippingAddress: $shippingAddress) {
      ...OrderFields
    }
  }
  ${ORDER_FRAGMENT}
`;

export const INITIATE_PAYMENT = gql`
  mutation InitiatePayment($orderId: ID!) {
    initiatePayment(orderId: $orderId) {
      clientSecret
      order {
        id
        status
      }
    }
  }
`;

export const MARK_SHIPPED = gql`
  mutation MarkShipped($orderId: ID!, $trackingNumber: String, $carrier: String) {
    markShipped(orderId: $orderId, trackingNumber: $trackingNumber, carrier: $carrier) {
      id
      status
      shippedAt
    }
  }
`;

export const COMPLETE_ORDER = gql`
  mutation CompleteOrder($orderId: ID!) {
    completeOrder(orderId: $orderId) {
      id
      status
      completedAt
    }
  }
`;

// Subscriptions

export const PRODUCT_UPDATED = gql`
  subscription ProductUpdated($productId: ID!) {
    productUpdated(productId: $productId) {
      id
      status
      price
    }
  }
`;

export const ORDER_UPDATED = gql`
  subscription OrderUpdated($orderId: ID!) {
    orderUpdated(orderId: $orderId) {
      id
      status
      shippedAt
      completedAt
    }
  }
`;
