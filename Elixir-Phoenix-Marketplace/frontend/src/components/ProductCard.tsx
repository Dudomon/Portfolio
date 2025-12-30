import { Link } from "react-router-dom";

/**
 * Product card for grid listing display.
 *
 * Design decisions:
 * - Image uses aspect-ratio CSS instead of fixed dimensions.
 *   Prevents layout shift during image load and handles varying
 *   aspect ratios from user uploads.
 * - Price formatting uses Intl.NumberFormat for locale awareness.
 *   A $1,000 watch displays as "1.000,00" for Brazilian users.
 * - Seller badge shows verification status. Verified sellers
 *   have completed identity verification; this builds buyer trust
 *   for high value items.
 */

interface Seller {
  id: string;
  displayName: string;
  avatarUrl: string | null;
  verifiedSeller: boolean;
  rating: string;
}

interface ProductCardProps {
  id: string;
  title: string;
  price: string;
  originalPrice: string | null;
  savingsPercent: number | null;
  condition: string;
  brand: string;
  images: string[];
  seller: Seller;
}

const CONDITION_LABELS: Record<string, string> = {
  NEW_WITH_TAGS: "New with tags",
  NEW_WITHOUT_TAGS: "New without tags",
  EXCELLENT: "Excellent",
  VERY_GOOD: "Very good",
  GOOD: "Good",
  FAIR: "Fair",
};

export function ProductCard({
  id,
  title,
  price,
  originalPrice,
  savingsPercent,
  condition,
  brand,
  images,
  seller,
}: ProductCardProps): JSX.Element {
  const formattedPrice = formatCurrency(price);
  const formattedOriginal = originalPrice ? formatCurrency(originalPrice) : null;
  const primaryImage = images[0] || "/placeholder.png";

  return (
    <article className="product-card">
      <Link to={`/products/${id}`} className="product-card__link">
        <div className="product-card__image-container">
          <img
            src={primaryImage}
            alt={title}
            className="product-card__image"
            loading="lazy"
          />
          {savingsPercent && savingsPercent > 0 && (
            <span className="product-card__badge product-card__badge--savings">
              {savingsPercent}% off
            </span>
          )}
        </div>

        <div className="product-card__content">
          <p className="product-card__brand">{brand}</p>
          <h3 className="product-card__title">{title}</h3>

          <div className="product-card__pricing">
            <span className="product-card__price">{formattedPrice}</span>
            {formattedOriginal && (
              <span className="product-card__original-price">
                {formattedOriginal}
              </span>
            )}
          </div>

          <p className="product-card__condition">
            {CONDITION_LABELS[condition] || condition}
          </p>
        </div>
      </Link>

      <div className="product-card__seller">
        <Link to={`/sellers/${seller.id}`} className="product-card__seller-link">
          {seller.avatarUrl ? (
            <img
              src={seller.avatarUrl}
              alt=""
              className="product-card__seller-avatar"
            />
          ) : (
            <div className="product-card__seller-avatar product-card__seller-avatar--placeholder" />
          )}
          <span className="product-card__seller-name">
            {seller.displayName}
            {seller.verifiedSeller && (
              <VerifiedBadge />
            )}
          </span>
        </Link>
      </div>
    </article>
  );
}

function VerifiedBadge(): JSX.Element {
  return (
    <svg
      className="verified-badge"
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="currentColor"
      aria-label="Verified seller"
    >
      <path d="M8 0L9.79 2.09L12.5 1.55L12.96 4.26L15.5 5.5L14.39 8L15.5 10.5L12.96 11.74L12.5 14.45L9.79 13.91L8 16L6.21 13.91L3.5 14.45L3.04 11.74L0.5 10.5L1.61 8L0.5 5.5L3.04 4.26L3.5 1.55L6.21 2.09L8 0Z" />
      <path
        d="M6.5 8L7.5 9L9.5 7"
        stroke="white"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

function formatCurrency(value: string): string {
  const num = parseFloat(value);
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(num);
}
