import { useState, useCallback } from "react";
import { useQuery } from "@apollo/client";
import { useSearchParams } from "react-router-dom";
import { GET_PRODUCTS, GET_CATEGORIES } from "../graphql/operations";
import { ProductCard } from "../components/ProductCard";

/**
 * Product listing page with filtering, sorting, and infinite scroll.
 *
 * URL driven state: All filter selections reflect in URL query params.
 * This enables shareable filtered views and browser back/forward navigation.
 * Users can bookmark "watches under $500" and return to exactly that view.
 *
 * Pagination uses cursor based infinite scroll over numbered pages.
 * For marketplace browsing, users scroll through items; they rarely need
 * to jump to "page 47". Infinite scroll matches the browsing mental model.
 */

type ProductSort = "NEWEST" | "PRICE_ASC" | "PRICE_DESC";
type Condition = "NEW_WITH_TAGS" | "NEW_WITHOUT_TAGS" | "EXCELLENT" | "VERY_GOOD" | "GOOD" | "FAIR";

interface FilterState {
  categoryId: string | null;
  minPrice: string | null;
  maxPrice: string | null;
  conditions: Condition[];
  search: string | null;
  sort: ProductSort;
}

const ITEMS_PER_PAGE = 24;

export function ProductListPage(): JSX.Element {
  const [searchParams, setSearchParams] = useSearchParams();
  const [page, setPage] = useState(1);

  // Parse filters from URL
  const filters: FilterState = {
    categoryId: searchParams.get("category"),
    minPrice: searchParams.get("minPrice"),
    maxPrice: searchParams.get("maxPrice"),
    conditions: parseConditions(searchParams.get("condition")),
    search: searchParams.get("q"),
    sort: parseSort(searchParams.get("sort")),
  };

  const { data, loading, error, fetchMore } = useQuery(GET_PRODUCTS, {
    variables: {
      categoryId: filters.categoryId,
      minPrice: filters.minPrice,
      maxPrice: filters.maxPrice,
      condition: filters.conditions.length > 0 ? filters.conditions : null,
      search: filters.search,
      sort: filters.sort,
      page: 1,
      perPage: ITEMS_PER_PAGE,
    },
    notifyOnNetworkStatusChange: true,
  });

  const { data: categoriesData } = useQuery(GET_CATEGORIES);

  const updateFilter = useCallback(
    (key: string, value: string | null) => {
      const params = new URLSearchParams(searchParams);
      if (value) {
        params.set(key, value);
      } else {
        params.delete(key);
      }
      setSearchParams(params);
      setPage(1);
    },
    [searchParams, setSearchParams]
  );

  const loadMore = useCallback(() => {
    if (!data?.products) return;

    const nextPage = page + 1;
    if (nextPage > data.products.totalPages) return;

    fetchMore({
      variables: { page: nextPage },
    }).then(() => {
      setPage(nextPage);
    });
  }, [data, page, fetchMore]);

  if (error) {
    return (
      <div className="error-state">
        <p>Failed to load products. Please try again.</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  const products = data?.products?.items || [];
  const totalCount = data?.products?.totalCount || 0;
  const hasMore = page < (data?.products?.totalPages || 0);

  return (
    <div className="product-list-page">
      <aside className="filters">
        <SearchInput
          value={filters.search || ""}
          onChange={(value) => updateFilter("q", value || null)}
        />

        <CategoryFilter
          categories={categoriesData?.categories || []}
          selected={filters.categoryId}
          onChange={(id) => updateFilter("category", id)}
        />

        <PriceRangeFilter
          min={filters.minPrice}
          max={filters.maxPrice}
          onMinChange={(value) => updateFilter("minPrice", value)}
          onMaxChange={(value) => updateFilter("maxPrice", value)}
        />

        <ConditionFilter
          selected={filters.conditions}
          onChange={(conditions) =>
            updateFilter("condition", conditions.join(",") || null)
          }
        />

        <SortSelect
          value={filters.sort}
          onChange={(sort) => updateFilter("sort", sort)}
        />
      </aside>

      <main className="product-list">
        <header className="product-list__header">
          <h1>
            {filters.search
              ? `Results for "${filters.search}"`
              : "All Products"}
          </h1>
          <p className="product-list__count">
            {totalCount.toLocaleString()} items
          </p>
        </header>

        {loading && products.length === 0 ? (
          <ProductGridSkeleton />
        ) : products.length === 0 ? (
          <EmptyState search={filters.search} />
        ) : (
          <>
            <div className="product-grid">
              {products.map((product: ProductData) => (
                <ProductCard key={product.id} {...product} />
              ))}
            </div>

            {hasMore && (
              <button
                className="load-more-button"
                onClick={loadMore}
                disabled={loading}
              >
                {loading ? "Loading..." : "Load more"}
              </button>
            )}
          </>
        )}
      </main>
    </div>
  );
}

// Sub components

interface SearchInputProps {
  value: string;
  onChange: (value: string) => void;
}

function SearchInput({ value, onChange }: SearchInputProps): JSX.Element {
  const [localValue, setLocalValue] = useState(value);

  // Debounce search to avoid query on every keystroke
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onChange(localValue);
  };

  return (
    <form className="search-input" onSubmit={handleSubmit}>
      <input
        type="search"
        placeholder="Search products..."
        value={localValue}
        onChange={(e) => setLocalValue(e.target.value)}
      />
      <button type="submit">Search</button>
    </form>
  );
}

interface Category {
  id: string;
  name: string;
  slug: string;
  depth: number;
  productCount: number;
}

interface CategoryFilterProps {
  categories: Category[];
  selected: string | null;
  onChange: (id: string | null) => void;
}

function CategoryFilter({
  categories,
  selected,
  onChange,
}: CategoryFilterProps): JSX.Element {
  return (
    <div className="filter-group">
      <h3 className="filter-group__title">Category</h3>
      <ul className="category-list">
        <li>
          <button
            className={`category-item ${!selected ? "category-item--active" : ""}`}
            onClick={() => onChange(null)}
          >
            All Categories
          </button>
        </li>
        {categories.map((category) => (
          <li key={category.id} style={{ paddingLeft: `${category.depth * 16}px` }}>
            <button
              className={`category-item ${selected === category.id ? "category-item--active" : ""}`}
              onClick={() => onChange(category.id)}
            >
              {category.name}
              <span className="category-item__count">({category.productCount})</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

interface PriceRangeFilterProps {
  min: string | null;
  max: string | null;
  onMinChange: (value: string | null) => void;
  onMaxChange: (value: string | null) => void;
}

function PriceRangeFilter({
  min,
  max,
  onMinChange,
  onMaxChange,
}: PriceRangeFilterProps): JSX.Element {
  return (
    <div className="filter-group">
      <h3 className="filter-group__title">Price</h3>
      <div className="price-range">
        <input
          type="number"
          placeholder="Min"
          value={min || ""}
          onChange={(e) => onMinChange(e.target.value || null)}
          min="0"
        />
        <span>to</span>
        <input
          type="number"
          placeholder="Max"
          value={max || ""}
          onChange={(e) => onMaxChange(e.target.value || null)}
          min="0"
        />
      </div>
    </div>
  );
}

interface ConditionFilterProps {
  selected: Condition[];
  onChange: (conditions: Condition[]) => void;
}

const CONDITIONS: { value: Condition; label: string }[] = [
  { value: "NEW_WITH_TAGS", label: "New with tags" },
  { value: "NEW_WITHOUT_TAGS", label: "New without tags" },
  { value: "EXCELLENT", label: "Excellent" },
  { value: "VERY_GOOD", label: "Very good" },
  { value: "GOOD", label: "Good" },
  { value: "FAIR", label: "Fair" },
];

function ConditionFilter({
  selected,
  onChange,
}: ConditionFilterProps): JSX.Element {
  const toggle = (condition: Condition) => {
    if (selected.includes(condition)) {
      onChange(selected.filter((c) => c !== condition));
    } else {
      onChange([...selected, condition]);
    }
  };

  return (
    <div className="filter-group">
      <h3 className="filter-group__title">Condition</h3>
      <div className="checkbox-group">
        {CONDITIONS.map(({ value, label }) => (
          <label key={value} className="checkbox-item">
            <input
              type="checkbox"
              checked={selected.includes(value)}
              onChange={() => toggle(value)}
            />
            {label}
          </label>
        ))}
      </div>
    </div>
  );
}

interface SortSelectProps {
  value: ProductSort;
  onChange: (sort: ProductSort) => void;
}

function SortSelect({ value, onChange }: SortSelectProps): JSX.Element {
  return (
    <div className="filter-group">
      <label htmlFor="sort" className="filter-group__title">
        Sort by
      </label>
      <select
        id="sort"
        value={value}
        onChange={(e) => onChange(e.target.value as ProductSort)}
      >
        <option value="NEWEST">Newest first</option>
        <option value="PRICE_ASC">Price: low to high</option>
        <option value="PRICE_DESC">Price: high to low</option>
      </select>
    </div>
  );
}

function ProductGridSkeleton(): JSX.Element {
  return (
    <div className="product-grid">
      {Array.from({ length: 12 }).map((_, i) => (
        <div key={i} className="product-card-skeleton">
          <div className="skeleton skeleton--image" />
          <div className="skeleton skeleton--text" />
          <div className="skeleton skeleton--text skeleton--short" />
        </div>
      ))}
    </div>
  );
}

interface EmptyStateProps {
  search: string | null;
}

function EmptyState({ search }: EmptyStateProps): JSX.Element {
  return (
    <div className="empty-state">
      <h2>No products found</h2>
      {search ? (
        <p>No results for "{search}". Try a different search term.</p>
      ) : (
        <p>No products match your filters. Try adjusting your criteria.</p>
      )}
    </div>
  );
}

// Utility functions

function parseConditions(param: string | null): Condition[] {
  if (!param) return [];
  return param.split(",").filter(isValidCondition);
}

function isValidCondition(value: string): value is Condition {
  return CONDITIONS.some((c) => c.value === value);
}

function parseSort(param: string | null): ProductSort {
  if (param === "PRICE_ASC" || param === "PRICE_DESC") return param;
  return "NEWEST";
}

interface ProductData {
  id: string;
  title: string;
  price: string;
  originalPrice: string | null;
  savingsPercent: number | null;
  condition: string;
  brand: string;
  images: string[];
  seller: {
    id: string;
    displayName: string;
    avatarUrl: string | null;
    verifiedSeller: boolean;
    rating: string;
  };
}
