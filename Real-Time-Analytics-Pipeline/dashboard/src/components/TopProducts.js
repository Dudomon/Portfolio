import React from 'react';
import './TopProducts.css';

const TopProducts = ({ products }) => {
  const maxSales = products.length > 0 ? Math.max(...products.map(p => p.sales_count)) : 1;

  return (
    <div className="top-products">
      {products.length === 0 ? (
        <p className="no-data">No data available</p>
      ) : (
        <div className="products-list">
          {products.map((product, index) => (
            <div key={product.product_id} className="product-item">
              <div className="product-rank">#{index + 1}</div>
              <div className="product-info">
                <div className="product-header">
                  <span className="product-name">{product.name}</span>
                  <span className="product-category">{product.category}</span>
                </div>
                <div className="product-stats">
                  <span className="sales-count">{product.sales_count} sales</span>
                  <span className="revenue">${product.total_revenue.toFixed(2)}</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${(product.sales_count / maxSales) * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TopProducts;
