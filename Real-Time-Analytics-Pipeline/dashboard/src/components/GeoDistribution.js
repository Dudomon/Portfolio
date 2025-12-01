import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './GeoDistribution.css';

const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140', '#30cfd0'];

const GeoDistribution = ({ data }) => {
  const formatData = data.map(item => ({
    country: item.country,
    revenue: item.revenue,
    users: item.users
  }));

  return (
    <div className="geo-distribution">
      {data.length === 0 ? (
        <p className="no-data">No data available</p>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={formatData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="country" 
                stroke="#666"
                style={{ fontSize: '0.85rem' }}
              />
              <YAxis 
                stroke="#666"
                style={{ fontSize: '0.85rem' }}
              />
              <Tooltip 
                contentStyle={{
                  background: 'white',
                  border: '1px solid #ddd',
                  borderRadius: '8px',
                  padding: '10px'
                }}
              />
              <Bar dataKey="revenue" radius={[8, 8, 0, 0]}>
                {formatData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          
          <div className="geo-list">
            {data.map((item, index) => (
              <div key={item.country} className="geo-item">
                <div className="geo-country">
                  <span className="country-flag">🌍</span>
                  <span className="country-name">{item.country}</span>
                </div>
                <div className="geo-stats">
                  <span className="geo-revenue">${item.revenue.toFixed(2)}</span>
                  <span className="geo-users">{item.users} users</span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default GeoDistribution;
