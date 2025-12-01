import React from 'react';
import './MetricCard.css';

const MetricCard = ({ title, value, icon, color }) => {
  return (
    <div className="metric-card" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="metric-icon" style={{ background: `${color}20` }}>
        <span style={{ fontSize: '2rem' }}>{icon}</span>
      </div>
      <div className="metric-content">
        <h3 className="metric-title">{title}</h3>
        <p className="metric-value" style={{ color }}>{value}</p>
      </div>
    </div>
  );
};

export default MetricCard;
