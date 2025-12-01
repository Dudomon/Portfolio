import React, { useState, useEffect } from 'react';
import { fetchAlerts } from '../services/api';
import './AlertFeed.css';

const AlertFeed = () => {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const loadAlerts = async () => {
      try {
        const data = await fetchAlerts(20);
        setAlerts(data.alerts || []);
      } catch (error) {
        console.error('Error loading alerts:', error);
      }
    };

    loadAlerts();
    const interval = setInterval(loadAlerts, 15000); // Refresh every 15 seconds

    return () => clearInterval(interval);
  }, []);

  const getSeverityColor = (severity) => {
    const colors = {
      low: '#4CAF50',
      medium: '#FF9800',
      high: '#f44336',
      critical: '#9C27B0'
    };
    return colors[severity] || '#666';
  };

  const getSeverityIcon = (severity) => {
    const icons = {
      low: 'ℹ️',
      medium: '⚠️',
      high: '🚨',
      critical: '🔥'
    };
    return icons[severity] || '📢';
  };

  return (
    <div className="alert-feed">
      {alerts.length === 0 ? (
        <p className="no-alerts">No alerts at this time ✅</p>
      ) : (
        <div className="alerts-list">
          {alerts.map((alert) => (
            <div 
              key={alert.alert_id} 
              className={`alert-item ${alert.resolved ? 'resolved' : ''}`}
              style={{ borderLeftColor: getSeverityColor(alert.severity) }}
            >
              <div className="alert-icon">
                {getSeverityIcon(alert.severity)}
              </div>
              <div className="alert-content">
                <div className="alert-header">
                  <span className="alert-type">{alert.alert_type}</span>
                  <span 
                    className="alert-severity"
                    style={{ background: getSeverityColor(alert.severity) }}
                  >
                    {alert.severity}
                  </span>
                </div>
                <p className="alert-message">{alert.message}</p>
                <div className="alert-footer">
                  <span className="alert-metric">
                    {alert.metric_name}: {alert.metric_value.toFixed(2)} 
                    {alert.threshold && ` (threshold: ${alert.threshold.toFixed(2)})`}
                  </span>
                  <span className="alert-time">
                    {new Date(alert.timestamp).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AlertFeed;
