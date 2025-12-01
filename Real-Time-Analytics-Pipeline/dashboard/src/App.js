import React, { useState, useEffect } from 'react';
import './App.css';
import MetricCard from './components/MetricCard';
import TimeSeriesChart from './components/TimeSeriesChart';
import TopProducts from './components/TopProducts';
import GeoDistribution from './components/GeoDistribution';
import AlertFeed from './components/AlertFeed';
import { useWebSocket } from './hooks/useWebSocket';
import { fetchMetrics, fetchTimeSeries, fetchTopProducts, fetchGeoDistribution } from './services/api';

function App() {
  const [metrics, setMetrics] = useState({
    total_events: 0,
    unique_users: 0,
    total_revenue: 0,
    avg_order_value: 0
  });
  
  const [timeSeries, setTimeSeries] = useState([]);
  const [topProducts, setTopProducts] = useState([]);
  const [geoData, setGeoData] = useState([]);
  const [timeWindow, setTimeWindow] = useState('1h');
  const [isConnected, setIsConnected] = useState(false);

  // WebSocket connection for real-time updates
  const wsMessage = useWebSocket('ws://localhost:8080/ws', {
    onOpen: () => setIsConnected(true),
    onClose: () => setIsConnected(false)
  });

  // Update metrics from WebSocket
  useEffect(() => {
    if (wsMessage && wsMessage.type === 'metrics_update') {
      setMetrics(wsMessage.data);
    }
  }, [wsMessage]);

  // Fetch initial data and set up polling
  useEffect(() => {
    const loadData = async () => {
      try {
        const [metricsData, timeSeriesData, productsData, geoDataResult] = await Promise.all([
          fetchMetrics(),
          fetchTimeSeries(timeWindow),
          fetchTopProducts(10),
          fetchGeoDistribution()
        ]);

        setMetrics(metricsData);
        setTimeSeries(timeSeriesData.data || []);
        setTopProducts(productsData.products || []);
        setGeoData(geoDataResult.distribution || []);
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };

    loadData();
    const interval = setInterval(loadData, 10000); // Refresh every 10 seconds

    return () => clearInterval(interval);
  }, [timeWindow]);

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1>📊 Real-Time Analytics Dashboard</h1>
          <div className="connection-status">
            <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></span>
            <span>{isConnected ? 'Live' : 'Disconnected'}</span>
          </div>
        </div>
      </header>

      <main className="dashboard-content">
        {/* Metric Cards */}
        <section className="metrics-grid">
          <MetricCard
            title="Total Events"
            value={metrics.total_events?.toLocaleString() || '0'}
            icon="📈"
            color="#4CAF50"
          />
          <MetricCard
            title="Unique Users"
            value={metrics.unique_users?.toLocaleString() || '0'}
            icon="👥"
            color="#2196F3"
          />
          <MetricCard
            title="Total Revenue"
            value={`$${(metrics.total_revenue || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
            icon="💰"
            color="#FF9800"
          />
          <MetricCard
            title="Avg Order Value"
            value={`$${(metrics.avg_order_value || 0).toFixed(2)}`}
            icon="🛒"
            color="#9C27B0"
          />
        </section>

        {/* Time Series Chart */}
        <section className="chart-section">
          <div className="section-header">
            <h2>Revenue Over Time</h2>
            <div className="time-window-selector">
              {['1m', '5m', '15m', '1h', '24h'].map(window => (
                <button
                  key={window}
                  className={`window-btn ${timeWindow === window ? 'active' : ''}`}
                  onClick={() => setTimeWindow(window)}
                >
                  {window}
                </button>
              ))}
            </div>
          </div>
          <TimeSeriesChart data={timeSeries} />
        </section>

        {/* Two Column Layout */}
        <section className="two-column-grid">
          <div className="column">
            <h2>Top Products</h2>
            <TopProducts products={topProducts} />
          </div>
          <div className="column">
            <h2>Geographic Distribution</h2>
            <GeoDistribution data={geoData} />
          </div>
        </section>

        {/* Alert Feed */}
        <section className="alert-section">
          <h2>Recent Alerts</h2>
          <AlertFeed />
        </section>
      </main>

      <footer className="App-footer">
        <p>Built with ❤️ by Eduardo Peiter | Real-Time Analytics Pipeline</p>
      </footer>
    </div>
  );
}

export default App;
