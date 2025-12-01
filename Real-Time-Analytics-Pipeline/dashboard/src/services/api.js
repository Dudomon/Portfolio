import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchMetrics = async () => {
  try {
    const response = await api.get('/metrics/realtime');
    return response.data;
  } catch (error) {
    console.error('Error fetching metrics:', error);
    return {};
  }
};

export const fetchTimeSeries = async (window = '1h') => {
  try {
    const response = await api.get('/metrics/timeseries', {
      params: { window }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching time series:', error);
    return { data: [] };
  }
};

export const fetchTopProducts = async (limit = 10) => {
  try {
    const response = await api.get('/products/top', {
      params: { limit }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching top products:', error);
    return { products: [] };
  }
};

export const fetchGeoDistribution = async () => {
  try {
    const response = await api.get('/geo/distribution');
    return response.data;
  } catch (error) {
    console.error('Error fetching geo distribution:', error);
    return { distribution: [] };
  }
};

export const fetchAlerts = async (limit = 50) => {
  try {
    const response = await api.get('/alerts', {
      params: { limit }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching alerts:', error);
    return { alerts: [] };
  }
};

export const sendEvent = async (event) => {
  try {
    const response = await api.post('/events', event);
    return response.data;
  } catch (error) {
    console.error('Error sending event:', error);
    throw error;
  }
};

export default api;
