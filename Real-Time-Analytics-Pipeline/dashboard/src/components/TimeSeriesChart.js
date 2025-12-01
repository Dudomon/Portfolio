import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './TimeSeriesChart.css';

const TimeSeriesChart = ({ data }) => {
  const formatData = data.map(item => ({
    time: new Date(item.timestamp).toLocaleTimeString(),
    revenue: item.revenue,
    events: item.events,
    users: item.users
  }));

  return (
    <div className="timeseries-chart">
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={formatData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="time" 
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
          <Legend 
            wrapperStyle={{
              paddingTop: '20px'
            }}
          />
          <Line 
            type="monotone" 
            dataKey="revenue" 
            stroke="#FF9800" 
            strokeWidth={3}
            dot={{ fill: '#FF9800', r: 4 }}
            activeDot={{ r: 6 }}
            name="Revenue ($)"
          />
          <Line 
            type="monotone" 
            dataKey="events" 
            stroke="#4CAF50" 
            strokeWidth={2}
            dot={{ fill: '#4CAF50', r: 3 }}
            name="Events"
          />
          <Line 
            type="monotone" 
            dataKey="users" 
            stroke="#2196F3" 
            strokeWidth={2}
            dot={{ fill: '#2196F3', r: 3 }}
            name="Users"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TimeSeriesChart;
