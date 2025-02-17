import React, { useState } from 'react';
import { predictStock } from '../api/stock';

const PredictionCard = () => {
  const [stockData, setStockData] = useState({
    symbol: '',
    company_name: '',
    price: 0,
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    setStockData({
      ...stockData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const result = await predictStock(stockData);
    setPrediction(result);
  };

  return (
    <div>
      <h2>Stock Prediction</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          name="symbol"
          placeholder="Stock Symbol"
          value={stockData.symbol}
          onChange={handleChange}
        />
        <input
          type="text"
          name="company_name"
          placeholder="Company Name"
          value={stockData.company_name}
          onChange={handleChange}
        />
        <input
          type="number"
          name="price"
          placeholder="Stock Price"
          value={stockData.price}
          onChange={handleChange}
        />
        <button type="submit">Predict</button>
      </form>
      {prediction && (
        <div>
          <h3>Prediction for {prediction.symbol}: {prediction.price}</h3>
        </div>
      )}
    </div>
  );
};

export default PredictionCard;
