import axios from 'axios';

const API_URL = "http://localhost:8000";  // Update this if you're using a cloud backend

export const predictStock = async (stockData) => {
  try {
    const response = await axios.post(`${API_URL}/predict-stock`, stockData);
    return response.data;
  } catch (error) {
    console.error("Error predicting stock:", error);
    throw error;
  }
};
