from fastapi import APIRouter, HTTPException
from services.prediction_service import get_stock_prediction
import numpy as np

router = APIRouter()

@router.get("/api/stock/prediction/{symbol}")
async def prediction_route(symbol: str):
    try:
        # Call the service function to get prediction
        prediction_data = await get_stock_prediction(symbol)
        
        # Convert NumPy types to Python native types to make them JSON serializable
        if prediction_data and prediction_data.get('success', False):
            for key, value in prediction_data.items():
                if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                    prediction_data[key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
                    
            # Handle NumPy arrays if any
            for key, value in prediction_data.items():
                if isinstance(value, np.ndarray):
                    prediction_data[key] = value.tolist()
        
        return prediction_data
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "direction": None,
            "prediction_score": 50,
            "confidence": "Low"
        }