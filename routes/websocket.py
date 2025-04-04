# routes/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.websocket_manager import manager
from services.market_service import get_market_data

router = APIRouter()

@router.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send current market data immediately upon connection
        current_data = await get_market_data()
        await websocket.send_json(current_data)
        
        # Keep the connection open and handle incoming messages
        while True:
            data = await websocket.receive_text()
            # You could handle specific client commands here if needed
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)