# services/websocket_manager.py
from fastapi import WebSocket
from typing import List
import asyncio
import json
from datetime import datetime

# Import the market service
from services.market_service import update_market_data

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.running = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Start the periodic update task if not already running
        if not self.running:
            asyncio.create_task(self.periodic_update())
            self.running = True

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
            
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting message: {e}")
                # Remove broken connections
                try:
                    self.active_connections.remove(connection)
                except:
                    pass

    async def periodic_update(self):
        """Periodically fetch and broadcast market data"""
        while True:
            try:
                # Update market data
                market_data = await update_market_data()
                
                # Broadcast to all connected clients
                await self.broadcast(market_data)
                
                # Wait before next update (5 minutes)
                await asyncio.sleep(300)
            except Exception as e:
                print(f"Error in periodic update: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

# Create a singleton instance
manager = ConnectionManager()