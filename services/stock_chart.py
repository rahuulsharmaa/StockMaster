# services/stock_chart.py
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta

def generate_stock_data(symbol="TSLA", days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Initial price and generate random price movements
    base_price = 780
    volatility = 2.5
    
    # Generate random walk
    random_walk = np.random.normal(0, volatility, len(dates))
    price_changes = np.cumsum(random_walk)
    prices = base_price + price_changes
    
    # Ensure no negative prices
    prices = np.maximum(prices, 1)
    
    # Create volume data
    volume = np.random.randint(500000, 5000000, size=len(dates))
    
    # Create DataFrame
    stock_data = pd.DataFrame({
        'Date': dates,
        'Open': prices * np.random.uniform(0.98, 1.0, len(dates)),
        'High': prices * np.random.uniform(1.0, 1.05, len(dates)),
        'Low': prices * np.random.uniform(0.95, 1.0, len(dates)),
        'Close': prices,
        'Volume': volume
    })
    
    return stock_data

def create_stock_chart():
    stock_data = generate_stock_data(days=30)
    
    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Price'
    )])
    
    # Add volume bars as a subplot
    fig.add_trace(go.Bar(
        x=stock_data['Date'], 
        y=stock_data['Volume'],
        name='Volume',
        yaxis='y2',
        marker_color='rgba(0, 150, 255, 0.3)'
    ))
    
    # Calculate current price and change
    current_price = stock_data['Close'].iloc[-1]
    previous_price = stock_data['Close'].iloc[-2]
    price_change = current_price - previous_price
    price_change_percent = (price_change / previous_price) * 100
    
    # Update layout
    fig.update_layout(
        title=f'TSLA - ${current_price:.2f} ({price_change_percent:.2f}%)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified',
        yaxis=dict(domain=[0.3, 1.0]),
        yaxis2=dict(domain=[0, 0.2], title='Volume', showticklabels=False),
        xaxis=dict(
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
    )
    
    # Configure color based on price change
    color = 'green' if price_change >= 0 else 'red'
    fig.update_layout(title_font_color=color)
    
    # Convert to JSON
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return chart_json