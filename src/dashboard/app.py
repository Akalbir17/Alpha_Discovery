import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import redis
import json
from datetime import datetime, timedelta
import sqlite3
import psycopg2
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import yfinance as yf
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Alpha Discovery Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    .neutral {
        color: #6c757d;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class DashboardConfig:
    """Configuration for dashboard components"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "alpha_discovery"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    refresh_interval: int = 5  # seconds
    max_data_points: int = 1000

class DataManager:
    """Manages data connections and retrieval"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.redis_client = None
        self.postgres_conn = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections"""
        try:
            # Redis connection
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            
            # PostgreSQL connection
            self.postgres_conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            # Use mock data for development
            self.redis_client = None
            self.postgres_conn = None
    
    def get_portfolio_data(self) -> Dict:
        """Get current portfolio data"""
        if self.redis_client:
            try:
                portfolio_data = self.redis_client.get("portfolio:current")
                if portfolio_data:
                    return json.loads(portfolio_data)
            except Exception as e:
                logger.error(f"Error fetching portfolio data: {e}")
        
        # Mock data for development
        return {
            "total_value": 1000000,
            "pnl_today": 15000,
            "pnl_total": 125000,
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "value": 15000, "pnl": 1500},
                {"symbol": "GOOGL", "quantity": 50, "value": 12500, "pnl": -500},
                {"symbol": "MSFT", "quantity": 75, "value": 22500, "pnl": 2000},
                {"symbol": "TSLA", "quantity": 25, "value": 5000, "pnl": -750},
            ]
        }
    
    def get_agent_consensus(self) -> Dict:
        """Get agent consensus data"""
        if self.redis_client:
            try:
                consensus_data = self.redis_client.get("agents:consensus")
                if consensus_data:
                    return json.loads(consensus_data)
            except Exception as e:
                logger.error(f"Error fetching agent consensus: {e}")
        
        # Mock data
        return {
            "overall_sentiment": "BULLISH",
            "confidence": 0.75,
            "agents": [
                {"name": "Technical Agent", "signal": "BUY", "confidence": 0.8, "reasoning": "Strong momentum indicators"},
                {"name": "Fundamental Agent", "signal": "HOLD", "confidence": 0.6, "reasoning": "Mixed earnings signals"},
                {"name": "Sentiment Agent", "signal": "BUY", "confidence": 0.9, "reasoning": "Positive social sentiment"},
                {"name": "Risk Agent", "signal": "HOLD", "confidence": 0.7, "reasoning": "Moderate volatility levels"},
            ],
            "debates": [
                {"topic": "Market Direction", "participants": ["Technical", "Fundamental"], "status": "Active"},
                {"topic": "Risk Level", "participants": ["Risk", "Sentiment"], "status": "Resolved"},
            ]
        }
    
    def get_market_microstructure(self) -> Dict:
        """Get market microstructure data"""
        if self.redis_client:
            try:
                microstructure_data = self.redis_client.get("market:microstructure")
                if microstructure_data:
                    return json.loads(microstructure_data)
            except Exception as e:
                logger.error(f"Error fetching microstructure data: {e}")
        
        # Mock data
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                 end=datetime.now(), freq='1min')
        return {
            "order_book": {
                "bids": [[100.5, 1000], [100.4, 1500], [100.3, 2000]],
                "asks": [[100.6, 1200], [100.7, 1800], [100.8, 2500]]
            },
            "trade_flow": {
                "timestamps": [t.isoformat() for t in timestamps[-20:]],
                "buy_volume": np.random.exponential(1000, 20).tolist(),
                "sell_volume": np.random.exponential(1000, 20).tolist()
            },
            "spread": {
                "timestamps": [t.isoformat() for t in timestamps[-50:]],
                "spreads": (np.random.normal(0.1, 0.02, 50)).tolist()
            }
        }
    
    def get_sentiment_data(self) -> Dict:
        """Get Reddit sentiment data"""
        if self.redis_client:
            try:
                sentiment_data = self.redis_client.get("sentiment:reddit")
                if sentiment_data:
                    return json.loads(sentiment_data)
            except Exception as e:
                logger.error(f"Error fetching sentiment data: {e}")
        
        # Mock data
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=7), 
                                 end=datetime.now(), freq='1H')
        return {
            "timestamps": [t.isoformat() for t in timestamps],
            "sentiment_scores": np.random.normal(0.1, 0.3, len(timestamps)).tolist(),
            "volume": np.random.poisson(100, len(timestamps)).tolist(),
            "top_mentions": [
                {"symbol": "AAPL", "mentions": 1250, "sentiment": 0.3},
                {"symbol": "TSLA", "mentions": 980, "sentiment": 0.1},
                {"symbol": "GME", "mentions": 750, "sentiment": 0.6},
                {"symbol": "NVDA", "mentions": 650, "sentiment": 0.4},
            ]
        }
    
    def get_regime_data(self) -> Dict:
        """Get regime detection data"""
        if self.redis_client:
            try:
                regime_data = self.redis_client.get("regime:current")
                if regime_data:
                    return json.loads(regime_data)
            except Exception as e:
                logger.error(f"Error fetching regime data: {e}")
        
        # Mock data
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                 end=datetime.now(), freq='1D')
        regimes = np.random.choice([0, 1, 2, 3], len(timestamps), p=[0.4, 0.3, 0.2, 0.1])
        
        return {
            "current_regime": "Bull Market",
            "confidence": 0.85,
            "regime_history": {
                "timestamps": [t.isoformat() for t in timestamps],
                "regimes": regimes.tolist(),
                "regime_names": ["Bull Market", "Bear Market", "High Volatility", "Low Volatility"]
            },
            "transition_probabilities": [
                [0.8, 0.1, 0.08, 0.02],
                [0.15, 0.7, 0.1, 0.05],
                [0.2, 0.2, 0.5, 0.1],
                [0.3, 0.1, 0.1, 0.5]
            ]
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        if self.postgres_conn:
            try:
                cursor = self.postgres_conn.cursor()
                cursor.execute("""
                    SELECT * FROM strategy_performance 
                    ORDER BY timestamp DESC LIMIT 100
                """)
                results = cursor.fetchall()
                if results:
                    # Process results into structured format
                    pass
            except Exception as e:
                logger.error(f"Error fetching performance metrics: {e}")
        
        # Mock data
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                 end=datetime.now(), freq='1D')
        returns = np.random.normal(0.001, 0.02, len(timestamps))
        cumulative_returns = np.cumprod(1 + returns) - 1
        
        return {
            "timestamps": [t.isoformat() for t in timestamps],
            "returns": returns.tolist(),
            "cumulative_returns": cumulative_returns.tolist(),
            "sharpe_ratio": 1.8,
            "sortino_ratio": 2.1,
            "max_drawdown": -0.08,
            "volatility": 0.15,
            "alpha": 0.05,
            "beta": 0.85,
            "win_rate": 0.65
        }
    
    def get_risk_metrics(self) -> Dict:
        """Get risk monitoring metrics"""
        if self.redis_client:
            try:
                risk_data = self.redis_client.get("risk:current")
                if risk_data:
                    return json.loads(risk_data)
            except Exception as e:
                logger.error(f"Error fetching risk metrics: {e}")
        
        # Mock data
        return {
            "var_95": -25000,
            "var_99": -45000,
            "expected_shortfall": -55000,
            "portfolio_beta": 0.85,
            "correlation_risk": 0.3,
            "concentration_risk": 0.25,
            "sector_exposure": {
                "Technology": 0.4,
                "Healthcare": 0.2,
                "Finance": 0.15,
                "Consumer": 0.15,
                "Energy": 0.1
            },
            "stress_test_results": {
                "2008_crisis": -0.35,
                "covid_crash": -0.28,
                "flash_crash": -0.15,
                "interest_rate_shock": -0.12
            }
        }

class DashboardComponents:
    """Contains all dashboard component functions"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def render_pnl_positions(self):
        """Render P&L and positions panel"""
        st.subheader("📊 Portfolio Overview")
        
        portfolio_data = self.data_manager.get_portfolio_data()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Value",
                f"${portfolio_data['total_value']:,.0f}",
                delta=f"${portfolio_data['pnl_today']:,.0f}"
            )
        
        with col2:
            st.metric(
                "Today's P&L",
                f"${portfolio_data['pnl_today']:,.0f}",
                delta=f"{(portfolio_data['pnl_today']/portfolio_data['total_value']*100):.2f}%"
            )
        
        with col3:
            st.metric(
                "Total P&L",
                f"${portfolio_data['pnl_total']:,.0f}",
                delta=f"{(portfolio_data['pnl_total']/portfolio_data['total_value']*100):.2f}%"
            )
        
        with col4:
            positions_count = len(portfolio_data['positions'])
            st.metric("Active Positions", positions_count)
        
        # Positions table
        st.subheader("Current Positions")
        positions_df = pd.DataFrame(portfolio_data['positions'])
        
        # Add color coding for P&L
        def color_pnl(val):
            if val > 0:
                return 'color: green'
            elif val < 0:
                return 'color: red'
            return 'color: gray'
        
        styled_positions = positions_df.style.applymap(color_pnl, subset=['pnl'])
        st.dataframe(styled_positions, use_container_width=True)
        
        # P&L chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=positions_df['symbol'],
            y=positions_df['pnl'],
            marker_color=['green' if x > 0 else 'red' for x in positions_df['pnl']],
            name='Position P&L'
        ))
        fig.update_layout(
            title="Position P&L",
            xaxis_title="Symbol",
            yaxis_title="P&L ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_agent_consensus(self):
        """Render agent consensus and debates panel"""
        st.subheader("🤖 Agent Consensus")
        
        consensus_data = self.data_manager.get_agent_consensus()
        
        # Overall sentiment
        col1, col2 = st.columns(2)
        with col1:
            sentiment_color = {
                'BULLISH': 'green',
                'BEARISH': 'red',
                'NEUTRAL': 'gray'
            }.get(consensus_data['overall_sentiment'], 'gray')
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Overall Sentiment</h3>
                <h2 style="color: {sentiment_color};">{consensus_data['overall_sentiment']}</h2>
                <p>Confidence: {consensus_data['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=consensus_data['confidence'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Consensus Confidence"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual agent signals
        st.subheader("Individual Agent Signals")
        agents_df = pd.DataFrame(consensus_data['agents'])
        
        for _, agent in agents_df.iterrows():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 3])
            
            with col1:
                st.write(f"**{agent['name']}**")
            
            with col2:
                signal_color = {
                    'BUY': 'green',
                    'SELL': 'red',
                    'HOLD': 'orange'
                }.get(agent['signal'], 'gray')
                st.markdown(f"<span style='color: {signal_color}; font-weight: bold;'>{agent['signal']}</span>", 
                          unsafe_allow_html=True)
            
            with col3:
                st.write(f"{agent['confidence']:.1%}")
            
            with col4:
                st.write(agent['reasoning'])
        
        # Active debates
        st.subheader("Active Debates")
        debates_df = pd.DataFrame(consensus_data['debates'])
        
        for _, debate in debates_df.iterrows():
            status_color = 'green' if debate['status'] == 'Resolved' else 'orange'
            st.markdown(f"""
            **{debate['topic']}** - <span style='color: {status_color};'>{debate['status']}</span>  
            Participants: {', '.join(debate['participants'])}
            """, unsafe_allow_html=True)
    
    def render_market_microstructure(self):
        """Render market microstructure panel"""
        st.subheader("📈 Market Microstructure")
        
        microstructure_data = self.data_manager.get_market_microstructure()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Order book
            st.subheader("Order Book")
            order_book = microstructure_data['order_book']
            
            fig = go.Figure()
            
            # Bids
            bids = np.array(order_book['bids'])
            fig.add_trace(go.Bar(
                x=bids[:, 1],
                y=bids[:, 0],
                orientation='h',
                name='Bids',
                marker_color='green',
                opacity=0.7
            ))
            
            # Asks
            asks = np.array(order_book['asks'])
            fig.add_trace(go.Bar(
                x=-asks[:, 1],  # Negative for left side
                y=asks[:, 0],
                orientation='h',
                name='Asks',
                marker_color='red',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Order Book Depth",
                xaxis_title="Volume",
                yaxis_title="Price",
                height=400,
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trade flow
            st.subheader("Trade Flow")
            trade_flow = microstructure_data['trade_flow']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trade_flow['timestamps'],
                y=trade_flow['buy_volume'],
                mode='lines+markers',
                name='Buy Volume',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=trade_flow['timestamps'],
                y=trade_flow['sell_volume'],
                mode='lines+markers',
                name='Sell Volume',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Trade Volume Flow",
                xaxis_title="Time",
                yaxis_title="Volume",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Spread analysis
        st.subheader("Bid-Ask Spread")
        spread_data = microstructure_data['spread']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spread_data['timestamps'],
            y=spread_data['spreads'],
            mode='lines',
            name='Spread',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Bid-Ask Spread Over Time",
            xaxis_title="Time",
            yaxis_title="Spread ($)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_sentiment_trends(self):
        """Render Reddit sentiment trends panel"""
        st.subheader("💬 Reddit Sentiment Trends")
        
        sentiment_data = self.data_manager.get_sentiment_data()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sentiment over time
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Sentiment Score', 'Mention Volume'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data['timestamps'],
                    y=sentiment_data['sentiment_scores'],
                    mode='lines',
                    name='Sentiment',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=sentiment_data['timestamps'],
                    y=sentiment_data['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Sentiment Trends (7 Days)",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top mentions
            st.subheader("Top Mentions")
            mentions_df = pd.DataFrame(sentiment_data['top_mentions'])
            
            for _, mention in mentions_df.iterrows():
                sentiment_color = 'green' if mention['sentiment'] > 0 else 'red' if mention['sentiment'] < 0 else 'gray'
                st.markdown(f"""
                **{mention['symbol']}**  
                Mentions: {mention['mentions']:,}  
                <span style='color: {sentiment_color};'>Sentiment: {mention['sentiment']:+.2f}</span>
                """, unsafe_allow_html=True)
                st.markdown("---")
    
    def render_regime_monitoring(self):
        """Render regime change monitoring panel"""
        st.subheader("🔄 Market Regime Monitoring")
        
        regime_data = self.data_manager.get_regime_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Current regime
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Regime</h3>
                <h2 style="color: #1f77b4;">{regime_data['current_regime']}</h2>
                <p>Confidence: {regime_data['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Transition probabilities heatmap
            st.subheader("Transition Probabilities")
            regime_names = regime_data['regime_history']['regime_names']
            transition_matrix = np.array(regime_data['transition_probabilities'])
            
            fig = go.Figure(data=go.Heatmap(
                z=transition_matrix,
                x=regime_names,
                y=regime_names,
                colorscale='Blues',
                text=transition_matrix,
                texttemplate="%{text:.2f}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Regime Transition Matrix",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Regime history
            st.subheader("Regime History (30 Days)")
            regime_history = regime_data['regime_history']
            
            # Create regime timeline
            regime_colors = ['green', 'red', 'orange', 'blue']
            regime_names = regime_history['regime_names']
            
            fig = go.Figure()
            
            for i, regime_name in enumerate(regime_names):
                regime_mask = np.array(regime_history['regimes']) == i
                if np.any(regime_mask):
                    fig.add_trace(go.Scatter(
                        x=[pd.to_datetime(t) for t, mask in zip(regime_history['timestamps'], regime_mask) if mask],
                        y=[i] * np.sum(regime_mask),
                        mode='markers',
                        name=regime_name,
                        marker=dict(color=regime_colors[i], size=8)
                    ))
            
            fig.update_layout(
                title="Regime Timeline",
                xaxis_title="Date",
                yaxis_title="Regime",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(regime_names))),
                    ticktext=regime_names
                ),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self):
        """Render strategy performance metrics panel"""
        st.subheader("📊 Strategy Performance")
        
        performance_data = self.data_manager.get_performance_metrics()
        
        # Key performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", f"{performance_data['sharpe_ratio']:.2f}")
        
        with col2:
            st.metric("Sortino Ratio", f"{performance_data['sortino_ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{performance_data['max_drawdown']:.1%}")
        
        with col4:
            st.metric("Win Rate", f"{performance_data['win_rate']:.1%}")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Cumulative returns
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=performance_data['timestamps'],
                y=performance_data['cumulative_returns'],
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title="Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Return",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Daily returns distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=performance_data['returns'],
                nbinsx=30,
                name='Daily Returns',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Return",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Alpha", f"{performance_data['alpha']:.3f}")
        
        with col2:
            st.metric("Beta", f"{performance_data['beta']:.2f}")
        
        with col3:
            st.metric("Volatility", f"{performance_data['volatility']:.1%}")
    
    def render_trade_execution(self):
        """Render trade execution interface panel"""
        st.subheader("💼 Trade Execution")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Manual Trade Entry")
            
            # Trade form
            with st.form("trade_form"):
                symbol = st.text_input("Symbol", value="AAPL")
                side = st.selectbox("Side", ["BUY", "SELL"])
                quantity = st.number_input("Quantity", min_value=1, value=100)
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "STOP"])
                
                if order_type in ["LIMIT", "STOP"]:
                    price = st.number_input("Price", min_value=0.01, value=100.0)
                else:
                    price = None
                
                submitted = st.form_submit_button("Submit Order")
                
                if submitted:
                    # Here you would integrate with your trading system
                    st.success(f"Order submitted: {side} {quantity} {symbol}")
        
        with col2:
            st.subheader("Recent Orders")
            
            # Mock recent orders
            recent_orders = [
                {"timestamp": "2024-01-15 10:30:00", "symbol": "AAPL", "side": "BUY", "quantity": 100, "price": 150.25, "status": "FILLED"},
                {"timestamp": "2024-01-15 10:25:00", "symbol": "GOOGL", "side": "SELL", "quantity": 50, "price": 2500.00, "status": "FILLED"},
                {"timestamp": "2024-01-15 10:20:00", "symbol": "MSFT", "side": "BUY", "quantity": 75, "price": 300.50, "status": "PENDING"},
            ]
            
            orders_df = pd.DataFrame(recent_orders)
            
            # Color code status
            def color_status(val):
                if val == 'FILLED':
                    return 'color: green'
                elif val == 'PENDING':
                    return 'color: orange'
                elif val == 'CANCELLED':
                    return 'color: red'
                return 'color: gray'
            
            styled_orders = orders_df.style.applymap(color_status, subset=['status'])
            st.dataframe(styled_orders, use_container_width=True)
        
        # Strategy controls
        st.subheader("Strategy Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Start All Strategies", type="primary"):
                st.success("All strategies started")
        
        with col2:
            if st.button("Stop All Strategies", type="secondary"):
                st.warning("All strategies stopped")
        
        with col3:
            if st.button("Emergency Stop", type="secondary"):
                st.error("Emergency stop activated")
        
        with col4:
            if st.button("Reset Positions"):
                st.info("Positions reset")
    
    def render_risk_monitoring(self):
        """Render risk monitoring panels"""
        st.subheader("⚠️ Risk Monitoring")
        
        risk_data = self.data_manager.get_risk_metrics()
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", f"${risk_data['var_95']:,.0f}")
        
        with col2:
            st.metric("VaR (99%)", f"${risk_data['var_99']:,.0f}")
        
        with col3:
            st.metric("Expected Shortfall", f"${risk_data['expected_shortfall']:,.0f}")
        
        with col4:
            st.metric("Portfolio Beta", f"{risk_data['portfolio_beta']:.2f}")
        
        # Risk breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector exposure
            st.subheader("Sector Exposure")
            sector_data = risk_data['sector_exposure']
            
            fig = go.Figure(data=[go.Pie(
                labels=list(sector_data.keys()),
                values=list(sector_data.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                title="Portfolio Sector Allocation",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stress test results
            st.subheader("Stress Test Results")
            stress_data = risk_data['stress_test_results']
            
            fig = go.Figure(data=[go.Bar(
                x=list(stress_data.keys()),
                y=list(stress_data.values()),
                marker_color=['red' if x < -0.2 else 'orange' if x < -0.1 else 'green' 
                             for x in stress_data.values()]
            )])
            
            fig.update_layout(
                title="Stress Test Scenarios",
                xaxis_title="Scenario",
                yaxis_title="Portfolio Impact",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk alerts
        st.subheader("Risk Alerts")
        
        # Mock alerts
        alerts = [
            {"level": "HIGH", "message": "Portfolio correlation above threshold (0.85)", "timestamp": "2024-01-15 10:30:00"},
            {"level": "MEDIUM", "message": "VaR limit approaching (85% of limit)", "timestamp": "2024-01-15 10:25:00"},
            {"level": "LOW", "message": "Sector concentration in Technology", "timestamp": "2024-01-15 10:20:00"},
        ]
        
        for alert in alerts:
            alert_color = {
                'HIGH': 'red',
                'MEDIUM': 'orange',
                'LOW': 'blue'
            }.get(alert['level'], 'gray')
            
            st.markdown(f"""
            <div style="padding: 0.5rem; border-left: 4px solid {alert_color}; margin: 0.5rem 0;">
                <strong style="color: {alert_color};">{alert['level']}</strong> - {alert['message']}
                <br><small>{alert['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Initialize configuration and data manager
    config = DashboardConfig()
    data_manager = DataManager(config)
    components = DashboardComponents(data_manager)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Alpha Discovery Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Select Dashboard",
        [
            "Portfolio Overview",
            "Agent Consensus",
            "Market Microstructure", 
            "Sentiment Analysis",
            "Regime Monitoring",
            "Performance Metrics",
            "Trade Execution",
            "Risk Monitoring",
            "All Dashboards"
        ]
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    if auto_refresh:
        refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 60, 5)
        st.sidebar.text(f"Next refresh in {refresh_rate}s")
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-header">Settings</div>', unsafe_allow_html=True)
    
    show_debug = st.sidebar.checkbox("Show Debug Info")
    if show_debug:
        st.sidebar.json({
            "Redis Connected": data_manager.redis_client is not None,
            "Postgres Connected": data_manager.postgres_conn is not None,
            "Current Time": datetime.now().isoformat()
        })
    
    # Main content area
    if page == "Portfolio Overview":
        components.render_pnl_positions()
    
    elif page == "Agent Consensus":
        components.render_agent_consensus()
    
    elif page == "Market Microstructure":
        components.render_market_microstructure()
    
    elif page == "Sentiment Analysis":
        components.render_sentiment_trends()
    
    elif page == "Regime Monitoring":
        components.render_regime_monitoring()
    
    elif page == "Performance Metrics":
        components.render_performance_metrics()
    
    elif page == "Trade Execution":
        components.render_trade_execution()
    
    elif page == "Risk Monitoring":
        components.render_risk_monitoring()
    
    elif page == "All Dashboards":
        # Render all dashboards in tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Portfolio", "Agents", "Microstructure", "Sentiment", 
            "Regime", "Performance", "Trading", "Risk"
        ])
        
        with tab1:
            components.render_pnl_positions()
        
        with tab2:
            components.render_agent_consensus()
        
        with tab3:
            components.render_market_microstructure()
        
        with tab4:
            components.render_sentiment_trends()
        
        with tab5:
            components.render_regime_monitoring()
        
        with tab6:
            components.render_performance_metrics()
        
        with tab7:
            components.render_trade_execution()
        
        with tab8:
            components.render_risk_monitoring()
    
    # Auto-refresh mechanism
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main() 