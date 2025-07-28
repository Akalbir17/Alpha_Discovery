"""
Order Flow Tool

Analyzes order flow and market microstructure to identify liquidity patterns,
order imbalances, and market impact.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OrderFlowTool:
    """
    Tool for analyzing order flow and market microstructure.
    
    Features:
    - Analyze order book imbalances
    - Track order flow patterns
    - Calculate market impact
    - Identify large orders and institutional activity
    - Monitor liquidity patterns
    """
    
    def __init__(self):
        self.order_flow_data = {}
        
    async def get_order_flow(self, symbol: str, timeframe: str = "1m") -> Dict[str, Any]:
        """
        Get order flow analysis for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary containing order flow analysis
        """
        try:
            logger.info(f"Analyzing order flow for {symbol}")
            
            # In a real implementation, this would connect to market data feeds
            # For now, we'll use mock data
            order_flow = self._get_mock_order_flow(symbol, timeframe)
            
            # Calculate order flow metrics
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'buy_volume': order_flow['buy_volume'],
                'sell_volume': order_flow['sell_volume'],
                'order_imbalance': self._calculate_order_imbalance(order_flow),
                'large_orders': self._identify_large_orders(order_flow),
                'liquidity_score': self._calculate_liquidity_score(order_flow),
                'market_impact': self._estimate_market_impact(order_flow),
                'flow_pattern': self._analyze_flow_pattern(order_flow)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing order flow for {symbol}: {e}")
            return self._get_mock_order_flow_analysis(symbol)
    
    def _calculate_order_imbalance(self, order_flow: Dict[str, Any]) -> float:
        """Calculate order flow imbalance"""
        try:
            buy_volume = order_flow.get('buy_volume', 0)
            sell_volume = order_flow.get('sell_volume', 0)
            
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                imbalance = (buy_volume - sell_volume) / total_volume
                return round(imbalance, 4)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating order imbalance: {e}")
            return 0.0
    
    def _identify_large_orders(self, order_flow: Dict[str, Any]) -> List[Dict]:
        """Identify large orders in the order flow"""
        try:
            large_orders = []
            orders = order_flow.get('orders', [])
            
            # Calculate average order size
            if orders:
                avg_size = sum(order['size'] for order in orders) / len(orders)
                threshold = avg_size * 3  # Orders 3x larger than average
                
                for order in orders:
                    if order['size'] > threshold:
                        large_orders.append({
                            'side': order['side'],
                            'size': order['size'],
                            'price': order['price'],
                            'timestamp': order['timestamp'],
                            'size_ratio': order['size'] / avg_size
                        })
            
            return large_orders
            
        except Exception as e:
            logger.error(f"Error identifying large orders: {e}")
            return []
    
    def _calculate_liquidity_score(self, order_flow: Dict[str, Any]) -> float:
        """Calculate liquidity score based on order flow"""
        try:
            # Factors affecting liquidity:
            # 1. Total volume
            # 2. Order size distribution
            # 3. Bid-ask spread
            # 4. Market depth
            
            total_volume = order_flow.get('buy_volume', 0) + order_flow.get('sell_volume', 0)
            spread = order_flow.get('spread', 0.01)
            depth = order_flow.get('market_depth', 1000000)
            
            # Normalize factors
            volume_score = min(total_volume / 1000000, 1.0)  # Cap at 1M volume
            spread_score = max(0, 1 - (spread * 100))  # Lower spread = higher score
            depth_score = min(depth / 1000000, 1.0)  # Cap at 1M depth
            
            # Weighted average
            liquidity_score = (volume_score * 0.4 + spread_score * 0.3 + depth_score * 0.3)
            return round(liquidity_score, 4)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    def _estimate_market_impact(self, order_flow: Dict[str, Any]) -> float:
        """Estimate market impact of large orders"""
        try:
            large_orders = self._identify_large_orders(order_flow)
            total_volume = order_flow.get('buy_volume', 0) + order_flow.get('sell_volume', 0)
            
            if not large_orders or total_volume == 0:
                return 0.0
            
            # Calculate impact based on large order volume relative to total volume
            large_order_volume = sum(order['size'] for order in large_orders)
            impact_ratio = large_order_volume / total_volume
            
            # Estimate price impact (simplified model)
            # In practice, this would use more sophisticated models
            estimated_impact = impact_ratio * 0.1  # 10% of volume ratio
            
            return round(min(estimated_impact, 0.05), 4)  # Cap at 5%
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return 0.0
    
    def _analyze_flow_pattern(self, order_flow: Dict[str, Any]) -> str:
        """Analyze the pattern of order flow"""
        try:
            buy_volume = order_flow.get('buy_volume', 0)
            sell_volume = order_flow.get('sell_volume', 0)
            
            if buy_volume > sell_volume * 1.5:
                return "heavy_buying"
            elif sell_volume > buy_volume * 1.5:
                return "heavy_selling"
            elif abs(buy_volume - sell_volume) / (buy_volume + sell_volume) < 0.1:
                return "balanced"
            else:
                return "mixed"
                
        except Exception as e:
            logger.error(f"Error analyzing flow pattern: {e}")
            return "unknown"
    
    def _get_mock_order_flow(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Generate mock order flow data"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'buy_volume': 500000,
            'sell_volume': 450000,
            'spread': 0.002,
            'market_depth': 2000000,
            'orders': [
                {
                    'side': 'buy',
                    'size': 10000,
                    'price': 100.0,
                    'timestamp': datetime.now()
                },
                {
                    'side': 'sell',
                    'size': 15000,
                    'price': 100.1,
                    'timestamp': datetime.now()
                },
                {
                    'side': 'buy',
                    'size': 5000,
                    'price': 99.9,
                    'timestamp': datetime.now()
                }
            ]
        }
    
    def _get_mock_order_flow_analysis(self, symbol: str) -> Dict[str, Any]:
        """Return mock order flow analysis"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'buy_volume': 500000,
            'sell_volume': 450000,
            'order_imbalance': 0.053,
            'large_orders': [
                {
                    'side': 'buy',
                    'size': 15000,
                    'price': 100.0,
                    'timestamp': datetime.now(),
                    'size_ratio': 3.5
                }
            ],
            'liquidity_score': 0.75,
            'market_impact': 0.015,
            'flow_pattern': 'heavy_buying'
        }
    
    async def track_order_flow_changes(self, symbol: str, lookback_minutes: int = 30) -> Dict[str, Any]:
        """
        Track changes in order flow over time.
        
        Args:
            symbol: Trading symbol
            lookback_minutes: Time window for analysis
            
        Returns:
            Dictionary containing order flow change analysis
        """
        try:
            logger.info(f"Tracking order flow changes for {symbol}")
            
            # In a real implementation, this would analyze historical order flow data
            # For now, we'll use mock data
            
            changes = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'volume_change': 0.15,  # 15% increase
                'imbalance_change': 0.02,  # 2% increase in buy imbalance
                'liquidity_change': -0.05,  # 5% decrease in liquidity
                'large_order_frequency': 0.25,  # 25% increase in large orders
                'trend': 'increasing_buy_pressure'
            }
            
            return changes
            
        except Exception as e:
            logger.error(f"Error tracking order flow changes for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'volume_change': 0.0,
                'imbalance_change': 0.0,
                'liquidity_change': 0.0,
                'large_order_frequency': 0.0,
                'trend': 'stable'
            }
    
    async def detect_unusual_activity(self, symbol: str, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect unusual order flow activity.
        
        Args:
            symbol: Trading symbol
            threshold: Threshold for unusual activity detection
            
        Returns:
            Dictionary containing unusual activity detection results
        """
        try:
            logger.info(f"Detecting unusual activity for {symbol}")
            
            # Get current order flow
            order_flow = await self.get_order_flow(symbol)
            
            # Calculate baseline metrics (would come from historical data)
            baseline_volume = 400000  # Mock baseline
            baseline_imbalance = 0.0
            
            current_volume = order_flow['buy_volume'] + order_flow['sell_volume']
            current_imbalance = order_flow['order_imbalance']
            
            # Check for unusual activity
            volume_ratio = current_volume / baseline_volume
            imbalance_deviation = abs(current_imbalance - baseline_imbalance)
            
            unusual_activity = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'unusual_volume': volume_ratio > threshold,
                'unusual_imbalance': imbalance_deviation > 0.1,
                'volume_ratio': volume_ratio,
                'imbalance_deviation': imbalance_deviation,
                'large_orders_detected': len(order_flow['large_orders']) > 3,
                'alert_level': 'medium'
            }
            
            # Determine alert level
            if unusual_activity['unusual_volume'] and unusual_activity['unusual_imbalance']:
                unusual_activity['alert_level'] = 'high'
            elif unusual_activity['unusual_volume'] or unusual_activity['unusual_imbalance']:
                unusual_activity['alert_level'] = 'medium'
            else:
                unusual_activity['alert_level'] = 'low'
            
            return unusual_activity
            
        except Exception as e:
            logger.error(f"Error detecting unusual activity for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'unusual_volume': False,
                'unusual_imbalance': False,
                'volume_ratio': 1.0,
                'imbalance_deviation': 0.0,
                'large_orders_detected': False,
                'alert_level': 'low'
            } 