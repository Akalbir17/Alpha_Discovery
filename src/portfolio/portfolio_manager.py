"""
Portfolio Manager
Handles portfolio tracking and performance analysis
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    cost_basis: float
    last_updated: datetime

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    total_pnl: float
    total_pnl_percent: float
    daily_pnl: float
    daily_pnl_percent: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    calculated_at: datetime

class PortfolioManager:
    """Portfolio manager for tracking positions and performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.historical_values: List[float] = []
        self.is_running = False
        self.initial_capital = config.get('initial_capital', 100000)
        
    async def start(self):
        """Start portfolio manager"""
        try:
            logger.info("Starting portfolio manager...")
            self.is_running = True
            logger.info("Portfolio manager started successfully")
        except Exception as e:
            logger.error(f"Failed to start portfolio manager: {e}")
            raise
            
    async def stop(self):
        """Stop portfolio manager"""
        try:
            logger.info("Stopping portfolio manager...")
            self.is_running = False
            logger.info("Portfolio manager stopped")
        except Exception as e:
            logger.error(f"Error stopping portfolio manager: {e}")
            
    async def update_position(self, symbol: str, quantity: float, price: float):
        """Update position in portfolio"""
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Update existing position
                new_quantity = position.quantity + quantity
                if new_quantity == 0:
                    # Position closed
                    del self.positions[symbol]
                    logger.info(f"Position closed: {symbol}")
                else:
                    # Update position
                    if quantity > 0:  # Adding to position
                        total_cost = (position.quantity * position.avg_price) + (quantity * price)
                        position.avg_price = total_cost / new_quantity
                    
                    position.quantity = new_quantity
                    position.cost_basis = position.quantity * position.avg_price
                    position.last_updated = datetime.now()
                    
                    logger.info(f"Position updated: {symbol} - {new_quantity} shares")
            else:
                # New position
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_percent=0.0,
                    cost_basis=quantity * price,
                    last_updated=datetime.now()
                )
                
                self.positions[symbol] = position
                logger.info(f"New position created: {symbol} - {quantity} shares")
                
        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
            raise
            
    async def update_market_prices(self, price_data: Dict[str, float]):
        """Update current market prices for all positions"""
        try:
            for symbol, current_price in price_data.items():
                if symbol in self.positions:
                    position = self.positions[symbol]
                    position.current_price = current_price
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = position.market_value - position.cost_basis
                    position.unrealized_pnl_percent = (position.unrealized_pnl / position.cost_basis) * 100
                    position.last_updated = datetime.now()
                    
        except Exception as e:
            logger.error(f"Error updating market prices: {e}")
            
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        try:
            total_value = sum(pos.market_value for pos in self.positions.values())
            total_cost = sum(pos.cost_basis for pos in self.positions.values())
            total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            cash = self.initial_capital - total_cost
            portfolio_value = total_value + cash
            
            return {
                'total_value': portfolio_value,
                'cash': cash,
                'invested_value': total_value,
                'total_cost': total_cost,
                'total_pnl': total_pnl,
                'total_pnl_percent': (total_pnl / total_cost) * 100 if total_cost > 0 else 0,
                'position_count': len(self.positions),
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'avg_price': pos.avg_price,
                        'current_price': pos.current_price,
                        'market_value': pos.market_value,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'unrealized_pnl_percent': pos.unrealized_pnl_percent,
                        'weight': (pos.market_value / portfolio_value) * 100 if portfolio_value > 0 else 0
                    }
                    for pos in self.positions.values()
                ],
                'updated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            raise
            
    async def calculate_performance_metrics(self) -> PortfolioMetrics:
        """Calculate portfolio performance metrics"""
        try:
            portfolio_summary = await self.get_portfolio_summary()
            total_value = portfolio_summary['total_value']
            total_pnl = portfolio_summary['total_pnl']
            total_pnl_percent = portfolio_summary['total_pnl_percent']
            
            # Add current value to history
            self.historical_values.append(total_value)
            
            # Calculate daily PnL if we have historical data
            daily_pnl = 0.0
            daily_pnl_percent = 0.0
            if len(self.historical_values) > 1:
                daily_pnl = self.historical_values[-1] - self.historical_values[-2]
                daily_pnl_percent = (daily_pnl / self.historical_values[-2]) * 100
            
            # Calculate additional metrics
            sharpe_ratio = self._calculate_sharpe_ratio()
            max_drawdown = self._calculate_max_drawdown()
            volatility = self._calculate_volatility()
            beta = 1.0  # Simplified - would need market data for proper calculation
            
            return PortfolioMetrics(
                total_value=total_value,
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                daily_pnl=daily_pnl,
                daily_pnl_percent=daily_pnl_percent,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                calculated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
            
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.historical_values) < 2:
            return 0.0
            
        returns = np.diff(self.historical_values) / self.historical_values[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
            
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.historical_values) < 2:
            return 0.0
            
        values = np.array(self.historical_values)
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        return abs(np.min(drawdown))
        
    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.historical_values) < 2:
            return 0.0
            
        returns = np.diff(self.historical_values) / self.historical_values[:-1]
        return np.std(returns) * np.sqrt(252)  # Annualized
        
    def get_cash_balance(self) -> float:
        """Get cash balance"""
        total_cost = sum(pos.cost_basis for pos in self.positions.values())
        return self.initial_capital - total_cost
        
    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate position size based on risk and confidence"""
        try:
            # Get available cash
            cash = self.get_cash_balance()
            if cash <= 0:
                return 0.0
            
            # Get max position size from config
            max_pos_size_abs = self.config.get('max_position_size', 100000)
            
            # Determine position size based on confidence
            # This is a simple linear scaling. A more sophisticated model could be used.
            pos_size = cash * confidence
            
            # Ensure position size does not exceed max limit
            pos_size = min(pos_size, max_pos_size_abs)
            
            # Ensure we have enough cash for the position
            pos_size = min(pos_size, cash)
            
            return pos_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
            
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self.positions.get(symbol)
        
    async def get_all_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self.positions.values())
        
    async def get_position_weights(self) -> Dict[str, float]:
        """Get position weights in portfolio"""
        portfolio_summary = await self.get_portfolio_summary()
        total_value = portfolio_summary['total_value']
        
        if total_value == 0:
            return {}
            
        return {
            pos.symbol: (pos.market_value / total_value) * 100
            for pos in self.positions.values()
        } 