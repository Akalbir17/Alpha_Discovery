"""
Risk Manager
Handles position and portfolio risk management
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    symbol: str
    var_95: float
    var_99: float
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    risk_level: RiskLevel
    calculated_at: datetime

@dataclass
class PositionRisk:
    """Position risk data structure"""
    symbol: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    position_risk: float
    risk_contribution: float
    risk_level: RiskLevel

class RiskManager:
    """Risk manager for position and portfolio risk management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_metrics: Dict[str, RiskMetrics] = {}
        self.position_risks: Dict[str, PositionRisk] = {}
        self.is_running = False
        
        # Risk limits from config
        self.max_position_size = config.get('max_position_size', 100000)
        self.max_portfolio_var = config.get('max_portfolio_var', 0.05)
        self.max_sector_concentration = config.get('max_sector_concentration', 0.3)
        
    async def start(self):
        """Start risk manager"""
        try:
            logger.info("Starting risk manager...")
            self.is_running = True
            logger.info("Risk manager started successfully")
        except Exception as e:
            logger.error(f"Failed to start risk manager: {e}")
            raise
            
    async def stop(self):
        """Stop risk manager"""
        try:
            logger.info("Stopping risk manager...")
            self.is_running = False
            logger.info("Risk manager stopped")
        except Exception as e:
            logger.error(f"Error stopping risk manager: {e}")
            
    async def calculate_position_risk(self, symbol: str, quantity: float, 
                                    price: float) -> PositionRisk:
        """Calculate risk for a position"""
        try:
            market_value = quantity * price
            
            # Get risk metrics for symbol
            risk_metrics = await self.get_risk_metrics(symbol)
            
            # Calculate position risk
            position_risk = market_value * risk_metrics.var_95
            risk_contribution = position_risk / self.max_position_size
            
            # Determine risk level
            if risk_contribution > 0.8:
                risk_level = RiskLevel.CRITICAL
            elif risk_contribution > 0.6:
                risk_level = RiskLevel.HIGH
            elif risk_contribution > 0.3:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            position_risk_obj = PositionRisk(
                symbol=symbol,
                quantity=quantity,
                market_value=market_value,
                unrealized_pnl=0.0,  # Would be calculated from current vs entry price
                position_risk=position_risk,
                risk_contribution=risk_contribution,
                risk_level=risk_level
            )
            
            self.position_risks[symbol] = position_risk_obj
            return position_risk_obj
            
        except Exception as e:
            logger.error(f"Error calculating position risk for {symbol}: {e}")
            raise
            
    async def get_risk_metrics(self, symbol: str) -> RiskMetrics:
        """Get risk metrics for a symbol"""
        if symbol in self.risk_metrics:
            return self.risk_metrics[symbol]
            
        # Calculate risk metrics (simplified)
        risk_metrics = RiskMetrics(
            symbol=symbol,
            var_95=0.05,  # 5% VaR
            var_99=0.08,  # 8% VaR
            volatility=0.25,  # 25% volatility
            beta=1.0,  # Market beta
            sharpe_ratio=0.8,  # Sharpe ratio
            max_drawdown=0.15,  # 15% max drawdown
            risk_level=RiskLevel.MEDIUM,
            calculated_at=datetime.now()
        )
        
        self.risk_metrics[symbol] = risk_metrics
        return risk_metrics
        
    async def check_position_limits(self, symbol: str, quantity: float, 
                                  price: float) -> bool:
        """Check if position is within limits"""
        try:
            position_value = abs(quantity * price)
            
            # Check position size limit
            if position_value > self.max_position_size:
                logger.warning(f"Position size limit exceeded for {symbol}: {position_value}")
                return False
                
            # Check risk limits
            position_risk = await self.calculate_position_risk(symbol, quantity, price)
            if position_risk.risk_level == RiskLevel.CRITICAL:
                logger.warning(f"Critical risk level for {symbol}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return False
            
    async def calculate_portfolio_risk(self, positions: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate portfolio-level risk metrics"""
        try:
            total_value = 0.0
            total_var = 0.0
            position_risks = []
            
            for symbol, position in positions.items():
                quantity = position.get('quantity', 0)
                price = position.get('price', 0)
                
                if quantity != 0:
                    position_risk = await self.calculate_position_risk(symbol, quantity, price)
                    position_risks.append(position_risk)
                    total_value += position_risk.market_value
                    total_var += position_risk.position_risk
            
            # Calculate portfolio VaR
            portfolio_var = total_var / total_value if total_value > 0 else 0
            
            # Determine portfolio risk level
            if portfolio_var > self.max_portfolio_var:
                portfolio_risk_level = RiskLevel.HIGH
            elif portfolio_var > self.max_portfolio_var * 0.7:
                portfolio_risk_level = RiskLevel.MEDIUM
            else:
                portfolio_risk_level = RiskLevel.LOW
            
            return {
                'total_value': total_value,
                'portfolio_var': portfolio_var,
                'risk_level': portfolio_risk_level,
                'position_risks': position_risks,
                'calculated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            raise
            
    async def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """Get current risk alerts"""
        alerts = []
        
        for symbol, position_risk in self.position_risks.items():
            if position_risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                alerts.append({
                    'type': 'position_risk',
                    'symbol': symbol,
                    'risk_level': position_risk.risk_level.value,
                    'message': f"High risk position in {symbol}",
                    'timestamp': datetime.now()
                })
        
        return alerts 