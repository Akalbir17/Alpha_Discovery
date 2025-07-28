"""
Trading Engine
Handles order execution and trade management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order data structure"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None

class TradingEngine:
    """Trading engine for order execution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orders: Dict[str, Order] = {}
        self.is_running = False
        self.order_counter = 0
        
    async def start(self):
        """Start trading engine"""
        try:
            logger.info("Starting trading engine...")
            self.is_running = True
            logger.info("Trading engine started successfully")
        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            raise
            
    async def stop(self):
        """Stop trading engine"""
        try:
            logger.info("Stopping trading engine...")
            self.is_running = False
            logger.info("Trading engine stopped")
        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")
            
    async def submit_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                          quantity: float, price: Optional[float] = None, 
                          stop_price: Optional[float] = None) -> Order:
        """Submit a new order"""
        try:
            self.order_counter += 1
            order_id = f"order_{self.order_counter}"
            
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                created_at=datetime.now()
            )
            
            self.orders[order_id] = order
            logger.info(f"Order submitted: {order_id}")
            
            # Simulate order processing
            await self._process_order(order)
            
            return order
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if order_id in self.orders:
                order = self.orders[order_id]
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.now()
                    logger.info(f"Order cancelled: {order_id}")
                    return True
                else:
                    logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
                    return False
            else:
                logger.warning(f"Order not found: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
            
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.orders.get(order_id)
        
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return [order for order in self.orders.values() 
                if order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]]
        
    async def get_order_history(self) -> List[Order]:
        """Get order history"""
        return list(self.orders.values())
        
    async def _process_order(self, order: Order):
        """Process order (simulation)"""
        try:
            # Simulate order processing delay
            await asyncio.sleep(0.1)
            
            # Simulate order fill
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = order.price or 100.0  # Simulate fill price
            order.updated_at = datetime.now()
            
            logger.info(f"Order filled: {order.id}")
            
        except Exception as e:
            logger.error(f"Error processing order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now() 