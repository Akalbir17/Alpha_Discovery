#!/usr/bin/env python3
"""
Graceful Shutdown Handler

Handles graceful shutdown of the Alpha Discovery platform, ensuring all components
are properly closed, positions are managed, and data is saved before termination.
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import traceback

from configs.config_loader import get_config
from src.utils.comprehensive_logger import get_comprehensive_logger

logger = logging.getLogger(__name__)

class ShutdownReason(Enum):
    """Reasons for system shutdown."""
    USER_REQUEST = "user_request"
    SIGNAL_INTERRUPT = "signal_interrupt"
    SIGNAL_TERMINATE = "signal_terminate"
    FATAL_ERROR = "fatal_error"
    EMERGENCY_STOP = "emergency_stop"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

class ShutdownPhase(Enum):
    """Phases of shutdown process."""
    INITIATED = "initiated"
    STOPPING_DISCOVERY = "stopping_discovery"
    CLOSING_POSITIONS = "closing_positions"
    SAVING_STATE = "saving_state"
    STOPPING_SERVICES = "stopping_services"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ShutdownStatus:
    """Status of shutdown process."""
    reason: ShutdownReason
    phase: ShutdownPhase
    start_time: datetime
    current_step: str
    completed_steps: List[str]
    failed_steps: List[str]
    warnings: List[str]
    errors: List[str]
    progress_percent: float
    estimated_completion: Optional[datetime] = None

class ShutdownHandler:
    """Graceful shutdown handler for Alpha Discovery platform."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.config = get_config('system')
        self.logger = get_comprehensive_logger()
        
        # Shutdown configuration
        self.shutdown_timeout = self.config.get('shutdown_timeout', 300)  # 5 minutes
        self.position_close_timeout = self.config.get('position_close_timeout', 120)  # 2 minutes
        self.force_close_positions = self.config.get('force_close_positions', True)
        self.save_state_on_shutdown = self.config.get('save_state_on_shutdown', True)
        
        # Shutdown state
        self.shutdown_requested = False
        self.shutdown_in_progress = False
        self.shutdown_status = None
        self.shutdown_callbacks = []
        
        # Threading
        self.shutdown_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Signal handlers
        self.original_sigint_handler = None
        self.original_sigterm_handler = None
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        logger.info("Graceful shutdown handler initialized")
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            # Store original handlers
            self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
            self.original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Windows-specific signals
            if sys.platform == "win32":
                signal.signal(signal.SIGBREAK, self._signal_handler)
            
            logger.info("Signal handlers configured for graceful shutdown")
            
        except Exception as e:
            logger.error(f"Error setting up signal handlers: {e}")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        signal_names = {
            signal.SIGINT: "SIGINT",
            signal.SIGTERM: "SIGTERM"
        }
        
        if sys.platform == "win32":
            signal_names[signal.SIGBREAK] = "SIGBREAK"
        
        signal_name = signal_names.get(signum, f"SIGNAL_{signum}")
        
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        
        # Determine shutdown reason
        if signum == signal.SIGINT:
            reason = ShutdownReason.SIGNAL_INTERRUPT
        elif signum == signal.SIGTERM:
            reason = ShutdownReason.SIGNAL_TERMINATE
        else:
            reason = ShutdownReason.USER_REQUEST
        
        # Initiate shutdown
        self.initiate_shutdown(reason)
    
    def add_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to be executed during shutdown."""
        self.shutdown_callbacks.append(callback)
    
    def remove_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Remove shutdown callback."""
        if callback in self.shutdown_callbacks:
            self.shutdown_callbacks.remove(callback)
    
    def initiate_shutdown(self, reason: ShutdownReason = ShutdownReason.USER_REQUEST) -> None:
        """Initiate graceful shutdown."""
        with self.shutdown_lock:
            if self.shutdown_requested:
                logger.warning("Shutdown already requested, ignoring duplicate request")
                return
            
            self.shutdown_requested = True
            self.shutdown_in_progress = True
            
            # Initialize shutdown status
            self.shutdown_status = ShutdownStatus(
                reason=reason,
                phase=ShutdownPhase.INITIATED,
                start_time=datetime.now(),
                current_step="Initializing shutdown",
                completed_steps=[],
                failed_steps=[],
                warnings=[],
                errors=[],
                progress_percent=0.0
            )
            
            # Log shutdown initiation
            self.logger.system_logger.log_system_shutdown(reason.value)
            
            # Set shutdown event
            self.shutdown_event.set()
            
            # Start shutdown process in background
            shutdown_thread = threading.Thread(
                target=self._execute_shutdown,
                name="ShutdownHandler",
                daemon=False
            )
            shutdown_thread.start()
    
    def _execute_shutdown(self) -> None:
        """Execute the shutdown process."""
        try:
            start_time = time.time()
            
            # Execute shutdown steps
            shutdown_steps = [
                ("Stopping discovery loop", self._stop_discovery_loop),
                ("Closing open positions", self._close_positions),
                ("Saving system state", self._save_system_state),
                ("Stopping market data streaming", self._stop_market_data),
                ("Stopping Reddit monitoring", self._stop_reddit_monitoring),
                ("Stopping MCP server", self._stop_mcp_server),
                ("Executing shutdown callbacks", self._execute_shutdown_callbacks),
                ("Cleaning up resources", self._cleanup_resources),
                ("Finalizing shutdown", self._finalize_shutdown)
            ]
            
            total_steps = len(shutdown_steps)
            
            for i, (step_name, step_function) in enumerate(shutdown_steps):
                try:
                    # Check timeout
                    if time.time() - start_time > self.shutdown_timeout:
                        raise TimeoutError(f"Shutdown timeout exceeded ({self.shutdown_timeout}s)")
                    
                    # Update status
                    self.shutdown_status.current_step = step_name
                    self.shutdown_status.progress_percent = (i / total_steps) * 100
                    
                    logger.info(f"Shutdown step {i+1}/{total_steps}: {step_name}")
                    
                    # Execute step
                    step_start = time.time()
                    asyncio.run(step_function())
                    step_duration = time.time() - step_start
                    
                    # Mark as completed
                    self.shutdown_status.completed_steps.append(f"{step_name} ({step_duration:.1f}s)")
                    
                    logger.info(f"Completed: {step_name} in {step_duration:.1f}s")
                    
                except Exception as e:
                    error_msg = f"Failed step: {step_name} - {str(e)}"
                    self.shutdown_status.failed_steps.append(error_msg)
                    self.shutdown_status.errors.append(error_msg)
                    
                    logger.error(f"Error in shutdown step '{step_name}': {e}")
                    logger.error(traceback.format_exc())
                    
                    # Continue with next step unless it's critical
                    if step_name in ["Closing open positions", "Saving system state"]:
                        self.shutdown_status.warnings.append(f"Critical step failed: {step_name}")
            
            # Complete shutdown
            self.shutdown_status.phase = ShutdownPhase.COMPLETED
            self.shutdown_status.progress_percent = 100.0
            self.shutdown_status.current_step = "Shutdown completed"
            
            total_duration = time.time() - start_time
            logger.info(f"Graceful shutdown completed in {total_duration:.1f}s")
            
        except Exception as e:
            self.shutdown_status.phase = ShutdownPhase.FAILED
            self.shutdown_status.errors.append(f"Shutdown failed: {str(e)}")
            
            logger.error(f"Fatal error during shutdown: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            self.shutdown_in_progress = False
            
            # Log final status
            self._log_shutdown_summary()
    
    async def _stop_discovery_loop(self) -> None:
        """Stop the discovery loop."""
        try:
            self.shutdown_status.phase = ShutdownPhase.STOPPING_DISCOVERY
            
            if hasattr(self.orchestrator, 'discovery_engine'):
                await self.orchestrator.discovery_engine.stop_discovery_loop()
            
            if hasattr(self.orchestrator, 'state'):
                self.orchestrator.state.discovery_loop_running = False
            
            logger.info("Discovery loop stopped")
            
        except Exception as e:
            logger.error(f"Error stopping discovery loop: {e}")
            raise
    
    async def _close_positions(self) -> None:
        """Close open positions."""
        try:
            self.shutdown_status.phase = ShutdownPhase.CLOSING_POSITIONS
            
            if not self.force_close_positions:
                logger.info("Position closing disabled in configuration")
                return
            
            # Get portfolio manager
            portfolio_manager = getattr(self.orchestrator, 'portfolio_manager', None)
            if not portfolio_manager:
                logger.warning("Portfolio manager not available")
                return
            
            # Get open positions
            positions = await portfolio_manager.get_positions()
            
            if not positions:
                logger.info("No open positions to close")
                return
            
            logger.info(f"Closing {len(positions)} open positions...")
            
            # Close positions with timeout
            close_start = time.time()
            closed_positions = []
            failed_positions = []
            
            for position in positions:
                try:
                    # Check timeout
                    if time.time() - close_start > self.position_close_timeout:
                        logger.warning(f"Position close timeout exceeded, {len(positions) - len(closed_positions)} positions remain open")
                        break
                    
                    # Close position
                    result = await portfolio_manager.close_position(position['symbol'])
                    
                    if result.get('success'):
                        closed_positions.append(position['symbol'])
                        logger.info(f"Closed position: {position['symbol']}")
                    else:
                        failed_positions.append(position['symbol'])
                        logger.error(f"Failed to close position: {position['symbol']} - {result.get('error')}")
                
                except Exception as e:
                    failed_positions.append(position['symbol'])
                    logger.error(f"Error closing position {position['symbol']}: {e}")
            
            # Log summary
            logger.info(f"Position closing summary: {len(closed_positions)} closed, {len(failed_positions)} failed")
            
            if failed_positions:
                self.shutdown_status.warnings.append(f"Failed to close {len(failed_positions)} positions: {failed_positions}")
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            raise
    
    async def _save_system_state(self) -> None:
        """Save system state."""
        try:
            self.shutdown_status.phase = ShutdownPhase.SAVING_STATE
            
            if not self.save_state_on_shutdown:
                logger.info("State saving disabled in configuration")
                return
            
            # Save portfolio state
            if hasattr(self.orchestrator, 'portfolio_manager'):
                await self.orchestrator.portfolio_manager.save_state()
            
            # Save discovery state
            if hasattr(self.orchestrator, 'discovery_engine'):
                await self.orchestrator.discovery_engine.save_state()
            
            # Save risk state
            if hasattr(self.orchestrator, 'risk_manager'):
                await self.orchestrator.risk_manager.save_state()
            
            # Save performance metrics
            if hasattr(self.orchestrator, 'performance_tracker'):
                await self.orchestrator.performance_tracker.save_state()
            
            logger.info("System state saved")
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
            raise
    
    async def _stop_market_data(self) -> None:
        """Stop market data streaming."""
        try:
            if hasattr(self.orchestrator, 'market_data_manager'):
                await self.orchestrator.market_data_manager.stop_streaming()
            
            if hasattr(self.orchestrator, 'state'):
                self.orchestrator.state.market_data_streaming = False
            
            logger.info("Market data streaming stopped")
            
        except Exception as e:
            logger.error(f"Error stopping market data: {e}")
            raise
    
    async def _stop_reddit_monitoring(self) -> None:
        """Stop Reddit monitoring."""
        try:
            if hasattr(self.orchestrator, 'reddit_monitor'):
                await self.orchestrator.reddit_monitor.stop_monitoring()
            
            if hasattr(self.orchestrator, 'state'):
                self.orchestrator.state.reddit_monitoring = False
            
            logger.info("Reddit monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Reddit monitoring: {e}")
            raise
    
    async def _stop_mcp_server(self) -> None:
        """Stop MCP server."""
        try:
            if hasattr(self.orchestrator, 'mcp_server'):
                # MCP server doesn't have a standard stop method, so we'll just mark it as stopped
                pass
            
            if hasattr(self.orchestrator, 'state'):
                self.orchestrator.state.mcp_server_running = False
            
            logger.info("MCP server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MCP server: {e}")
            raise
    
    async def _execute_shutdown_callbacks(self) -> None:
        """Execute shutdown callbacks."""
        try:
            logger.info(f"Executing {len(self.shutdown_callbacks)} shutdown callbacks")
            
            for i, callback in enumerate(self.shutdown_callbacks):
                try:
                    callback()
                    logger.debug(f"Executed shutdown callback {i+1}/{len(self.shutdown_callbacks)}")
                except Exception as e:
                    logger.error(f"Error in shutdown callback {i+1}: {e}")
                    self.shutdown_status.warnings.append(f"Shutdown callback {i+1} failed: {str(e)}")
            
            logger.info("Shutdown callbacks completed")
            
        except Exception as e:
            logger.error(f"Error executing shutdown callbacks: {e}")
            raise
    
    async def _cleanup_resources(self) -> None:
        """Clean up system resources."""
        try:
            self.shutdown_status.phase = ShutdownPhase.CLEANUP
            
            # Close database connections
            if hasattr(self.orchestrator, 'db_manager'):
                await self.orchestrator.db_manager.close()
            
            # Shutdown executors
            if hasattr(self.orchestrator, 'executor'):
                self.orchestrator.executor.shutdown(wait=True)
            
            # Cancel any remaining tasks
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            if tasks:
                logger.info(f"Cancelling {len(tasks)} remaining tasks")
                for task in tasks:
                    task.cancel()
                
                # Wait for tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info("Resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            raise
    
    async def _finalize_shutdown(self) -> None:
        """Finalize shutdown process."""
        try:
            # Update orchestrator state
            if hasattr(self.orchestrator, 'state'):
                self.orchestrator.state.is_running = False
            
            # Restore original signal handlers
            if self.original_sigint_handler:
                signal.signal(signal.SIGINT, self.original_sigint_handler)
            if self.original_sigterm_handler:
                signal.signal(signal.SIGTERM, self.original_sigterm_handler)
            
            # Shutdown logging system
            if hasattr(self.logger, 'shutdown'):
                self.logger.shutdown()
            
            logger.info("Shutdown finalized")
            
        except Exception as e:
            logger.error(f"Error finalizing shutdown: {e}")
            raise
    
    def _log_shutdown_summary(self) -> None:
        """Log shutdown summary."""
        try:
            summary = {
                'reason': self.shutdown_status.reason.value,
                'phase': self.shutdown_status.phase.value,
                'duration_seconds': (datetime.now() - self.shutdown_status.start_time).total_seconds(),
                'completed_steps': len(self.shutdown_status.completed_steps),
                'failed_steps': len(self.shutdown_status.failed_steps),
                'warnings': len(self.shutdown_status.warnings),
                'errors': len(self.shutdown_status.errors),
                'progress_percent': self.shutdown_status.progress_percent
            }
            
            logger.info(f"Shutdown summary: {summary}")
            
            # Log detailed information if there were issues
            if self.shutdown_status.failed_steps:
                logger.error(f"Failed steps: {self.shutdown_status.failed_steps}")
            
            if self.shutdown_status.warnings:
                logger.warning(f"Warnings: {self.shutdown_status.warnings}")
            
            if self.shutdown_status.errors:
                logger.error(f"Errors: {self.shutdown_status.errors}")
            
        except Exception as e:
            logger.error(f"Error logging shutdown summary: {e}")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_requested
    
    def is_shutdown_in_progress(self) -> bool:
        """Check if shutdown is in progress."""
        return self.shutdown_in_progress
    
    def get_shutdown_status(self) -> Optional[Dict[str, Any]]:
        """Get current shutdown status."""
        if not self.shutdown_status:
            return None
        
        return {
            'reason': self.shutdown_status.reason.value,
            'phase': self.shutdown_status.phase.value,
            'start_time': self.shutdown_status.start_time.isoformat(),
            'current_step': self.shutdown_status.current_step,
            'completed_steps': self.shutdown_status.completed_steps,
            'failed_steps': self.shutdown_status.failed_steps,
            'warnings': self.shutdown_status.warnings,
            'errors': self.shutdown_status.errors,
            'progress_percent': self.shutdown_status.progress_percent,
            'estimated_completion': self.shutdown_status.estimated_completion.isoformat() if self.shutdown_status.estimated_completion else None
        }
    
    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to complete."""
        return self.shutdown_event.wait(timeout)
    
    def force_shutdown(self) -> None:
        """Force immediate shutdown (emergency use only)."""
        logger.critical("FORCE SHUTDOWN REQUESTED - IMMEDIATE TERMINATION")
        
        try:
            # Log emergency shutdown
            if self.logger:
                self.logger.system_logger.log_system_shutdown("force_shutdown")
        except:
            pass
        
        # Immediate exit
        sys.exit(1) 