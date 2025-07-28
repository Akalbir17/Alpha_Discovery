#!/usr/bin/env python3
"""
Alpha Discovery Database Migration Script

This script handles database schema creation and migrations for the Alpha Discovery platform.
It creates all necessary tables, indexes, and constraints for the trading system.
"""

import os
import sys
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime
import hashlib
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handles database migrations for Alpha Discovery platform."""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.migration_history = []
        
        # Database connection parameters
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'alpha_discovery'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '')
        }
    
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self.cursor = self.connection.cursor()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")
    
    def create_migration_table(self):
        """Create migration tracking table."""
        sql = """
        CREATE TABLE IF NOT EXISTS migration_history (
            id SERIAL PRIMARY KEY,
            migration_name VARCHAR(255) NOT NULL UNIQUE,
            migration_hash VARCHAR(64) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            applied_by VARCHAR(100) DEFAULT CURRENT_USER
        );
        """
        self.cursor.execute(sql)
        logger.info("Migration history table created")
    
    def is_migration_applied(self, migration_name, migration_hash):
        """Check if migration has been applied."""
        sql = """
        SELECT COUNT(*) FROM migration_history 
        WHERE migration_name = %s AND migration_hash = %s
        """
        self.cursor.execute(sql, (migration_name, migration_hash))
        return self.cursor.fetchone()[0] > 0
    
    def record_migration(self, migration_name, migration_hash):
        """Record migration in history."""
        sql = """
        INSERT INTO migration_history (migration_name, migration_hash)
        VALUES (%s, %s)
        """
        self.cursor.execute(sql, (migration_name, migration_hash))
        logger.info(f"Migration recorded: {migration_name}")
    
    def execute_migration(self, migration_name, migration_sql):
        """Execute a migration if not already applied."""
        migration_hash = hashlib.sha256(migration_sql.encode()).hexdigest()
        
        if self.is_migration_applied(migration_name, migration_hash):
            logger.info(f"Migration {migration_name} already applied, skipping")
            return
        
        try:
            logger.info(f"Applying migration: {migration_name}")
            self.cursor.execute(migration_sql)
            self.record_migration(migration_name, migration_hash)
            logger.info(f"Migration {migration_name} applied successfully")
        except Exception as e:
            logger.error(f"Failed to apply migration {migration_name}: {e}")
            raise
    
    def run_all_migrations(self):
        """Run all database migrations."""
        logger.info("Starting database migrations")
        
        # Create migration tracking table
        self.create_migration_table()
        
        # Run migrations in order
        migrations = [
            ("001_create_users_table", self.create_users_table),
            ("002_create_portfolios_table", self.create_portfolios_table),
            ("003_create_positions_table", self.create_positions_table),
            ("004_create_orders_table", self.create_orders_table),
            ("005_create_executions_table", self.create_executions_table),
            ("006_create_strategies_table", self.create_strategies_table),
            ("007_create_signals_table", self.create_signals_table),
            ("008_create_risk_metrics_table", self.create_risk_metrics_table),
            ("009_create_performance_table", self.create_performance_table),
            ("010_create_audit_log_table", self.create_audit_log_table),
            ("011_create_configurations_table", self.create_configurations_table),
            ("012_create_alerts_table", self.create_alerts_table),
            ("013_create_indexes", self.create_indexes),
            ("014_create_functions", self.create_functions),
            ("015_create_triggers", self.create_triggers),
            ("016_insert_initial_data", self.insert_initial_data)
        ]
        
        for migration_name, migration_func in migrations:
            migration_sql = migration_func()
            self.execute_migration(migration_name, migration_sql)
        
        logger.info("All migrations completed successfully")
    
    def create_users_table(self):
        """Create users table."""
        return """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            first_name VARCHAR(50),
            last_name VARCHAR(50),
            role VARCHAR(20) DEFAULT 'user',
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            api_key VARCHAR(64) UNIQUE,
            api_key_created_at TIMESTAMP
        );
        """
    
    def create_portfolios_table(self):
        """Create portfolios table."""
        return """
        CREATE TABLE IF NOT EXISTS portfolios (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            name VARCHAR(100) NOT NULL,
            description TEXT,
            initial_capital DECIMAL(15,2) NOT NULL,
            current_value DECIMAL(15,2) NOT NULL DEFAULT 0,
            cash_balance DECIMAL(15,2) NOT NULL DEFAULT 0,
            total_pnl DECIMAL(15,2) DEFAULT 0,
            daily_pnl DECIMAL(15,2) DEFAULT 0,
            strategy_config JSONB,
            risk_config JSONB,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    
    def create_positions_table(self):
        """Create positions table."""
        return """
        CREATE TABLE IF NOT EXISTS positions (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
            symbol VARCHAR(10) NOT NULL,
            quantity DECIMAL(15,4) NOT NULL,
            average_price DECIMAL(15,4) NOT NULL,
            current_price DECIMAL(15,4),
            market_value DECIMAL(15,2),
            unrealized_pnl DECIMAL(15,2),
            realized_pnl DECIMAL(15,2) DEFAULT 0,
            position_type VARCHAR(10) DEFAULT 'LONG',
            sector VARCHAR(50),
            asset_class VARCHAR(20),
            opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP,
            UNIQUE(portfolio_id, symbol)
        );
        """
    
    def create_orders_table(self):
        """Create orders table."""
        return """
        CREATE TABLE IF NOT EXISTS orders (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
            symbol VARCHAR(10) NOT NULL,
            order_type VARCHAR(20) NOT NULL,
            side VARCHAR(10) NOT NULL,
            quantity DECIMAL(15,4) NOT NULL,
            price DECIMAL(15,4),
            stop_price DECIMAL(15,4),
            time_in_force VARCHAR(10) DEFAULT 'DAY',
            status VARCHAR(20) DEFAULT 'PENDING',
            filled_quantity DECIMAL(15,4) DEFAULT 0,
            average_fill_price DECIMAL(15,4),
            commission DECIMAL(10,4) DEFAULT 0,
            strategy_id INTEGER,
            signal_id INTEGER,
            broker_order_id VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            filled_at TIMESTAMP,
            cancelled_at TIMESTAMP
        );
        """
    
    def create_executions_table(self):
        """Create executions table."""
        return """
        CREATE TABLE IF NOT EXISTS executions (
            id SERIAL PRIMARY KEY,
            order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
            portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
            symbol VARCHAR(10) NOT NULL,
            side VARCHAR(10) NOT NULL,
            quantity DECIMAL(15,4) NOT NULL,
            price DECIMAL(15,4) NOT NULL,
            commission DECIMAL(10,4) DEFAULT 0,
            execution_id VARCHAR(100),
            broker_execution_id VARCHAR(100),
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    
    def create_strategies_table(self):
        """Create strategies table."""
        return """
        CREATE TABLE IF NOT EXISTS strategies (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            description TEXT,
            strategy_type VARCHAR(50) NOT NULL,
            parameters JSONB,
            is_active BOOLEAN DEFAULT true,
            created_by INTEGER REFERENCES users(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    
    def create_signals_table(self):
        """Create signals table."""
        return """
        CREATE TABLE IF NOT EXISTS signals (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
            symbol VARCHAR(10) NOT NULL,
            signal_type VARCHAR(20) NOT NULL,
            strength DECIMAL(5,4) NOT NULL,
            confidence DECIMAL(5,4) NOT NULL,
            side VARCHAR(10) NOT NULL,
            target_price DECIMAL(15,4),
            stop_loss DECIMAL(15,4),
            take_profit DECIMAL(15,4),
            metadata JSONB,
            is_active BOOLEAN DEFAULT true,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        );
        """
    
    def create_risk_metrics_table(self):
        """Create risk metrics table."""
        return """
        CREATE TABLE IF NOT EXISTS risk_metrics (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
            metric_date DATE NOT NULL,
            var_95 DECIMAL(15,4),
            var_99 DECIMAL(15,4),
            expected_shortfall_95 DECIMAL(15,4),
            expected_shortfall_99 DECIMAL(15,4),
            max_drawdown DECIMAL(15,4),
            volatility DECIMAL(15,4),
            beta DECIMAL(15,4),
            sharpe_ratio DECIMAL(15,4),
            sortino_ratio DECIMAL(15,4),
            calmar_ratio DECIMAL(15,4),
            portfolio_value DECIMAL(15,2),
            leverage DECIMAL(15,4),
            concentration_risk DECIMAL(15,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(portfolio_id, metric_date)
        );
        """
    
    def create_performance_table(self):
        """Create performance table."""
        return """
        CREATE TABLE IF NOT EXISTS performance (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
            performance_date DATE NOT NULL,
            daily_return DECIMAL(15,6),
            cumulative_return DECIMAL(15,6),
            benchmark_return DECIMAL(15,6),
            active_return DECIMAL(15,6),
            portfolio_value DECIMAL(15,2),
            cash_balance DECIMAL(15,2),
            total_pnl DECIMAL(15,2),
            realized_pnl DECIMAL(15,2),
            unrealized_pnl DECIMAL(15,2),
            number_of_trades INTEGER DEFAULT 0,
            win_rate DECIMAL(5,4),
            profit_factor DECIMAL(15,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(portfolio_id, performance_date)
        );
        """
    
    def create_audit_log_table(self):
        """Create audit log table."""
        return """
        CREATE TABLE IF NOT EXISTS audit_log (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            action VARCHAR(50) NOT NULL,
            table_name VARCHAR(50),
            record_id INTEGER,
            old_values JSONB,
            new_values JSONB,
            ip_address INET,
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    
    def create_configurations_table(self):
        """Create configurations table."""
        return """
        CREATE TABLE IF NOT EXISTS configurations (
            id SERIAL PRIMARY KEY,
            config_key VARCHAR(100) UNIQUE NOT NULL,
            config_value JSONB NOT NULL,
            config_type VARCHAR(20) DEFAULT 'system',
            description TEXT,
            is_encrypted BOOLEAN DEFAULT false,
            created_by INTEGER REFERENCES users(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    
    def create_alerts_table(self):
        """Create alerts table."""
        return """
        CREATE TABLE IF NOT EXISTS alerts (
            id SERIAL PRIMARY KEY,
            alert_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            title VARCHAR(200) NOT NULL,
            message TEXT NOT NULL,
            source VARCHAR(50),
            metadata JSONB,
            is_acknowledged BOOLEAN DEFAULT false,
            acknowledged_by INTEGER REFERENCES users(id),
            acknowledged_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    
    def create_indexes(self):
        """Create database indexes."""
        return """
        -- Users table indexes
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);
        
        -- Portfolios table indexes
        CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id);
        CREATE INDEX IF NOT EXISTS idx_portfolios_active ON portfolios(is_active);
        
        -- Positions table indexes
        CREATE INDEX IF NOT EXISTS idx_positions_portfolio_id ON positions(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
        CREATE INDEX IF NOT EXISTS idx_positions_sector ON positions(sector);
        CREATE INDEX IF NOT EXISTS idx_positions_asset_class ON positions(asset_class);
        
        -- Orders table indexes
        CREATE INDEX IF NOT EXISTS idx_orders_portfolio_id ON orders(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
        CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
        CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
        CREATE INDEX IF NOT EXISTS idx_orders_broker_order_id ON orders(broker_order_id);
        
        -- Executions table indexes
        CREATE INDEX IF NOT EXISTS idx_executions_order_id ON executions(order_id);
        CREATE INDEX IF NOT EXISTS idx_executions_portfolio_id ON executions(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_executions_symbol ON executions(symbol);
        CREATE INDEX IF NOT EXISTS idx_executions_executed_at ON executions(executed_at);
        
        -- Strategies table indexes
        CREATE INDEX IF NOT EXISTS idx_strategies_name ON strategies(name);
        CREATE INDEX IF NOT EXISTS idx_strategies_type ON strategies(strategy_type);
        CREATE INDEX IF NOT EXISTS idx_strategies_active ON strategies(is_active);
        
        -- Signals table indexes
        CREATE INDEX IF NOT EXISTS idx_signals_strategy_id ON signals(strategy_id);
        CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
        CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type);
        CREATE INDEX IF NOT EXISTS idx_signals_generated_at ON signals(generated_at);
        CREATE INDEX IF NOT EXISTS idx_signals_active ON signals(is_active);
        
        -- Risk metrics table indexes
        CREATE INDEX IF NOT EXISTS idx_risk_metrics_portfolio_id ON risk_metrics(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_risk_metrics_date ON risk_metrics(metric_date);
        
        -- Performance table indexes
        CREATE INDEX IF NOT EXISTS idx_performance_portfolio_id ON performance(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(performance_date);
        
        -- Audit log table indexes
        CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
        CREATE INDEX IF NOT EXISTS idx_audit_log_table_name ON audit_log(table_name);
        CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);
        
        -- Configurations table indexes
        CREATE INDEX IF NOT EXISTS idx_configurations_key ON configurations(config_key);
        CREATE INDEX IF NOT EXISTS idx_configurations_type ON configurations(config_type);
        
        -- Alerts table indexes
        CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
        CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
        CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(is_acknowledged);
        CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
        """
    
    def create_functions(self):
        """Create database functions."""
        return """
        -- Function to update updated_at timestamp
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        -- Function to calculate portfolio value
        CREATE OR REPLACE FUNCTION calculate_portfolio_value(p_portfolio_id INTEGER)
        RETURNS DECIMAL(15,2) AS $$
        DECLARE
            total_value DECIMAL(15,2);
        BEGIN
            SELECT COALESCE(SUM(market_value), 0) INTO total_value
            FROM positions
            WHERE portfolio_id = p_portfolio_id;
            
            RETURN total_value;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Function to calculate position PnL
        CREATE OR REPLACE FUNCTION calculate_position_pnl(
            p_quantity DECIMAL(15,4),
            p_average_price DECIMAL(15,4),
            p_current_price DECIMAL(15,4)
        )
        RETURNS DECIMAL(15,2) AS $$
        BEGIN
            RETURN (p_current_price - p_average_price) * p_quantity;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Function to get portfolio performance
        CREATE OR REPLACE FUNCTION get_portfolio_performance(
            p_portfolio_id INTEGER,
            p_start_date DATE,
            p_end_date DATE
        )
        RETURNS TABLE(
            performance_date DATE,
            daily_return DECIMAL(15,6),
            cumulative_return DECIMAL(15,6),
            portfolio_value DECIMAL(15,2)
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                p.performance_date,
                p.daily_return,
                p.cumulative_return,
                p.portfolio_value
            FROM performance p
            WHERE p.portfolio_id = p_portfolio_id
            AND p.performance_date BETWEEN p_start_date AND p_end_date
            ORDER BY p.performance_date;
        END;
        $$ LANGUAGE plpgsql;
        """
    
    def create_triggers(self):
        """Create database triggers."""
        return """
        -- Trigger to update updated_at on users table
        DROP TRIGGER IF EXISTS update_users_updated_at ON users;
        CREATE TRIGGER update_users_updated_at
            BEFORE UPDATE ON users
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        
        -- Trigger to update updated_at on portfolios table
        DROP TRIGGER IF EXISTS update_portfolios_updated_at ON portfolios;
        CREATE TRIGGER update_portfolios_updated_at
            BEFORE UPDATE ON portfolios
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        
        -- Trigger to update updated_at on positions table
        DROP TRIGGER IF EXISTS update_positions_updated_at ON positions;
        CREATE TRIGGER update_positions_updated_at
            BEFORE UPDATE ON positions
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        
        -- Trigger to update updated_at on orders table
        DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;
        CREATE TRIGGER update_orders_updated_at
            BEFORE UPDATE ON orders
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        
        -- Trigger to update updated_at on strategies table
        DROP TRIGGER IF EXISTS update_strategies_updated_at ON strategies;
        CREATE TRIGGER update_strategies_updated_at
            BEFORE UPDATE ON strategies
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        
        -- Trigger to update updated_at on configurations table
        DROP TRIGGER IF EXISTS update_configurations_updated_at ON configurations;
        CREATE TRIGGER update_configurations_updated_at
            BEFORE UPDATE ON configurations
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """
    
    def insert_initial_data(self):
        """Insert initial data."""
        return """
        -- Insert default admin user
        INSERT INTO users (username, email, password_hash, first_name, last_name, role, api_key)
        VALUES (
            'admin',
            'admin@alphadiscovery.com',
            '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/8K.8k6u6G',  -- password: admin123
            'Admin',
            'User',
            'admin',
            'ak_' || encode(gen_random_bytes(32), 'hex')
        )
        ON CONFLICT (username) DO NOTHING;
        
        -- Insert default strategies
        INSERT INTO strategies (name, description, strategy_type, parameters)
        VALUES 
            (
                'Momentum Strategy',
                'Momentum-based trading strategy using technical indicators',
                'momentum',
                '{"lookback_period": 20, "momentum_threshold": 0.05}'::jsonb
            ),
            (
                'Mean Reversion Strategy',
                'Mean reversion strategy using statistical analysis',
                'mean_reversion',
                '{"lookback_period": 50, "deviation_threshold": 2.0}'::jsonb
            ),
            (
                'Arbitrage Strategy',
                'Statistical arbitrage strategy',
                'arbitrage',
                '{"min_spread": 0.001, "max_spread": 0.01}'::jsonb
            )
        ON CONFLICT DO NOTHING;
        
        -- Insert default configuration
        INSERT INTO configurations (config_key, config_value, config_type, description)
        VALUES
            (
                'system.version',
                '"1.0.0"'::jsonb,
                'system',
                'System version'
            ),
            (
                'trading.enabled',
                'true'::jsonb,
                'trading',
                'Enable/disable trading'
            ),
            (
                'risk.max_daily_var',
                '0.02'::jsonb,
                'risk',
                'Maximum daily VaR limit'
            )
        ON CONFLICT (config_key) DO NOTHING;
        """

def main():
    """Main function to run database migrations."""
    logger.info("Starting Alpha Discovery database migrations")
    
    migrator = DatabaseMigrator()
    
    try:
        # Connect to database
        migrator.connect()
        
        # Run all migrations
        migrator.run_all_migrations()
        
        logger.info("Database migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
    
    finally:
        # Disconnect from database
        migrator.disconnect()

if __name__ == "__main__":
    main() 