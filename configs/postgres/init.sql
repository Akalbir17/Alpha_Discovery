-- TimescaleDB initialization script for Alpha Discovery Platform
-- This script sets up the database schema and hypertables for time-series data

-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS strategy_data;
CREATE SCHEMA IF NOT EXISTS performance_data;
CREATE SCHEMA IF NOT EXISTS risk_data;

-- Market Data Tables
CREATE TABLE IF NOT EXISTS market_data.price_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL(15,6),
    high DECIMAL(15,6),
    low DECIMAL(15,6),
    close DECIMAL(15,6),
    volume BIGINT,
    adjusted_close DECIMAL(15,6),
    PRIMARY KEY (time, symbol)
);

CREATE TABLE IF NOT EXISTS market_data.tick_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(15,6),
    size INTEGER,
    side VARCHAR(4), -- 'BUY' or 'SELL'
    exchange VARCHAR(10),
    PRIMARY KEY (time, symbol)
);

CREATE TABLE IF NOT EXISTS market_data.orderbook_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    level INTEGER NOT NULL,
    bid_price DECIMAL(15,6),
    bid_size DECIMAL(15,6),
    ask_price DECIMAL(15,6),
    ask_size DECIMAL(15,6),
    PRIMARY KEY (time, symbol, level)
);

-- Strategy Data Tables
CREATE TABLE IF NOT EXISTS strategy_data.strategy_signals (
    time TIMESTAMPTZ NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    signal_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20),
    signal_strength DECIMAL(5,4),
    confidence DECIMAL(5,4),
    metadata JSONB,
    PRIMARY KEY (time, strategy_id, signal_id)
);

CREATE TABLE IF NOT EXISTS strategy_data.position_data (
    time TIMESTAMPTZ NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    position_size DECIMAL(15,6),
    average_price DECIMAL(15,6),
    market_value DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    PRIMARY KEY (time, strategy_id, symbol)
);

-- Performance Data Tables
CREATE TABLE IF NOT EXISTS performance_data.strategy_performance (
    time TIMESTAMPTZ NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    total_return DECIMAL(8,6),
    daily_return DECIMAL(8,6),
    cumulative_return DECIMAL(8,6),
    sharpe_ratio DECIMAL(6,4),
    sortino_ratio DECIMAL(6,4),
    max_drawdown DECIMAL(6,4),
    volatility DECIMAL(6,4),
    alpha DECIMAL(6,4),
    beta DECIMAL(6,4),
    PRIMARY KEY (time, strategy_id)
);

CREATE TABLE IF NOT EXISTS performance_data.pnl_data (
    time TIMESTAMPTZ NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    realized_pnl DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    total_pnl DECIMAL(15,2),
    fees DECIMAL(15,2),
    slippage DECIMAL(15,2),
    PRIMARY KEY (time, strategy_id)
);

-- Risk Data Tables
CREATE TABLE IF NOT EXISTS risk_data.risk_metrics (
    time TIMESTAMPTZ NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    var_95 DECIMAL(15,2),
    var_99 DECIMAL(15,2),
    expected_shortfall DECIMAL(15,2),
    maximum_drawdown DECIMAL(6,4),
    volatility DECIMAL(6,4),
    beta DECIMAL(6,4),
    correlation DECIMAL(6,4),
    PRIMARY KEY (time, strategy_id)
);

CREATE TABLE IF NOT EXISTS risk_data.stress_test_results (
    time TIMESTAMPTZ NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    scenario_name VARCHAR(100) NOT NULL,
    loss_amount DECIMAL(15,2),
    loss_percentage DECIMAL(6,4),
    PRIMARY KEY (time, strategy_id, scenario_name)
);

-- Create hypertables for time-series data
SELECT create_hypertable('market_data.price_data', 'time', chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('market_data.tick_data', 'time', chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('market_data.orderbook_data', 'time', chunk_time_interval => INTERVAL '1 hour');
SELECT create_hypertable('strategy_data.strategy_signals', 'time', chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('strategy_data.position_data', 'time', chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('performance_data.strategy_performance', 'time', chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('performance_data.pnl_data', 'time', chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('risk_data.risk_metrics', 'time', chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('risk_data.stress_test_results', 'time', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_price_data_symbol_time ON market_data.price_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_tick_data_symbol_time ON market_data.tick_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_signals_strategy_time ON strategy_data.strategy_signals (strategy_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_position_data_strategy_time ON strategy_data.position_data (strategy_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_performance_strategy_time ON performance_data.strategy_performance (strategy_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_strategy_time ON risk_data.risk_metrics (strategy_id, time DESC);

-- Create continuous aggregates for better performance
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data.price_data_1h
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume
FROM market_data.price_data
GROUP BY bucket, symbol;

CREATE MATERIALIZED VIEW IF NOT EXISTS performance_data.daily_performance
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS bucket,
    strategy_id,
    last(total_return, time) AS total_return,
    last(cumulative_return, time) AS cumulative_return,
    avg(sharpe_ratio) AS avg_sharpe_ratio,
    min(max_drawdown) AS max_drawdown,
    avg(volatility) AS avg_volatility
FROM performance_data.strategy_performance
GROUP BY bucket, strategy_id;

-- Enable compression for older data
-- SELECT add_compression_policy('market_data.price_data', INTERVAL '7 days');
-- SELECT add_compression_policy('market_data.tick_data', INTERVAL '1 day');
-- SELECT add_compression_policy('strategy_data.strategy_signals', INTERVAL '30 days');
-- SELECT add_compression_policy('performance_data.strategy_performance', INTERVAL '30 days');

-- Create retention policies
-- SELECT add_retention_policy('market_data.tick_data', INTERVAL '30 days');
-- SELECT add_retention_policy('market_data.orderbook_data', INTERVAL '7 days');

-- Create database users (safe pattern)
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'grafana_user') THEN
      CREATE USER grafana_user WITH PASSWORD 'grafana_password';
   END IF;
END
$$;

DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'jupyter_user') THEN
      CREATE USER jupyter_user WITH PASSWORD 'jupyter_password';
   END IF;
END
$$;

DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'api_user') THEN
      CREATE USER api_user WITH PASSWORD 'api_password';
   END IF;
END
$$;

-- Grant permissions
GRANT USAGE ON SCHEMA market_data TO grafana_user, jupyter_user, api_user;
GRANT USAGE ON SCHEMA strategy_data TO grafana_user, jupyter_user, api_user;
GRANT USAGE ON SCHEMA performance_data TO grafana_user, jupyter_user, api_user;
GRANT USAGE ON SCHEMA risk_data TO grafana_user, jupyter_user, api_user;

GRANT SELECT ON ALL TABLES IN SCHEMA market_data TO grafana_user, jupyter_user;
GRANT SELECT ON ALL TABLES IN SCHEMA strategy_data TO grafana_user, jupyter_user;
GRANT SELECT ON ALL TABLES IN SCHEMA performance_data TO grafana_user, jupyter_user;
GRANT SELECT ON ALL TABLES IN SCHEMA risk_data TO grafana_user, jupyter_user;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA market_data TO api_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA strategy_data TO api_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA performance_data TO api_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA risk_data TO api_user;

-- Commented out compression and retention policies for compatibility
-- SELECT add_compression_policy('market_data.price_data', INTERVAL '7 days');
-- SELECT add_compression_policy('market_data.tick_data', INTERVAL '1 day');
-- SELECT add_compression_policy('strategy_data.strategy_signals', INTERVAL '30 days');
-- SELECT add_compression_policy('performance_data.strategy_performance', INTERVAL '30 days');
-- SELECT add_retention_policy('market_data.tick_data', INTERVAL '30 days');
-- SELECT add_retention_policy('market_data.orderbook_data', INTERVAL '7 days');

-- Create functions for common queries
CREATE OR REPLACE FUNCTION get_latest_price(symbol_param VARCHAR(20))
RETURNS TABLE(symbol VARCHAR(20), price DECIMAL(15,6), price_time TIMESTAMPTZ) AS $$
BEGIN
    RETURN QUERY
    SELECT p.symbol, p.close, p.time
    FROM market_data.price_data p
    WHERE p.symbol = symbol_param
    ORDER BY p.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_strategy_performance_summary(strategy_param VARCHAR(50), days_back INTEGER DEFAULT 30)
RETURNS TABLE(
    strategy_id VARCHAR(50),
    total_return DECIMAL(8,6),
    sharpe_ratio DECIMAL(6,4),
    max_drawdown DECIMAL(6,4),
    volatility DECIMAL(6,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.strategy_id,
        p.total_return,
        p.sharpe_ratio,
        p.max_drawdown,
        p.volatility
    FROM performance_data.strategy_performance p
    WHERE p.strategy_id = strategy_param
    AND p.time >= NOW() - INTERVAL '1 day' * days_back
    ORDER BY p.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Insert sample data for testing
INSERT INTO market_data.price_data (time, symbol, open, high, low, close, volume, adjusted_close)
VALUES 
    (NOW() - INTERVAL '1 hour', 'AAPL', 150.00, 152.00, 149.50, 151.25, 1000000, 151.25),
    (NOW() - INTERVAL '1 hour', 'GOOGL', 2800.00, 2820.00, 2790.00, 2810.50, 500000, 2810.50),
    (NOW() - INTERVAL '1 hour', 'MSFT', 300.00, 302.00, 299.00, 301.75, 800000, 301.75);

INSERT INTO strategy_data.strategy_signals (time, strategy_id, signal_id, symbol, signal_type, signal_strength, confidence, metadata)
VALUES 
    (NOW(), 'momentum_strategy', 'signal_001', 'AAPL', 'BUY', 0.75, 0.85, '{"factor": "momentum", "lookback": 20}'),
    (NOW(), 'mean_reversion_strategy', 'signal_002', 'GOOGL', 'SELL', -0.60, 0.78, '{"factor": "mean_reversion", "zscore": -2.1}');

INSERT INTO performance_data.strategy_performance (time, strategy_id, total_return, daily_return, cumulative_return, sharpe_ratio, max_drawdown, volatility, alpha, beta)
VALUES 
    (NOW(), 'momentum_strategy', 0.15, 0.002, 0.15, 1.25, -0.05, 0.12, 0.08, 0.95),
    (NOW(), 'mean_reversion_strategy', 0.12, 0.001, 0.12, 0.95, -0.08, 0.15, 0.06, 1.05);

-- Create notification triggers
CREATE OR REPLACE FUNCTION notify_new_signal() RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('new_signal', row_to_json(NEW)::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER signal_notification
    AFTER INSERT ON strategy_data.strategy_signals
    FOR EACH ROW EXECUTE FUNCTION notify_new_signal();

-- Refresh continuous aggregates (commented out to avoid initialization errors)
-- SELECT refresh_continuous_aggregate('market_data.price_data_1h', NULL, NULL);
-- SELECT refresh_continuous_aggregate('performance_data.daily_performance', NULL, NULL);

-- COMMIT; (removed as it's not needed in initialization script) 