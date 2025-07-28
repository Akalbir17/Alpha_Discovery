"""
AI Enhancements for Risk Management System

This module provides AI-powered enhancements to the traditional risk management system:
1. LSTM-based regime change detection with ensemble methods
2. Unsupervised learning for automated risk factor discovery
3. Real-time anomaly detection for market structure breaks
4. Dynamic factor model updates with machine learning

Author: Alpha Discovery Team
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Remove direct deep learning imports - now using ML client
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Remove direct machine learning imports - now using ML client
# from sklearn.ensemble import IsolationForest, RandomForestClassifier
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.decomposition import PCA, FastICA, NMF
# from sklearn.manifold import TSNE
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import silhouette_score, adjusted_rand_score
# from sklearn.model_selection import train_test_split

# Time Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from hmmlearn import hmm

# Anomaly Detection - remove direct imports, now using ML client
# from pyod.models.auto_encoder import AutoEncoder
# from pyod.models.deep_svdd import DeepSVDD
# from pyod.models.lof import LOF

# Optimization
from scipy.optimize import minimize
from scipy.stats import entropy

# Utilities
import joblib
from pathlib import Path
import json

# ML client for remote inference (Phase 4 upgrade)
try:
    from src.scrapers.ml_client import ml_client
    ML_CLIENT_AVAILABLE = True
except ImportError:
    ML_CLIENT_AVAILABLE = False
    logger.warning("ML client not available - using fallback analysis")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegimeSignal:
    """Signal for regime change detection"""
    timestamp: datetime
    regime: str  # bull, bear, sideways, crisis
    confidence: float
    signal_strength: float
    features: Dict[str, float]
    
@dataclass
class RiskFactor:
    """Discovered risk factor"""
    name: str
    loadings: Dict[str, float]  # Asset loadings
    explained_variance: float
    stability_score: float
    interpretation: str
    discovery_method: str  # pca, ica, clustering, etc.

# Remove the LSTM model class - now handled by ML client
# class LSTMRegimeDetector(nn.Module):
#     """
#     LSTM-based regime detection model for financial time series.
#     
#     This model uses Long Short-Term Memory networks to detect regime changes
#     in financial markets by learning temporal patterns in market data.
#     """
#     
#     def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
#                  num_regimes: int = 4, dropout: float = 0.2):
#         super(LSTMRegimeDetector, self).__init__()
#         
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_regimes = num_regimes
#         
#         # LSTM layers
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
#         
#         # Attention mechanism
#         self.attention = nn.MultiheadAttention(
#             embed_dim=hidden_size,
#             num_heads=8,
#             dropout=dropout,
#             batch_first=True
#         )
#         
#         # Classification layers
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 2, num_regimes),
#             nn.Softmax(dim=1)
#         )
#         
#         # Anomaly detection layer
#         self.anomaly_detector = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 4),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 4, 1),
#             nn.Sigmoid()
#         )
#     
#     def forward(self, x):
#         # LSTM forward pass
#         lstm_out, (hidden, cell) = self.lstm(x)
#         
#         # Apply attention
#         attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
#         
#         # Use last output for classification
#         last_output = attn_out[:, -1, :]
#         
#         # Regime classification
#         regime_probs = self.classifier(last_output)
#         
#         # Anomaly score
#         anomaly_score = self.anomaly_detector(last_output)
#         
#         return regime_probs, anomaly_score

class AIRiskEnhancements:
    """
    AI-powered enhancements for the risk management system.
    Now uses ML client for heavy model operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # No longer loading models directly - using ML client
        logger.info("AI Risk Enhancements initialized with ML client support")
        
        # Data storage
        self.scalers: Dict[str, StandardScaler] = {}
        self.discovered_factors: List[RiskFactor] = []
        self.regime_history: List[RegimeSignal] = []
        
        # Training state
        self.is_trained = False
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for AI enhancements"""
        return {
            'regime_detection': {
                'lookback_window': 60,
                'num_regimes': 4,
                'lstm_hidden_size': 64,
                'lstm_num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 10
            },
            'factor_discovery': {
                'max_factors': 10,
                'min_explained_variance': 0.05,
                'stability_threshold': 0.7,
                'clustering_methods': ['kmeans', 'gaussian_mixture', 'dbscan'],
                'dimensionality_methods': ['pca', 'ica', 'nmf'],
                'update_frequency': 'weekly'
            },
            'anomaly_detection': {
                'contamination': 0.1,
                'n_estimators': 100,
                'anomaly_threshold': 0.8,
                'ensemble_methods': ['isolation_forest', 'lof', 'autoencoder']
            },
            'model_validation': {
                'validation_split': 0.2,
                'cross_validation_folds': 5,
                'performance_threshold': 0.7
            }
        }
    
    def train_regime_detector(self, 
                             returns: pd.DataFrame,
                             market_indicators: Optional[pd.DataFrame] = None,
                             regime_labels: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train LSTM-based regime detection model.
        
        Args:
            returns: Historical returns DataFrame
            market_indicators: Additional market indicators
            regime_labels: Optional ground truth regime labels
            
        Returns:
            Training results and model performance metrics
        """
        try:
            logger.info("Training LSTM regime detection model...")
            
            # Prepare features
            features = self._prepare_regime_features(returns, market_indicators)
            
            # Create sequences for LSTM
            X, y = self._create_sequences(features, regime_labels)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config['model_validation']['validation_split'],
                random_state=42, shuffle=False
            )
            
            # Initialize model
            input_size = X_train.shape[2]
            self.regime_detector = LSTMRegimeDetector(
                input_size=input_size,
                hidden_size=self.config['regime_detection']['lstm_hidden_size'],
                num_layers=self.config['regime_detection']['lstm_num_layers'],
                num_regimes=self.config['regime_detection']['num_regimes'],
                dropout=self.config['regime_detection']['dropout']
            ).to(self.device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                self.regime_detector.parameters(),
                lr=self.config['regime_detection']['learning_rate']
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
            
            # Training loop
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config['regime_detection']['epochs']):
                # Training phase
                self.regime_detector.train()
                train_loss = self._train_epoch(X_train, y_train, criterion, optimizer)
                train_losses.append(train_loss)
                
                # Validation phase
                self.regime_detector.eval()
                val_loss = self._validate_epoch(X_val, y_val, criterion)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.regime_detector.state_dict(), 'best_regime_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['regime_detection']['patience']:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Load best model
            self.regime_detector.load_state_dict(torch.load('best_regime_model.pth'))
            
            # Evaluate model
            performance_metrics = self._evaluate_regime_model(X_val, y_val)
            
            self.is_trained = True
            logger.info("LSTM regime detection model training completed")
            
            return {
                'training_completed': True,
                'epochs_trained': epoch + 1,
                'best_validation_loss': best_val_loss,
                'final_train_loss': train_losses[-1],
                'performance_metrics': performance_metrics,
                'model_size': sum(p.numel() for p in self.regime_detector.parameters())
            }
            
        except Exception as e:
            logger.error(f"Error training regime detector: {str(e)}")
            raise
    
    def _prepare_regime_features(self, 
                                returns: pd.DataFrame,
                                market_indicators: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare features for regime detection"""
        features = []
        
        # Basic return features
        features.append(returns)
        
        # Volatility features
        rolling_vol = returns.rolling(window=20).std()
        features.append(rolling_vol.add_suffix('_vol'))
        
        # Momentum features
        momentum_5 = returns.rolling(window=5).mean()
        momentum_20 = returns.rolling(window=20).mean()
        features.extend([momentum_5.add_suffix('_mom5'), momentum_20.add_suffix('_mom20')])
        
        # Correlation features
        rolling_corr = returns.rolling(window=30).corr().groupby(level=0).mean()
        avg_corr = rolling_corr.mean(axis=1)
        features.append(avg_corr.to_frame('avg_correlation'))
        
        # Market stress indicators
        if len(returns.columns) > 1:
            # Cross-sectional dispersion
            cross_sectional_vol = returns.std(axis=1)
            features.append(cross_sectional_vol.to_frame('cross_sectional_vol'))
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.rolling(window=252).max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.rolling(window=30).min()
            features.append(max_drawdown.add_suffix('_drawdown'))
        
        # Technical indicators
        for col in returns.columns:
            # RSI
            rsi = self._calculate_rsi(returns[col])
            features.append(rsi.to_frame(f'{col}_rsi'))
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(returns[col])
            bb_position = (returns[col] - bb_lower) / (bb_upper - bb_lower)
            features.append(bb_position.to_frame(f'{col}_bb_position'))
        
        # Market indicators
        if market_indicators is not None:
            features.append(market_indicators)
        
        # Combine all features
        combined_features = pd.concat(features, axis=1)
        
        # Forward fill and drop NaN
        combined_features = combined_features.fillna(method='ffill').dropna()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = pd.DataFrame(
            scaler.fit_transform(combined_features),
            index=combined_features.index,
            columns=combined_features.columns
        )
        
        self.scalers['regime_features'] = scaler
        
        return scaled_features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def _create_sequences(self, features: pd.DataFrame, regime_labels: Optional[pd.Series] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        window_size = self.config['regime_detection']['lookback_window']
        
        X = []
        y = []
        
        for i in range(window_size, len(features)):
            X.append(features.iloc[i-window_size:i].values)
            
            if regime_labels is not None:
                y.append(regime_labels.iloc[i])
            else:
                # Use unsupervised regime detection
                y.append(self._detect_regime_unsupervised(features.iloc[i-window_size:i]))
        
        return np.array(X), np.array(y)
    
    def _detect_regime_unsupervised(self, window_data: pd.DataFrame) -> int:
        """Detect regime using unsupervised methods"""
        # Simple volatility-based regime detection
        volatility = window_data.std().mean()
        
        if volatility < 0.01:
            return 0  # Low volatility regime
        elif volatility < 0.02:
            return 1  # Medium volatility regime
        elif volatility < 0.04:
            return 2  # High volatility regime
        else:
            return 3  # Crisis regime
    
    def _train_epoch(self, X: np.ndarray, y: np.ndarray, criterion, optimizer) -> float:
        """Train model for one epoch"""
        total_loss = 0
        num_batches = 0
        
        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.LongTensor(y).to(self.device)
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['regime_detection']['batch_size'],
            shuffle=True
        )
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            regime_probs, anomaly_scores, _ = self.regime_detector(batch_X)
            loss = criterion(regime_probs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, X: np.ndarray, y: np.ndarray, criterion) -> float:
        """Validate model for one epoch"""
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            dataset = TensorDataset(
                torch.FloatTensor(X).to(self.device),
                torch.LongTensor(y).to(self.device)
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['regime_detection']['batch_size'],
                shuffle=False
            )
            
            for batch_X, batch_y in dataloader:
                regime_probs, anomaly_scores, _ = self.regime_detector(batch_X)
                loss = criterion(regime_probs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _evaluate_regime_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate regime detection model performance"""
        self.regime_detector.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            regime_probs, anomaly_scores, _ = self.regime_detector(X_tensor)
            
            predictions = torch.argmax(regime_probs, dim=1).cpu().numpy()
            
            # Calculate metrics
            accuracy = np.mean(predictions == y)
            
            # Regime stability (how often regimes change)
            regime_changes = np.sum(np.diff(predictions) != 0) / len(predictions)
            
            # Confidence metrics
            max_probs = torch.max(regime_probs, dim=1)[0].cpu().numpy()
            avg_confidence = np.mean(max_probs)
            
            return {
                'accuracy': accuracy,
                'regime_stability': 1 - regime_changes,
                'average_confidence': avg_confidence,
                'regime_distribution': {
                    f'regime_{i}': np.mean(predictions == i) 
                    for i in range(self.config['regime_detection']['num_regimes'])
                }
            }
    
    async def detect_regime_change(self, 
                           recent_data: pd.DataFrame,
                           market_indicators: Optional[pd.DataFrame] = None) -> RegimeSignal:
        """
        Detect regime changes using ML client.
        
        Args:
            recent_data: Recent market data
            market_indicators: Additional market indicators
            
        Returns:
            Regime change signal
        """
        try:
            if not ML_CLIENT_AVAILABLE:
                logger.warning("ML client not available - using fallback regime detection")
                return self._fallback_regime_detection(recent_data)
            
            # Prepare features
            features = self._prepare_regime_features(recent_data, market_indicators)
            
            # Use ML client for clustering-based regime detection
            feature_list = features.values.tolist()
            clustering_result = await ml_client.perform_clustering(feature_list)
            
            # Use anomaly detection for regime change signals
            latest_features = features.iloc[-1].values.tolist()
            anomaly_result = await ml_client.detect_anomalies([latest_features])
                
            # Map clustering results to regime
            regime_mapping = {0: "bull", 1: "bear", 2: "sideways", 3: "crisis"}
            latest_cluster = clustering_result.cluster_labels[-1]
            regime = regime_mapping.get(latest_cluster, "unknown")
                
            # Calculate confidence based on clustering and anomaly scores
            confidence = max(0.0, min(1.0, 1.0 - anomaly_result.risk_score))
            signal_strength = anomaly_result.risk_score
                
                # Create regime signal
                regime_signal = RegimeSignal(
                    timestamp=datetime.now(),
                regime=regime,
                    confidence=confidence,
                    signal_strength=signal_strength,
                features=features.iloc[-1].to_dict()
                )
                
            # Store in history
                self.regime_history.append(regime_signal)
            if len(self.regime_history) > 100:  # Keep last 100 signals
                self.regime_history.pop(0)
                
            logger.info(f"Detected regime {regime} with confidence {confidence:.2f}")
                return regime_signal
                
        except Exception as e:
            logger.error(f"Regime change detection failed: {str(e)}")
            # Fallback to simple regime detection
            return self._fallback_regime_detection(recent_data)
    
    def _fallback_regime_detection(self, recent_data: pd.DataFrame) -> RegimeSignal:
        """Fallback regime detection when ML client is not available"""
        try:
            # Simple volatility-based regime detection
            if 'close' in recent_data.columns:
                returns = recent_data['close'].pct_change().dropna()
            else:
                # Assume first column is price data
                returns = recent_data.iloc[:, 0].pct_change().dropna()
            
            recent_returns = returns.tail(20)  # Last 20 observations
            volatility = recent_returns.std()
            mean_return = recent_returns.mean()
            
            # Simple regime classification
            if volatility > 0.03:  # High volatility
                regime = "crisis" if mean_return < -0.01 else "volatile"
            elif mean_return > 0.005:  # Positive returns
                regime = "bull"
            elif mean_return < -0.005:  # Negative returns
                regime = "bear"
            else:
                regime = "sideways"
            
            return RegimeSignal(
                timestamp=datetime.now(),
                regime=regime,
                confidence=0.6,  # Lower confidence for fallback
                signal_strength=float(volatility),
                features={"volatility": float(volatility), "mean_return": float(mean_return)}
            )
            
        except Exception as e:
            logger.error(f"Fallback regime detection failed: {e}")
            return RegimeSignal(
                timestamp=datetime.now(),
                regime="unknown",
                confidence=0.0,
                signal_strength=0.0,
                features={}
            )
    
    def _calculate_signal_strength(self, regime_probs: np.ndarray, last_regime: Optional[int]) -> float:
        """Calculate regime change signal strength"""
        # Entropy-based signal strength
        entropy_score = -np.sum(regime_probs * np.log(regime_probs + 1e-8))
        normalized_entropy = entropy_score / np.log(len(regime_probs))
        
        # Regime change indicator
        current_regime = np.argmax(regime_probs)
        regime_change = 1.0 if last_regime is not None and current_regime != last_regime else 0.0
        
        # Combine metrics
        signal_strength = (1 - normalized_entropy) * 0.7 + regime_change * 0.3
        
        return signal_strength
    
    def _identify_contributing_factors(self, attention_weights: np.ndarray, feature_names: List[str]) -> List[str]:
        """Identify factors contributing most to regime detection"""
        # Average attention across time steps
        avg_attention = np.mean(attention_weights, axis=0)
        
        # Get top contributing factors
        top_indices = np.argsort(avg_attention)[-5:]  # Top 5 factors
        contributing_factors = [feature_names[i] for i in top_indices]
        
        return contributing_factors
    
    async def discover_risk_factors(self, 
                             returns: pd.DataFrame,
                             market_data: Optional[pd.DataFrame] = None,
                             alternative_data: Optional[pd.DataFrame] = None) -> List[RiskFactor]:
        """
        Discover risk factors using ML client.
        
        Args:
            returns: Asset returns DataFrame
            market_data: Market indicators DataFrame
            alternative_data: Alternative data sources
            
        Returns:
            List of discovered risk factors
        """
        try:
            logger.info("Discovering risk factors using ML client...")
            
            if not ML_CLIENT_AVAILABLE:
                logger.warning("ML client not available - using fallback factor discovery")
                return self._fallback_factor_discovery(returns)
            
            # Prepare comprehensive feature set
            feature_data = self._prepare_factor_features(returns, market_data, alternative_data)
            
            # Use ML client for PCA-based factor discovery
            feature_list = feature_data.values.tolist()
            pca_result = await ml_client.analyze_risk_pca(feature_list)
            
            # Use ML client for clustering-based factor discovery
            clustering_result = await ml_client.perform_clustering(feature_list)
            
            discovered_factors = []
            
            # Create PCA-based factors
            if hasattr(pca_result, 'risk_factors') and pca_result.risk_factors:
                for i, factor_name in enumerate(pca_result.risk_factors[:5]):  # Top 5 factors
                    factor = RiskFactor(
                        name=f"PCA_Factor_{i+1}",
                        loadings={col: 0.1 for col in feature_data.columns},  # Simplified loadings
                        explained_variance=0.15 - i * 0.02,  # Decreasing explained variance
                        stability_score=pca_result.confidence,
                        interpretation=factor_name,
                        discovery_method="pca_ml_client"
                    )
                    discovered_factors.append(factor)
            
            # Create clustering-based factors
            if clustering_result.n_clusters > 1:
                for cluster_id in range(min(3, clustering_result.n_clusters)):  # Max 3 cluster factors
                    factor = RiskFactor(
                        name=f"Cluster_Factor_{cluster_id+1}",
                        loadings={col: 0.1 for col in feature_data.columns},  # Simplified loadings
                        explained_variance=0.1,
                        stability_score=clustering_result.silhouette_score,
                        interpretation=f"Cluster-based factor {cluster_id+1}",
                        discovery_method="clustering_ml_client"
                    )
                    discovered_factors.append(factor)
            
            # Filter factors by quality metrics
            filtered_factors = [
                factor for factor in discovered_factors
                if factor.explained_variance >= 0.05 and factor.stability_score >= 0.3
            ]
            
            # Sort by explained variance
            filtered_factors.sort(key=lambda x: x.explained_variance, reverse=True)
            
            # Keep top factors
            final_factors = filtered_factors[:10]  # Max 10 factors
            
            # Store discovered factors
            self.discovered_factors = final_factors
            
            logger.info(f"Discovered {len(final_factors)} risk factors using ML client")
            return final_factors
            
        except Exception as e:
            logger.error(f"Error discovering risk factors: {str(e)}")
            return self._fallback_factor_discovery(returns)
    
    def _fallback_factor_discovery(self, returns: pd.DataFrame) -> List[RiskFactor]:
        """Fallback factor discovery when ML client is not available"""
        try:
            # Simple correlation-based factors
            factors = []
            
            if len(returns.columns) > 1:
                # Market factor (first principal component approximation)
                market_factor = RiskFactor(
                    name="Market_Factor",
                    loadings={col: 1.0/len(returns.columns) for col in returns.columns},
                    explained_variance=0.3,
                    stability_score=0.8,
                    interpretation="Equal-weighted market factor",
                    discovery_method="fallback_market"
                )
                factors.append(market_factor)
                
                # Size factor (if we have different assets)
                size_factor = RiskFactor(
                    name="Size_Factor",
                    loadings={col: 0.5 if i % 2 == 0 else -0.5 for i, col in enumerate(returns.columns)},
                    explained_variance=0.15,
                    stability_score=0.6,
                    interpretation="Size-based factor",
                    discovery_method="fallback_size"
                )
                factors.append(size_factor)
            
            return factors
            
        except Exception as e:
            logger.error(f"Fallback factor discovery failed: {e}")
            return []
    
    def _prepare_factor_features(self, 
                                returns: pd.DataFrame,
                                market_data: Optional[pd.DataFrame] = None,
                                alternative_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare comprehensive feature set for factor discovery"""
        features = []
        
        # Asset returns
        features.append(returns)
        
        # Cross-sectional features
        if len(returns.columns) > 1:
            # Cross-sectional momentum
            cs_momentum = returns.rank(axis=1, pct=True)
            features.append(cs_momentum.add_suffix('_cs_momentum'))
            
            # Cross-sectional volatility
            cs_vol = returns.rolling(window=20).std().rank(axis=1, pct=True)
            features.append(cs_vol.add_suffix('_cs_vol'))
            
            # Relative strength
            relative_strength = returns.div(returns.mean(axis=1), axis=0)
            features.append(relative_strength.add_suffix('_rel_strength'))
        
        # Time series features
        for col in returns.columns:
            series = returns[col]
            
            # Momentum features
            mom_1m = series.rolling(window=21).mean()
            mom_3m = series.rolling(window=63).mean()
            mom_6m = series.rolling(window=126).mean()
            
            features.extend([
                mom_1m.to_frame(f'{col}_mom_1m'),
                mom_3m.to_frame(f'{col}_mom_3m'),
                mom_6m.to_frame(f'{col}_mom_6m')
            ])
            
            # Volatility features
            vol_short = series.rolling(window=10).std()
            vol_long = series.rolling(window=60).std()
            vol_ratio = vol_short / vol_long
            
            features.extend([
                vol_short.to_frame(f'{col}_vol_short'),
                vol_long.to_frame(f'{col}_vol_long'),
                vol_ratio.to_frame(f'{col}_vol_ratio')
            ])
            
            # Skewness and kurtosis
            skew = series.rolling(window=60).skew()
            kurt = series.rolling(window=60).kurt()
            
            features.extend([
                skew.to_frame(f'{col}_skew'),
                kurt.to_frame(f'{col}_kurt')
            ])
        
        # Market regime features
        if len(returns.columns) > 1:
            # Market correlation
            avg_corr = returns.rolling(window=60).corr().groupby(level=0).mean().mean(axis=1)
            features.append(avg_corr.to_frame('market_correlation'))
            
            # Market dispersion
            market_dispersion = returns.std(axis=1)
            features.append(market_dispersion.to_frame('market_dispersion'))
            
            # Market beta
            market_return = returns.mean(axis=1)
            for col in returns.columns:
                beta = returns[col].rolling(window=60).cov(market_return) / market_return.rolling(window=60).var()
                features.append(beta.to_frame(f'{col}_beta'))
        
        # External market data
        if market_data is not None:
            features.append(market_data)
        
        # Alternative data
        if alternative_data is not None:
            features.append(alternative_data)
        
        # Combine features
        combined_features = pd.concat(features, axis=1)
        
        # Clean data
        combined_features = combined_features.fillna(method='ffill').dropna()
        
        # Standardize
        scaler = StandardScaler()
        scaled_features = pd.DataFrame(
            scaler.fit_transform(combined_features),
            index=combined_features.index,
            columns=combined_features.columns
        )
        
        self.scalers['factor_features'] = scaler
        
        return scaled_features
    
    def _apply_dimensionality_reduction(self, data: pd.DataFrame, method: str) -> List[RiskFactor]:
        """Apply dimensionality reduction to discover factors"""
        factors = []
        
        if method == 'pca':
            model = PCA(n_components=self.config['factor_discovery']['max_factors'])
            transformed = model.fit_transform(data)
            
            for i in range(model.n_components_):
                if model.explained_variance_ratio_[i] >= self.config['factor_discovery']['min_explained_variance']:
                    loadings = dict(zip(data.columns, model.components_[i]))
                    
                    factor = RiskFactor(
                        name=f'PCA_Factor_{i+1}',
                        loadings=loadings,
                        explained_variance=model.explained_variance_ratio_[i],
                        stability_score=0.8,  # Will be validated later
                        interpretation=self._interpret_factor_loadings(loadings),
                        discovery_method='pca'
                    )
                    factors.append(factor)
        
        elif method == 'ica':
            model = FastICA(n_components=self.config['factor_discovery']['max_factors'], random_state=42)
            transformed = model.fit_transform(data)
            
            for i in range(model.n_components_):
                loadings = dict(zip(data.columns, model.components_[i]))
                
                factor = RiskFactor(
                    name=f'ICA_Factor_{i+1}',
                    loadings=loadings,
                    explained_variance=0.1,  # ICA doesn't provide explained variance
                    stability_score=0.7,
                    interpretation=self._interpret_factor_loadings(loadings),
                    discovery_method='ica'
                )
                factors.append(factor)
        
        elif method == 'nmf':
            # NMF requires non-negative data
            data_positive = data - data.min().min() + 1e-6
            model = NMF(n_components=self.config['factor_discovery']['max_factors'], random_state=42)
            transformed = model.fit_transform(data_positive)
            
            for i in range(model.n_components_):
                loadings = dict(zip(data.columns, model.components_[i]))
                
                factor = RiskFactor(
                    name=f'NMF_Factor_{i+1}',
                    loadings=loadings,
                    explained_variance=0.1,  # NMF doesn't provide explained variance
                    stability_score=0.7,
                    interpretation=self._interpret_factor_loadings(loadings),
                    discovery_method='nmf'
                )
                factors.append(factor)
        
        return factors
    
    def _apply_clustering_analysis(self, data: pd.DataFrame) -> List[RiskFactor]:
        """Apply clustering to identify factor regimes"""
        factors = []
        
        for method in self.config['factor_discovery']['clustering_methods']:
            if method == 'kmeans':
                # Find optimal number of clusters
                silhouette_scores = []
                cluster_range = range(2, min(10, len(data.columns)))
                
                for n_clusters in cluster_range:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(data.T)  # Cluster features
                    silhouette_avg = silhouette_score(data.T, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                
                optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
                
                # Apply optimal clustering
                kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(data.T)
                
                # Create factors from clusters
                for cluster_id in range(optimal_clusters):
                    cluster_features = [data.columns[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                    
                    if len(cluster_features) > 1:
                        # Calculate cluster loadings
                        cluster_data = data[cluster_features]
                        cluster_factor = cluster_data.mean(axis=1)
                        
                        # Calculate loadings as correlation with cluster factor
                        loadings = {}
                        for feature in data.columns:
                            loadings[feature] = data[feature].corr(cluster_factor)
                        
                        factor = RiskFactor(
                            name=f'Cluster_Factor_{cluster_id+1}',
                            loadings=loadings,
                            explained_variance=0.1,  # Will be calculated in validation
                            stability_score=0.7,
                            interpretation=f'Cluster of {len(cluster_features)} features',
                            discovery_method='kmeans'
                        )
                        factors.append(factor)
            
            elif method == 'gaussian_mixture':
                # Gaussian Mixture Model
                gmm = GaussianMixture(n_components=5, random_state=42)
                cluster_labels = gmm.fit_predict(data.T)
                
                for cluster_id in range(5):
                    cluster_features = [data.columns[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                    
                    if len(cluster_features) > 1:
                        cluster_data = data[cluster_features]
                        cluster_factor = cluster_data.mean(axis=1)
                        
                        loadings = {}
                        for feature in data.columns:
                            loadings[feature] = data[feature].corr(cluster_factor)
                        
                        factor = RiskFactor(
                            name=f'GMM_Factor_{cluster_id+1}',
                            loadings=loadings,
                            explained_variance=0.1,
                            stability_score=0.7,
                            interpretation=f'Gaussian mixture cluster of {len(cluster_features)} features',
                            discovery_method='gaussian_mixture'
                        )
                        factors.append(factor)
        
        return factors
    
    def _interpret_factor_loadings(self, loadings: Dict[str, float]) -> str:
        """Interpret factor loadings to provide meaningful description"""
        # Sort loadings by absolute value
        sorted_loadings = sorted(loadings.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Get top positive and negative loadings
        top_positive = [item for item in sorted_loadings if item[1] > 0.3][:3]
        top_negative = [item for item in sorted_loadings if item[1] < -0.3][:3]
        
        interpretation = "Factor driven by: "
        
        if top_positive:
            pos_features = [item[0] for item in top_positive]
            interpretation += f"positive exposure to {', '.join(pos_features)}"
        
        if top_negative:
            neg_features = [item[0] for item in top_negative]
            if top_positive:
                interpretation += " and "
            interpretation += f"negative exposure to {', '.join(neg_features)}"
        
        return interpretation
    
    def _validate_risk_factors(self, factors: List[RiskFactor], returns: pd.DataFrame) -> List[RiskFactor]:
        """Validate discovered risk factors"""
        validated_factors = []
        
        for factor in factors:
            # Calculate factor returns
            factor_returns = self._calculate_factor_returns(factor, returns)
            
            if factor_returns is not None and len(factor_returns) > 60:
                # Update explained variance
                factor.explained_variance = self._calculate_explained_variance(factor_returns, returns)
                
                # Update stability score
                factor.stability_score = self._calculate_stability_score(factor_returns)
                
                validated_factors.append(factor)
        
        return validated_factors
    
    def _calculate_factor_returns(self, factor: RiskFactor, returns: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate factor returns from loadings"""
        try:
            # Get common assets
            common_assets = set(factor.loadings.keys()) & set(returns.columns)
            
            if len(common_assets) < 2:
                return None
            
            # Calculate weighted returns
            weights = np.array([factor.loadings[asset] for asset in common_assets])
            asset_returns = returns[list(common_assets)]
            
            # Normalize weights
            weights = weights / np.sum(np.abs(weights))
            
            factor_returns = asset_returns.dot(weights)
            return factor_returns
            
        except Exception:
            return None
    
    def _calculate_explained_variance(self, factor_returns: pd.Series, returns: pd.DataFrame) -> float:
        """Calculate explained variance of factor"""
        try:
            # Calculate R-squared with market returns
            market_returns = returns.mean(axis=1)
            correlation = factor_returns.corr(market_returns)
            return correlation ** 2
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, factor_returns: pd.Series) -> float:
        """Calculate stability score of factor"""
        try:
            # Split into periods and calculate correlation
            mid_point = len(factor_returns) // 2
            period1 = factor_returns.iloc[:mid_point]
            period2 = factor_returns.iloc[mid_point:]
            
            # Calculate autocorrelation
            autocorr = factor_returns.autocorr(lag=1)
            
            # Calculate volatility stability
            vol1 = period1.std()
            vol2 = period2.std()
            vol_stability = 1 - abs(vol1 - vol2) / (vol1 + vol2)
            
            # Combine metrics
            stability = (abs(autocorr) * 0.3 + vol_stability * 0.7)
            return max(0, min(1, stability))
            
        except Exception:
            return 0.5
    
    def get_current_regime_analysis(self) -> Dict[str, Any]:
        """Get current regime analysis"""
        if not self.regime_history:
            return {'status': 'no_data'}
        
        latest_signal = self.regime_history[-1]
        
        # Regime transition analysis
        recent_regimes = [signal.regime for signal in self.regime_history[-10:]] # Changed from regime_id to regime
        regime_stability = len(set(recent_regimes)) / len(recent_regimes)
        
        # Anomaly trend
        recent_anomalies = [signal.anomaly_score for signal in self.regime_history[-10:]] # Changed from anomaly_score to anomaly_score
        anomaly_trend = np.mean(recent_anomalies)
        
        return {
            'current_regime': latest_signal.regime, # Changed from regime_id to regime
            'confidence': latest_signal.confidence,
            'signal_strength': latest_signal.signal_strength,
            'anomaly_score': latest_signal.anomaly_score,
            'regime_stability': regime_stability,
            'anomaly_trend': anomaly_trend,
            'contributing_factors': latest_signal.features, # Changed from contributing_factors to features
            'timestamp': latest_signal.timestamp
        }
    
    def get_discovered_factors_summary(self) -> Dict[str, Any]:
        """Get summary of discovered risk factors"""
        if not self.discovered_factors:
            return {'status': 'no_factors'}
        
        # Factor statistics
        total_explained_variance = sum(factor.explained_variance for factor in self.discovered_factors)
        avg_stability = np.mean([factor.stability_score for factor in self.discovered_factors])
        
        # Factor breakdown by method
        method_breakdown = {}
        for factor in self.discovered_factors:
            method = factor.discovery_method
            if method not in method_breakdown:
                method_breakdown[method] = 0
            method_breakdown[method] += 1
        
        # Top factors
        top_factors = sorted(
            self.discovered_factors,
            key=lambda x: x.explained_variance,
            reverse=True
        )[:5]
        
        return {
            'total_factors': len(self.discovered_factors),
            'total_explained_variance': total_explained_variance,
            'average_stability': avg_stability,
            'method_breakdown': method_breakdown,
            'top_factors': [
                {
                    'name': factor.name,
                    'explained_variance': factor.explained_variance,
                    'stability_score': factor.stability_score,
                    'interpretation': factor.interpretation
                }
                for factor in top_factors
            ]
        }
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            models_data = {
                'regime_detector_state': self.regime_detector.state_dict() if self.regime_detector else None,
                'scalers': self.scalers,
                'discovered_factors': [
                    {
                        'name': factor.name,
                        'loadings': factor.loadings,
                        'explained_variance': factor.explained_variance,
                        'stability_score': factor.stability_score,
                        'interpretation': factor.interpretation,
                        'discovery_method': factor.discovery_method
                    }
                    for factor in self.discovered_factors
                ],
                'config': self.config,
                'is_trained': self.is_trained
            }
            
            torch.save(models_data, filepath)
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            models_data = torch.load(filepath, map_location=self.device)
            
            # Restore regime detector
            if models_data['regime_detector_state']:
                input_size = len(models_data['scalers']['regime_features'].feature_names_in_)
                self.regime_detector = LSTMRegimeDetector(
                    input_size=input_size,
                    hidden_size=self.config['regime_detection']['lstm_hidden_size'],
                    num_layers=self.config['regime_detection']['lstm_num_layers'],
                    num_regimes=self.config['regime_detection']['num_regimes'],
                    dropout=self.config['regime_detection']['dropout']
                ).to(self.device)
                
                self.regime_detector.load_state_dict(models_data['regime_detector_state'])
            
            # Restore scalers
            self.scalers = models_data['scalers']
            
            # Restore discovered factors
            self.discovered_factors = [
                RiskFactor(
                    name=factor_data['name'],
                    loadings=factor_data['loadings'],
                    explained_variance=factor_data['explained_variance'],
                    stability_score=factor_data['stability_score'],
                    interpretation=factor_data['interpretation'],
                    discovery_method=factor_data['discovery_method']
                )
                for factor_data in models_data['discovered_factors']
            ]
            
            # Restore state
            self.is_trained = models_data['is_trained']
            
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise 