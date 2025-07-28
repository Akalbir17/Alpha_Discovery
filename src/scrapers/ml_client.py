"""
ML Client for Remote Inference
Handles HTTP requests to the MCP server for ML model inference
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import aiohttp
import structlog
from dataclasses import dataclass, field

logger = structlog.get_logger(__name__)

# ML Client availability flag
ML_CLIENT_AVAILABLE = True

@dataclass
class SentimentResult:
    label: str
    score: float
    confidence: float

@dataclass
class EmotionResult:
    emotion: str
    score: float
    all_emotions: Dict[str, float]

@dataclass
class NERResult:
    entities: List[Dict[str, Any]]
    tickers: List[str]
    confidence_scores: List[float] = field(default_factory=list)
    entity_types: List[str] = field(default_factory=list)

@dataclass
class RegimeDetectionResult:
    regime: str  # normal, stressed, volatile
    anomaly_score: float
    confidence: float

@dataclass
class ImbalancePredictionResult:
    predicted_imbalance: float
    confidence: float
    feature_importance: Dict[str, float]

@dataclass
class LambdaPredictionResult:
    predicted_lambda: float
    confidence: float
    market_impact_estimate: float

@dataclass
class FlowClassificationResult:
    flow_type: str  # institutional, retail, mixed
    probability: float
    flow_direction: str  # buy, sell, neutral
    confidence: float

@dataclass
class FinancialSentimentResult:
    sentiment_label: str  # positive, negative, neutral
    sentiment_score: float  # -1 to 1
    confidence: float
    individual_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class VisionAnalysisResult:
    analysis_type: str
    activity_level: float
    objects_detected: int
    economic_signal: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestPredictionResult:
    prediction: float
    confidence: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_type: str = "unknown"

@dataclass
class RiskAnalysisResult:
    risk_score: float
    risk_factors: List[str]
    confidence: float
    risk_breakdown: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ClusteringResult:
    cluster_labels: List[int]
    n_clusters: int
    silhouette_score: float
    cluster_centers: List[List[float]] = field(default_factory=list)
    cluster_info: Dict[str, Any] = field(default_factory=dict)

class MLClient:
    """HTTP client for ML model inference via MCP server"""
    
    def __init__(self, mcp_host: str = "localhost", mcp_port: int = 8002):
        self.base_url = f"http://{mcp_host}:{mcp_port}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure session is available"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
    
    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/sentiment-analysis",
                json={"text": text}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    sentiment_data = data["sentiment"]
                    return SentimentResult(
                        label=sentiment_data["label"],
                        score=sentiment_data["score"],
                        confidence=sentiment_data["confidence"]
                    )
                else:
                    logger.error(f"Sentiment analysis failed: {response.status}")
                    return SentimentResult(label="neutral", score=0.5, confidence=0.0)
                    
        except asyncio.TimeoutError:
            logger.error("Sentiment analysis request timed out")
            return SentimentResult(label="neutral", score=0.5, confidence=0.0)
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return SentimentResult(label="neutral", score=0.5, confidence=0.0)
    
    async def analyze_emotion(self, text: str) -> EmotionResult:
        """Analyze emotions via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/emotion-analysis",
                json={"text": text}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    emotion_data = data["emotion"]
                    return EmotionResult(
                        emotion=emotion_data["emotion"],
                        score=emotion_data["score"],
                        all_emotions=emotion_data["all_emotions"]
                    )
                else:
                    logger.error(f"Emotion analysis failed: {response.status}")
                    return EmotionResult(
                        emotion="neutral", 
                        score=0.5, 
                        all_emotions={"neutral": 0.5}
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Emotion analysis request timed out")
            return EmotionResult(
                emotion="neutral", 
                score=0.0, 
                all_emotions={"neutral": 0.0}
            )
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return EmotionResult(
                emotion="neutral", 
                score=0.0, 
                all_emotions={"neutral": 0.0}
            )
    
    async def extract_entities(self, text: str) -> NERResult:
        """Extract named entities via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/ner",
                json={"text": text}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    entities_data = data["entities"]
                    return NERResult(
                        entities=entities_data["entities"],
                        tickers=entities_data["tickers"]
                    )
                else:
                    logger.error(f"NER extraction failed: {response.status}")
                    return NERResult(entities=[], tickers=[])
                    
        except asyncio.TimeoutError:
            logger.error("NER extraction request timed out")
            return NERResult(entities=[], tickers=[])
        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
            return NERResult(entities=[], tickers=[])
    
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/similarity",
                json={"text1": text1, "text2": text2}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["similarity"]
                else:
                    logger.error(f"Similarity computation failed: {response.status}")
                    return 0.0
                    
        except asyncio.TimeoutError:
            logger.error("Similarity computation request timed out")
            return 0.0
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    async def check_health(self) -> bool:
        """Check if MCP server is healthy"""
        await self._ensure_session()
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def detect_regime(self, features: List[float]) -> RegimeDetectionResult:
        """Detect market regime via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/regime-detection",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    regime_data = data["regime"]
                    return RegimeDetectionResult(
                        regime=regime_data["regime"],
                        anomaly_score=regime_data["anomaly_score"],
                        confidence=regime_data["confidence"]
                    )
                else:
                    logger.error(f"Regime detection failed: {response.status}")
                    return RegimeDetectionResult(
                        regime="normal",
                        anomaly_score=0.0,
                        confidence=0.0
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Regime detection request timed out")
            return RegimeDetectionResult(
                regime="normal",
                anomaly_score=0.0,
                confidence=0.0
            )
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return RegimeDetectionResult(
                regime="normal",
                anomaly_score=0.0,
                confidence=0.0
            )
    
    async def predict_imbalance(self, features: List[float]) -> ImbalancePredictionResult:
        """Predict order flow imbalance via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/imbalance-prediction",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    imbalance_data = data["imbalance"]
                    return ImbalancePredictionResult(
                        predicted_imbalance=imbalance_data["predicted_imbalance"],
                        confidence=imbalance_data["confidence"],
                        feature_importance=imbalance_data["feature_importance"]
                    )
                else:
                    logger.error(f"Imbalance prediction failed: {response.status}")
                    return ImbalancePredictionResult(
                        predicted_imbalance=0.0,
                        confidence=0.0,
                        feature_importance={}
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Imbalance prediction request timed out")
            return ImbalancePredictionResult(
                predicted_imbalance=0.0,
                confidence=0.0,
                feature_importance={}
            )
        except Exception as e:
            logger.error(f"Error in imbalance prediction: {e}")
            return ImbalancePredictionResult(
                predicted_imbalance=0.0,
                confidence=0.0,
                feature_importance={}
            )
    
    async def predict_lambda(self, features: List[float]) -> LambdaPredictionResult:
        """Predict Kyle's lambda via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/lambda-prediction",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    lambda_data = data["lambda"]
                    return LambdaPredictionResult(
                        predicted_lambda=lambda_data["predicted_lambda"],
                        confidence=lambda_data["confidence"],
                        market_impact_estimate=lambda_data["market_impact_estimate"]
                    )
                else:
                    logger.error(f"Lambda prediction failed: {response.status}")
                    return LambdaPredictionResult(
                        predicted_lambda=0.0,
                        confidence=0.0,
                        market_impact_estimate=0.0
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Lambda prediction request timed out")
            return LambdaPredictionResult(
                predicted_lambda=0.0,
                confidence=0.0,
                market_impact_estimate=0.0
            )
        except Exception as e:
            logger.error(f"Error in lambda prediction: {e}")
            return LambdaPredictionResult(
                predicted_lambda=0.0,
                confidence=0.0,
                market_impact_estimate=0.0
            )
    
    async def classify_flow(self, features: List[float]) -> FlowClassificationResult:
        """Classify order flow type via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/flow-classification",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    flow_data = data["flow"]
                    return FlowClassificationResult(
                        flow_type=flow_data["flow_type"],
                        probability=flow_data["probability"],
                        flow_direction=flow_data["flow_direction"],
                        confidence=flow_data["confidence"]
                    )
                else:
                    logger.error(f"Flow classification failed: {response.status}")
                    return FlowClassificationResult(
                        flow_type="mixed",
                        probability=0.5,
                        flow_direction="neutral",
                        confidence=0.0
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Flow classification request timed out")
            return FlowClassificationResult(
                flow_type="mixed",
                probability=0.5,
                flow_direction="neutral",
                confidence=0.0
            )
        except Exception as e:
            logger.error(f"Error in flow classification: {e}")
            return FlowClassificationResult(
                flow_type="mixed",
                probability=0.5,
                flow_direction="neutral",
                confidence=0.0
            )
    
    async def get_ml_health(self) -> Dict[str, Any]:
        """Get detailed ML service health status"""
        await self._ensure_session()
        
        try:
            async with self.session.get(f"{self.base_url}/ml-health") as response:
                if response.status == 200:
                    data = await response.json()
                    return data["ml_status"]
                else:
                    return {"error": f"Health check failed: {response.status}"}
        except Exception as e:
            logger.error(f"ML health check failed: {e}")
            return {"error": str(e)}
    
    # State-of-the-art ML methods (2025 upgrades)
    
    async def extract_entities_gliner(self, text: str, entity_types: List[str] = None) -> NERResult:
        """Extract entities using revolutionary GLiNER via MCP server"""
        await self._ensure_session()
        
        try:
            payload = {"text": text}
            if entity_types:
                payload["entity_types"] = entity_types
                
            async with self.session.post(
                f"{self.base_url}/gliner-ner",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    entities_data = data["entities"]
                    return NERResult(
                        entities=entities_data["entities"],
                        tickers=entities_data["tickers"],
                        confidence_scores=entities_data.get("confidence_scores", []),
                        entity_types=entities_data.get("entity_types", [])
                    )
                else:
                    logger.error(f"GLiNER NER failed: {response.status}")
                    return NERResult(entities=[], tickers=[])
                    
        except asyncio.TimeoutError:
            logger.error("GLiNER NER request timed out")
            return NERResult(entities=[], tickers=[])
        except Exception as e:
            logger.error(f"Error in GLiNER NER: {e}")
            return NERResult(entities=[], tickers=[])
    
    async def predict_imbalance_xgb(self, features: List[float]) -> ImbalancePredictionResult:
        """Predict order flow imbalance using XGBoost via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/xgboost-imbalance",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    imbalance_data = data["imbalance"]
                    return ImbalancePredictionResult(
                        predicted_imbalance=imbalance_data["predicted_imbalance"],
                        confidence=imbalance_data["confidence"],
                        feature_importance=imbalance_data["feature_importance"]
                    )
                else:
                    logger.error(f"XGBoost imbalance prediction failed: {response.status}")
                    return ImbalancePredictionResult(
                        predicted_imbalance=0.0,
                        confidence=0.0,
                        feature_importance={}
                    )
                    
        except asyncio.TimeoutError:
            logger.error("XGBoost imbalance prediction request timed out")
            return ImbalancePredictionResult(
                predicted_imbalance=0.0,
                confidence=0.0,
                feature_importance={}
            )
        except Exception as e:
            logger.error(f"Error in XGBoost imbalance prediction: {e}")
            return ImbalancePredictionResult(
                predicted_imbalance=0.0,
                confidence=0.0,
                feature_importance={}
            )
    
    async def predict_lambda_lgb(self, features: List[float]) -> LambdaPredictionResult:
        """Predict Kyle's lambda using LightGBM via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/lightgbm-lambda",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    lambda_data = data["lambda"]
                    return LambdaPredictionResult(
                        predicted_lambda=lambda_data["predicted_lambda"],
                        confidence=lambda_data["confidence"],
                        market_impact_estimate=lambda_data["market_impact_estimate"]
                    )
                else:
                    logger.error(f"LightGBM lambda prediction failed: {response.status}")
                    return LambdaPredictionResult(
                        predicted_lambda=0.0,
                        confidence=0.0,
                        market_impact_estimate=0.0
                    )
                    
        except asyncio.TimeoutError:
            logger.error("LightGBM lambda prediction request timed out")
            return LambdaPredictionResult(
                predicted_lambda=0.0,
                confidence=0.0,
                market_impact_estimate=0.0
            )
        except Exception as e:
            logger.error(f"Error in LightGBM lambda prediction: {e}")
            return LambdaPredictionResult(
                predicted_lambda=0.0,
                confidence=0.0,
                market_impact_estimate=0.0
            )
    
    async def classify_flow_catboost(self, features: List[float]) -> FlowClassificationResult:
        """Classify order flow type using CatBoost via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/catboost-flow",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    flow_data = data["flow"]
                    return FlowClassificationResult(
                        flow_type=flow_data["flow_type"],
                        probability=flow_data["probability"],
                        flow_direction=flow_data["flow_direction"],
                        confidence=flow_data["confidence"]
                    )
                else:
                    logger.error(f"CatBoost flow classification failed: {response.status}")
                    return FlowClassificationResult(
                        flow_type="mixed",
                        probability=0.5,
                        flow_direction="neutral",
                        confidence=0.0
                    )
                    
        except asyncio.TimeoutError:
            logger.error("CatBoost flow classification request timed out")
            return FlowClassificationResult(
                flow_type="mixed",
                probability=0.5,
                flow_direction="neutral",
                confidence=0.0
            )
        except Exception as e:
            logger.error(f"Error in CatBoost flow classification: {e}")
            return FlowClassificationResult(
                flow_type="mixed",
                probability=0.5,
                flow_direction="neutral",
                confidence=0.0
            )
    
    async def detect_regime_modern(self, features: List[float]) -> RegimeDetectionResult:
        """Detect market regime using modern methods via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/modern-regime",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    regime_data = data["regime"]
                    return RegimeDetectionResult(
                        regime=regime_data["regime"],
                        anomaly_score=regime_data["anomaly_score"],
                        confidence=regime_data["confidence"]
                    )
                else:
                    logger.error(f"Modern regime detection failed: {response.status}")
                    return RegimeDetectionResult(
                        regime="normal",
                        anomaly_score=0.0,
                        confidence=0.0
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Modern regime detection request timed out")
            return RegimeDetectionResult(
                regime="normal",
                anomaly_score=0.0,
                confidence=0.0
            )
        except Exception as e:
            logger.error(f"Error in modern regime detection: {e}")
            return RegimeDetectionResult(
                regime="normal",
                anomaly_score=0.0,
                confidence=0.0
            )
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

    # Alternative Data Agent methods (Phase 3)
    
    async def analyze_financial_sentiment(self, text: str) -> FinancialSentimentResult:
        """Analyze financial sentiment using enhanced FinBERT via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/financial-sentiment",
                json={"text": text}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    sentiment_data = data["sentiment"]
                    return FinancialSentimentResult(
                        sentiment_label=sentiment_data["sentiment_label"],
                        sentiment_score=sentiment_data["sentiment_score"],
                        confidence=sentiment_data["confidence"],
                        individual_scores=sentiment_data["individual_scores"]
                    )
                else:
                    logger.error(f"Financial sentiment analysis failed: {response.status}")
                    return FinancialSentimentResult(
                        sentiment_label="neutral",
                        sentiment_score=0.0,
                        confidence=0.0,
                        individual_scores={"positive": 0.33, "negative": 0.33, "neutral": 0.34}
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Financial sentiment analysis request timed out")
            return FinancialSentimentResult(
                sentiment_label="neutral",
                sentiment_score=0.0,
                confidence=0.0,
                individual_scores={"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            )
        except Exception as e:
            logger.error(f"Error in financial sentiment analysis: {e}")
            return FinancialSentimentResult(
                sentiment_label="neutral",
                sentiment_score=0.0,
                confidence=0.0,
                individual_scores={"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            )
    
    async def analyze_vision(self, image_path: str, analysis_type: str = "economic_activity") -> VisionAnalysisResult:
        """Analyze images for economic indicators via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/vision-analysis",
                json={"image_path": image_path, "analysis_type": analysis_type}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    vision_data = data["vision"]
                    return VisionAnalysisResult(
                        analysis_type=vision_data["analysis_type"],
                        activity_level=vision_data["activity_level"],
                        objects_detected=vision_data["objects_detected"],
                        economic_signal=vision_data["economic_signal"],
                        confidence=vision_data["confidence"],
                        metadata=vision_data.get("metadata", {})
                    )
                else:
                    logger.error(f"Vision analysis failed: {response.status}")
                    return VisionAnalysisResult(
                        analysis_type=analysis_type,
                        activity_level=0.0,
                        objects_detected=0,
                        economic_signal="error",
                        confidence=0.0,
                        metadata={"error": f"HTTP {response.status}"}
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Vision analysis request timed out")
            return VisionAnalysisResult(
                analysis_type=analysis_type,
                activity_level=0.0,
                objects_detected=0,
                economic_signal="timeout",
                confidence=0.0,
                metadata={"error": "Request timed out"}
            )
        except Exception as e:
            logger.error(f"Error in vision analysis: {e}")
            return VisionAnalysisResult(
                analysis_type=analysis_type,
                activity_level=0.0,
                objects_detected=0,
                economic_signal="error",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    # Advanced ML methods (Phase 4)
    
    async def predict_with_mlp(self, features: List[float], target_name: str = "risk") -> BacktestPredictionResult:
        """Predict using MLP Regressor via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/mlp-prediction",
                json={"features": features, "target_name": target_name}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    prediction_data = data["prediction"]
                    return BacktestPredictionResult(
                        prediction=prediction_data["prediction"],
                        confidence=prediction_data["confidence"],
                        feature_importance=prediction_data["feature_importance"],
                        model_type=prediction_data["model_type"]
                    )
                else:
                    logger.error(f"MLP prediction failed: {response.status}")
                    return BacktestPredictionResult(
                        prediction=0.0,
                        confidence=0.0,
                        feature_importance={},
                        model_type="MLPRegressor"
                    )
                    
        except asyncio.TimeoutError:
            logger.error("MLP prediction request timed out")
            return BacktestPredictionResult(
                prediction=0.0,
                confidence=0.0,
                feature_importance={},
                model_type="MLPRegressor"
            )
        except Exception as e:
            logger.error(f"Error in MLP prediction: {e}")
            return BacktestPredictionResult(
                prediction=0.0,
                confidence=0.0,
                feature_importance={},
                model_type="MLPRegressor"
            )
    
    async def predict_with_gradient_boosting(self, features: List[float]) -> BacktestPredictionResult:
        """Predict using Gradient Boosting via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/gradient-boosting",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    prediction_data = data["prediction"]
                    return BacktestPredictionResult(
                        prediction=prediction_data["prediction"],
                        confidence=prediction_data["confidence"],
                        feature_importance=prediction_data["feature_importance"],
                        model_type=prediction_data["model_type"]
                    )
                else:
                    logger.error(f"Gradient Boosting prediction failed: {response.status}")
                    return BacktestPredictionResult(
                        prediction=0.0,
                        confidence=0.0,
                        feature_importance={},
                        model_type="GradientBoostingRegressor"
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Gradient Boosting prediction request timed out")
            return BacktestPredictionResult(
                prediction=0.0,
                confidence=0.0,
                feature_importance={},
                model_type="GradientBoostingRegressor"
            )
        except Exception as e:
            logger.error(f"Error in Gradient Boosting prediction: {e}")
            return BacktestPredictionResult(
                prediction=0.0,
                confidence=0.0,
                feature_importance={},
                model_type="GradientBoostingRegressor"
            )
    
    async def analyze_risk_pca(self, features: List[List[float]]) -> RiskAnalysisResult:
        """Analyze risk using PCA via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/risk-pca",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    risk_data = data["risk_analysis"]
                    return RiskAnalysisResult(
                        risk_score=risk_data["risk_score"],
                        risk_factors=risk_data["risk_factors"],
                        confidence=risk_data["confidence"],
                        risk_breakdown=risk_data["risk_breakdown"],
                        recommendations=risk_data["recommendations"]
                    )
                else:
                    logger.error(f"PCA risk analysis failed: {response.status}")
                    return RiskAnalysisResult(
                        risk_score=0.5,
                        risk_factors=["unknown"],
                        confidence=0.0,
                        risk_breakdown={},
                        recommendations=["Unable to analyze risk"]
                    )
                    
        except asyncio.TimeoutError:
            logger.error("PCA risk analysis request timed out")
            return RiskAnalysisResult(
                risk_score=0.5,
                risk_factors=["timeout"],
                confidence=0.0,
                risk_breakdown={},
                recommendations=["Request timed out"]
            )
        except Exception as e:
            logger.error(f"Error in PCA risk analysis: {e}")
            return RiskAnalysisResult(
                risk_score=0.5,
                risk_factors=["error"],
                confidence=0.0,
                risk_breakdown={},
                recommendations=[f"Error: {str(e)}"]
            )
    
    async def perform_clustering(self, features: List[List[float]], n_clusters: int = 3) -> ClusteringResult:
        """Perform clustering analysis via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/clustering",
                json={"features": features, "n_clusters": n_clusters}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    clustering_data = data["clustering"]
                    return ClusteringResult(
                        cluster_labels=clustering_data["cluster_labels"],
                        n_clusters=clustering_data["n_clusters"],
                        silhouette_score=clustering_data["silhouette_score"],
                        cluster_centers=clustering_data["cluster_centers"],
                        cluster_info=clustering_data["cluster_info"]
                    )
                else:
                    logger.error(f"Clustering analysis failed: {response.status}")
                    return ClusteringResult(
                        cluster_labels=[0] * len(features),
                        n_clusters=1,
                        silhouette_score=0.0,
                        cluster_centers=[],
                        cluster_info={}
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Clustering analysis request timed out")
            return ClusteringResult(
                cluster_labels=[0] * len(features),
                n_clusters=1,
                silhouette_score=0.0,
                cluster_centers=[],
                cluster_info={}
            )
        except Exception as e:
            logger.error(f"Error in clustering analysis: {e}")
            return ClusteringResult(
                cluster_labels=[0] * len(features),
                n_clusters=1,
                silhouette_score=0.0,
                cluster_centers=[],
                cluster_info={}
            )
    
    async def detect_anomalies(self, features: List[List[float]]) -> RiskAnalysisResult:
        """Detect anomalies via MCP server"""
        await self._ensure_session()
        
        try:
            async with self.session.post(
                f"{self.base_url}/anomaly-detection",
                json={"features": features}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    anomaly_data = data["anomaly_analysis"]
                    return RiskAnalysisResult(
                        risk_score=anomaly_data["risk_score"],
                        risk_factors=anomaly_data["risk_factors"],
                        confidence=anomaly_data["confidence"],
                        risk_breakdown=anomaly_data["risk_breakdown"],
                        recommendations=anomaly_data["recommendations"]
                    )
                else:
                    logger.error(f"Anomaly detection failed: {response.status}")
                    return RiskAnalysisResult(
                        risk_score=0.5,
                        risk_factors=["unknown"],
                        confidence=0.0,
                        risk_breakdown={},
                        recommendations=["Unable to detect anomalies"]
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Anomaly detection request timed out")
            return RiskAnalysisResult(
                risk_score=0.5,
                risk_factors=["timeout"],
                confidence=0.0,
                risk_breakdown={},
                recommendations=["Request timed out"]
            )
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return RiskAnalysisResult(
                risk_score=0.5,
                risk_factors=["error"],
                confidence=0.0,
                risk_breakdown={},
                recommendations=[f"Error: {str(e)}"]
            )
    
    # Analytics/Performance Methods (Phase 5.1)
    async def perform_linear_regression(self, X: List[List[float]], y: List[float]) -> BacktestPredictionResult:
        """Perform linear regression via ML client"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    f"{self.base_url}/linear-regression",
                    json={"X": X, "y": y}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        regression_result = result["regression_result"]
                        return BacktestPredictionResult(
                            prediction=regression_result["prediction"],
                            confidence=regression_result["confidence"],
                            feature_importance=regression_result["feature_importance"],
                            model_type=regression_result["model_type"]
                        )
                    else:
                        logger.error(f"Linear regression request failed: {response.status}")
                        return BacktestPredictionResult(
                            prediction=0.0,
                            confidence=0.0,
                            feature_importance={},
                            model_type="error"
                        )
        except asyncio.TimeoutError:
            logger.error("Linear regression request timed out")
            return BacktestPredictionResult(
                prediction=0.0,
                confidence=0.0,
                feature_importance={},
                model_type="timeout"
            )
        except Exception as e:
            logger.error(f"Error in linear regression: {e}")
            return BacktestPredictionResult(
                prediction=0.0,
                confidence=0.0,
                feature_importance={},
                model_type="error"
            )
    
    async def perform_ridge_regression(self, X: List[List[float]], y: List[float], alpha: float = 1.0) -> BacktestPredictionResult:
        """Perform Ridge regression via ML client"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    f"{self.base_url}/ridge-regression",
                    json={"X": X, "y": y, "alpha": alpha}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        regression_result = result["regression_result"]
                        return BacktestPredictionResult(
                            prediction=regression_result["prediction"],
                            confidence=regression_result["confidence"],
                            feature_importance=regression_result["feature_importance"],
                            model_type=regression_result["model_type"]
                        )
                    else:
                        logger.error(f"Ridge regression request failed: {response.status}")
                        return BacktestPredictionResult(
                            prediction=0.0,
                            confidence=0.0,
                            feature_importance={},
                            model_type="error"
                        )
        except asyncio.TimeoutError:
            logger.error("Ridge regression request timed out")
            return BacktestPredictionResult(
                prediction=0.0,
                confidence=0.0,
                feature_importance={},
                model_type="timeout"
            )
        except Exception as e:
            logger.error(f"Error in ridge regression: {e}")
            return BacktestPredictionResult(
                prediction=0.0,
                confidence=0.0,
                feature_importance={},
                model_type="error"
            )
    
    # Reinforcement Learning Methods (Phase 5.1 - CRITICAL!)
    async def train_rl_model(self, model_type: str, total_timesteps: int = 10000) -> Dict[str, Any]:
        """Train RL model via ML client"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post(
                    f"{self.base_url}/rl-training",
                    json={"model_type": model_type, "total_timesteps": total_timesteps}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["training_result"]
                    else:
                        logger.error(f"RL training request failed: {response.status}")
                        return {
                            "model_type": model_type,
                            "training_timesteps": 0,
                            "status": "error",
                            "confidence": 0.0
                        }
        except asyncio.TimeoutError:
            logger.error("RL training request timed out")
            return {
                "model_type": model_type,
                "training_timesteps": 0,
                "status": "timeout",
                "confidence": 0.0
            }
        except Exception as e:
            logger.error(f"Error in RL training: {e}")
            return {
                "model_type": model_type,
                "training_timesteps": 0,
                "status": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def predict_with_rl_model(self, model_type: str, observation: List[float]) -> Dict[str, Any]:
        """Predict with RL model via ML client"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    f"{self.base_url}/rl-prediction",
                    json={"model_type": model_type, "observation": observation}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["prediction_result"]
                    else:
                        logger.error(f"RL prediction request failed: {response.status}")
                        return {
                            "model_type": model_type,
                            "action": 0,
                            "confidence": 0.0
                        }
        except asyncio.TimeoutError:
            logger.error("RL prediction request timed out")
            return {
                "model_type": model_type,
                "action": 0,
                "confidence": 0.0,
                "error": "timeout"
            }
        except Exception as e:
            logger.error(f"Error in RL prediction: {e}")
            return {
                "model_type": model_type,
                "action": 0,
                "confidence": 0.0,
                "error": str(e)
            }


# Global ML client instance
ml_client = MLClient() 