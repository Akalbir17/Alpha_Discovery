"""
Text Processing Utilities
Provides text processing and NLP utilities
"""

import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text processing utilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Common stock symbol pattern
        self.symbol_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        
        # Sentiment keywords
        self.positive_keywords = {
            'bullish', 'buy', 'long', 'up', 'rise', 'gain', 'profit', 'strong',
            'good', 'great', 'excellent', 'positive', 'optimistic', 'moon', 'rocket'
        }
        
        self.negative_keywords = {
            'bearish', 'sell', 'short', 'down', 'fall', 'loss', 'weak', 'bad',
            'terrible', 'negative', 'pessimistic', 'crash', 'dump', 'bear'
        }
        
    def extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        try:
            # Find potential symbols
            matches = self.symbol_pattern.findall(text.upper())
            
            # Filter out common words that aren't symbols
            common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'WORD', 'BUT', 'WHAT', 'SOME', 'WE', 'CAN', 'OUT', 'OTHER', 'WERE', 'WHICH', 'THEIR', 'TIME', 'WILL', 'HOW', 'SAID', 'EACH', 'TELL', 'DOES', 'SET', 'THREE', 'WANT', 'AIR', 'WELL', 'ALSO', 'PLAY', 'SMALL', 'END', 'PUT', 'HOME', 'READ', 'HAND', 'PORT', 'LARGE', 'SPELL', 'ADD', 'EVEN', 'LAND', 'HERE', 'MUST', 'BIG', 'HIGH', 'SUCH', 'FOLLOW', 'ACT', 'WHY', 'ASK', 'MEN', 'CHANGE', 'WENT', 'LIGHT', 'KIND', 'OFF', 'NEED', 'HOUSE', 'PICTURE', 'TRY', 'US', 'AGAIN', 'ANIMAL', 'POINT', 'MOTHER', 'WORLD', 'NEAR', 'BUILD', 'SELF', 'EARTH', 'FATHER', 'HEAD', 'STAND', 'OWN', 'PAGE', 'SHOULD', 'COUNTRY', 'FOUND', 'ANSWER', 'SCHOOL', 'GROW', 'STUDY', 'STILL', 'LEARN', 'PLANT', 'COVER', 'FOOD', 'SUN', 'FOUR', 'BETWEEN', 'STATE', 'KEEP', 'EYE', 'NEVER', 'LAST', 'LET', 'THOUGHT', 'CITY', 'TREE', 'CROSS', 'FARM', 'HARD', 'START', 'MIGHT', 'STORY', 'SAW', 'FAR', 'SEA', 'DRAW', 'LEFT', 'LATE', 'RUN', 'DONT', 'WHILE', 'PRESS', 'CLOSE', 'NIGHT', 'REAL', 'LIFE', 'FEW', 'NORTH', 'OPEN', 'SEEM', 'TOGETHER', 'NEXT', 'WHITE', 'CHILDREN', 'BEGINNING', 'GOT', 'WALK', 'EXAMPLE', 'EASE', 'PAPER', 'GROUP', 'ALWAYS', 'MUSIC', 'THOSE', 'BOTH', 'MARK', 'OFTEN', 'LETTER', 'UNTIL', 'MILE', 'RIVER', 'CAR', 'FEET', 'CARE', 'SECOND', 'BOOK', 'CARRY', 'TOOK', 'SCIENCE', 'EAT', 'ROOM', 'FRIEND', 'BEGAN', 'IDEA', 'FISH', 'MOUNTAIN', 'STOP', 'ONCE', 'BASE', 'HEAR', 'HORSE', 'CUT', 'SURE', 'WATCH', 'COLOR', 'FACE', 'WOOD', 'MAIN', 'ENOUGH', 'PLAIN', 'GIRL', 'USUAL', 'YOUNG', 'READY', 'ABOVE', 'EVER', 'RED', 'LIST', 'THOUGH', 'FEEL', 'TALK', 'BIRD', 'SOON', 'BODY', 'DOG', 'FAMILY', 'DIRECT', 'POSE', 'LEAVE', 'SONG', 'MEASURE', 'DOOR', 'PRODUCT', 'BLACK', 'SHORT', 'NUMERAL', 'CLASS', 'WIND', 'QUESTION', 'HAPPEN', 'COMPLETE', 'SHIP', 'AREA', 'HALF', 'ROCK', 'ORDER', 'FIRE', 'SOUTH', 'PROBLEM', 'PIECE', 'TOLD', 'KNEW', 'PASS', 'SINCE', 'TOP', 'WHOLE', 'KING', 'SPACE', 'HEARD', 'BEST', 'HOUR', 'BETTER', 'DURING', 'HUNDRED', 'FIVE', 'REMEMBER', 'STEP', 'EARLY', 'HOLD', 'WEST', 'GROUND', 'INTEREST', 'REACH', 'FAST', 'VERB', 'SING', 'LISTEN', 'SIX', 'TABLE', 'TRAVEL', 'LESS', 'MORNING', 'TEN', 'SIMPLE', 'SEVERAL', 'VOWEL', 'TOWARD', 'WAR', 'LAY', 'AGAINST', 'PATTERN', 'SLOW', 'CENTER', 'LOVE', 'PERSON', 'MONEY', 'SERVE', 'APPEAR', 'ROAD', 'MAP', 'RAIN', 'RULE', 'GOVERN', 'PULL', 'COLD', 'NOTICE', 'VOICE', 'UNIT', 'POWER', 'TOWN', 'FINE', 'CERTAIN', 'FLY', 'FALL', 'LEAD', 'CRY', 'DARK', 'MACHINE', 'NOTE', 'WAIT', 'PLAN', 'FIGURE', 'STAR', 'BOX', 'NOUN', 'FIELD', 'REST', 'CORRECT', 'ABLE', 'POUND', 'DONE', 'BEAUTY', 'DRIVE', 'STOOD', 'CONTAIN', 'FRONT', 'TEACH', 'WEEK', 'FINAL', 'GAVE', 'GREEN', 'OH', 'QUICK', 'DEVELOP', 'OCEAN', 'WARM', 'FREE', 'MINUTE', 'STRONG', 'SPECIAL', 'MIND', 'BEHIND', 'CLEAR', 'TAIL', 'PRODUCE', 'FACT', 'STREET', 'INCH', 'MULTIPLY', 'NOTHING', 'COURSE', 'STAY', 'WHEEL', 'FULL', 'FORCE', 'BLUE', 'OBJECT', 'DECIDE', 'SURFACE', 'DEEP', 'MOON', 'ISLAND', 'FOOT', 'SYSTEM', 'BUSY', 'TEST', 'RECORD', 'BOAT', 'COMMON', 'GOLD', 'POSSIBLE', 'PLANE', 'STEAD', 'DRY', 'WONDER', 'LAUGH', 'THOUSANDS', 'AGO', 'RAN', 'CHECK', 'GAME', 'SHAPE', 'EQUATE', 'MISS', 'BROUGHT', 'HEAT', 'SNOW', 'TIRE', 'BRING', 'YES', 'DISTANT', 'FILL', 'EAST', 'PAINT', 'LANGUAGE', 'AMONG'}
            
            symbols = [match for match in matches if match not in common_words]
            
            # Remove duplicates while preserving order
            unique_symbols = []
            seen = set()
            for symbol in symbols:
                if symbol not in seen:
                    unique_symbols.append(symbol)
                    seen.add(symbol)
                    
            return unique_symbols
            
        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")
            return []
            
    def calculate_sentiment(self, text: str) -> Dict[str, Any]:
        """Calculate sentiment score for text"""
        try:
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            positive_count = sum(1 for word in words if word in self.positive_keywords)
            negative_count = sum(1 for word in words if word in self.negative_keywords)
            
            total_sentiment_words = positive_count + negative_count
            
            if total_sentiment_words == 0:
                sentiment_score = 0.0
                confidence = 0.0
            else:
                sentiment_score = (positive_count - negative_count) / total_sentiment_words
                confidence = min(1.0, total_sentiment_words / 10)  # Max confidence at 10+ sentiment words
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'positive_words': positive_count,
                'negative_words': negative_count,
                'total_words': len(words),
                'sentiment_words': total_sentiment_words
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'positive_words': 0,
                'negative_words': 0,
                'total_words': 0,
                'sentiment_words': 0
            }
            
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        try:
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove mentions and hashtags
            text = re.sub(r'[@#]\w+', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
            
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text"""
        try:
            # Simple phrase extraction - look for sequences of capitalized words
            phrases = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
            
            # Filter out single words and common phrases
            filtered_phrases = []
            for phrase in phrases:
                if len(phrase.split()) > 1 and len(phrase) > 3:
                    filtered_phrases.append(phrase)
            
            # Remove duplicates and limit
            unique_phrases = list(set(filtered_phrases))[:max_phrases]
            
            return unique_phrases
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
            
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        try:
            cleaned_text = self.clean_text(text)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'symbols': self.extract_symbols(text),
                'sentiment': self.calculate_sentiment(cleaned_text),
                'key_phrases': self.extract_key_phrases(text),
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text),
                'analyzed_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                'original_text': text,
                'error': str(e),
                'analyzed_at': datetime.now()
            } 