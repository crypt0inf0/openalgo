"""
Subscription Manager for WebSocket Proxy
Properly tracks client subscriptions and manages broker subscriptions with reference counting
"""

import threading
import time
from typing import Dict, Set, Optional, Tuple, Any, List
from collections import defaultdict
from utils.logging import get_logger

logger = get_logger(__name__)


class SubscriptionManager:
    """
    Manages client subscriptions with proper reference counting
    Ensures broker subscriptions are only sent when needed and unsubscriptions
    only happen when no clients are subscribed to a symbol
    """
    
    def __init__(self):
        # Thread safety
        self._lock = threading.RLock()
        
        # Client subscription tracking: client_id -> set of topics
        self.client_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Symbol reference counting: topic -> count of subscribed clients
        self.symbol_ref_count: Dict[str, int] = defaultdict(int)
        
        # Broker subscription status: topic -> bool (True if subscribed to broker)
        self.broker_subscriptions: Dict[str, bool] = defaultdict(bool)
        
        # Topic to symbol/exchange/mode mapping for broker operations
        self.topic_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            'total_subscriptions': 0,
            'total_unsubscriptions': 0,
            'broker_subscriptions_sent': 0,
            'broker_unsubscriptions_sent': 0,
            'duplicate_subscriptions_avoided': 0,
            'premature_unsubscriptions_avoided': 0,
            'orphaned_symbols_cleaned': 0,
            'last_cleanup_time': 0
        }
    
    def _create_topic(self, symbol: str, exchange: str, mode: int) -> str:
        """Create topic string from symbol, exchange, and mode"""
        mode_str = {1: 'LTP', 2: 'QUOTE', 3: 'DEPTH'}.get(mode, 'LTP')
        return f"{exchange}_{symbol}_{mode_str}"
    
    def subscribe_client(self, client_id: str, symbol: str, exchange: str, mode: int) -> Tuple[bool, bool, str]:
        """
        Subscribe a client to a symbol
        
        Args:
            client_id: Unique client identifier
            symbol: Trading symbol
            exchange: Exchange name
            mode: Subscription mode (1=LTP, 2=QUOTE, 3=DEPTH)
            
        Returns:
            Tuple[bool, bool, str]: (success, should_subscribe_to_broker, message)
        """
        with self._lock:
            try:
                topic = self._create_topic(symbol, exchange, mode)
                
                # Store topic metadata
                self.topic_metadata[topic] = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'mode': mode
                }
                
                # Check if client is already subscribed to this topic
                if topic in self.client_subscriptions[client_id]:
                    logger.debug(f"Client {client_id} already subscribed to {topic}")
                    return True, False, f"Already subscribed to {symbol}"
                
                # Add client subscription
                self.client_subscriptions[client_id].add(topic)
                self.symbol_ref_count[topic] += 1
                self.stats['total_subscriptions'] += 1
                
                # Determine if we need to subscribe to broker
                should_subscribe_to_broker = False
                if not self.broker_subscriptions[topic]:
                    # First client for this symbol - need to subscribe to broker
                    should_subscribe_to_broker = True
                    self.broker_subscriptions[topic] = True
                    self.stats['broker_subscriptions_sent'] += 1
                    logger.info(f"First subscription for {topic} - will subscribe to broker")
                else:
                    # Already subscribed to broker - reuse existing subscription
                    self.stats['duplicate_subscriptions_avoided'] += 1
                    logger.debug(f"Reusing existing broker subscription for {topic} (ref_count: {self.symbol_ref_count[topic]})")
                
                logger.info(f"Client {client_id} subscribed to {topic} (ref_count: {self.symbol_ref_count[topic]})")
                return True, should_subscribe_to_broker, f"Subscribed to {symbol}"
                
            except Exception as e:
                logger.error(f"Error in subscribe_client: {e}")
                return False, False, f"Subscription failed: {str(e)}"
    
    def unsubscribe_client(self, client_id: str, symbol: str, exchange: str, mode: int) -> Tuple[bool, bool, str]:
        """
        Unsubscribe a client from a symbol
        
        Args:
            client_id: Unique client identifier
            symbol: Trading symbol
            exchange: Exchange name
            mode: Subscription mode
            
        Returns:
            Tuple[bool, bool, str]: (success, should_unsubscribe_from_broker, message)
        """
        with self._lock:
            try:
                topic = self._create_topic(symbol, exchange, mode)
                
                # Check if client is subscribed to this topic
                if topic not in self.client_subscriptions[client_id]:
                    logger.debug(f"Client {client_id} not subscribed to {topic}")
                    return True, False, f"Not subscribed to {symbol}"
                
                # Remove client subscription
                self.client_subscriptions[client_id].discard(topic)
                self.symbol_ref_count[topic] -= 1
                self.stats['total_unsubscriptions'] += 1
                
                # Determine if we need to unsubscribe from broker
                should_unsubscribe_from_broker = False
                if self.symbol_ref_count[topic] <= 0:
                    # No more clients for this symbol - unsubscribe from broker
                    should_unsubscribe_from_broker = True
                    self.broker_subscriptions[topic] = False
                    self.stats['broker_unsubscriptions_sent'] += 1
                    
                    # Clean up
                    del self.symbol_ref_count[topic]
                    self.topic_metadata.pop(topic, None)
                    
                    logger.info(f"Last client unsubscribed from {topic} - will unsubscribe from broker")
                else:
                    # Other clients still subscribed - keep broker subscription
                    self.stats['premature_unsubscriptions_avoided'] += 1
                    logger.debug(f"Other clients still subscribed to {topic} (ref_count: {self.symbol_ref_count[topic]}) - keeping broker subscription")
                
                logger.info(f"Client {client_id} unsubscribed from {topic} (ref_count: {self.symbol_ref_count.get(topic, 0)})")
                return True, should_unsubscribe_from_broker, f"Unsubscribed from {symbol}"
                
            except Exception as e:
                logger.error(f"Error in unsubscribe_client: {e}")
                return False, False, f"Unsubscription failed: {str(e)}"
    
    def cleanup_client(self, client_id: str) -> Dict[str, Any]:
        """
        Clean up all subscriptions for a disconnected client
        
        Args:
            client_id: Client identifier to clean up
            
        Returns:
            Dict with cleanup information including symbols to unsubscribe from broker
        """
        with self._lock:
            cleanup_info = {
                'client_topics_removed': [],
                'broker_unsubscriptions_needed': [],
                'remaining_subscriptions': {}
            }
            
            try:
                if client_id not in self.client_subscriptions:
                    return cleanup_info
                
                # Get all topics this client was subscribed to
                client_topics = self.client_subscriptions[client_id].copy()
                cleanup_info['client_topics_removed'] = list(client_topics)
                
                # Process each topic
                for topic in client_topics:
                    # Decrease reference count
                    self.symbol_ref_count[topic] -= 1
                    
                    # Check if we need to unsubscribe from broker
                    if self.symbol_ref_count[topic] <= 0:
                        # No more clients - need to unsubscribe from broker
                        if topic in self.topic_metadata:
                            metadata = self.topic_metadata[topic]
                            cleanup_info['broker_unsubscriptions_needed'].append({
                                'symbol': metadata['symbol'],
                                'exchange': metadata['exchange'],
                                'mode': metadata['mode'],
                                'topic': topic
                            })
                        
                        # Clean up
                        self.broker_subscriptions[topic] = False
                        del self.symbol_ref_count[topic]
                        self.topic_metadata.pop(topic, None)
                        self.stats['broker_unsubscriptions_sent'] += 1
                    else:
                        # Other clients still subscribed
                        cleanup_info['remaining_subscriptions'][topic] = self.symbol_ref_count[topic]
                
                # Remove client subscriptions
                del self.client_subscriptions[client_id]
                
                logger.info(f"Cleaned up client {client_id}: removed {len(client_topics)} subscriptions, "
                           f"broker unsubscriptions needed: {len(cleanup_info['broker_unsubscriptions_needed'])}")
                
            except Exception as e:
                logger.error(f"Error in cleanup_client: {e}")
                cleanup_info['error'] = str(e)
            
            return cleanup_info
    
    def get_client_subscriptions(self, client_id: str) -> Set[str]:
        """Get all topics a client is subscribed to"""
        with self._lock:
            return self.client_subscriptions[client_id].copy()
    
    def get_symbol_subscribers_count(self, symbol: str, exchange: str, mode: int) -> int:
        """Get number of clients subscribed to a symbol"""
        with self._lock:
            topic = self._create_topic(symbol, exchange, mode)
            return self.symbol_ref_count.get(topic, 0)
    
    def is_broker_subscribed(self, symbol: str, exchange: str, mode: int) -> bool:
        """Check if we're subscribed to this symbol at broker level"""
        with self._lock:
            topic = self._create_topic(symbol, exchange, mode)
            return self.broker_subscriptions.get(topic, False)
    
    def get_all_broker_subscriptions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active broker subscriptions with metadata"""
        with self._lock:
            active_subscriptions = {}
            for topic, is_subscribed in self.broker_subscriptions.items():
                if is_subscribed and topic in self.topic_metadata:
                    active_subscriptions[topic] = {
                        **self.topic_metadata[topic],
                        'client_count': self.symbol_ref_count.get(topic, 0)
                    }
            return active_subscriptions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get subscription manager statistics"""
        with self._lock:
            return {
                **self.stats,
                'active_clients': len(self.client_subscriptions),
                'active_topics': len([t for t, subscribed in self.broker_subscriptions.items() if subscribed]),
                'total_client_subscriptions': sum(len(subs) for subs in self.client_subscriptions.values()),
                'topics_with_multiple_clients': len([t for t, count in self.symbol_ref_count.items() if count > 1])
            }
    
    def cleanup_orphaned_symbols(self) -> Dict[str, Any]:
        """
        Clean up orphaned symbols that have broker subscriptions but no client subscriptions
        This can happen due to network issues, client crashes, or other edge cases
        
        Returns:
            Dict with cleanup information
        """
        with self._lock:
            cleanup_info = {
                'orphaned_topics_found': [],
                'broker_unsubscriptions_needed': [],
                'cleaned_count': 0,
                'cleanup_time': time.time()
            }
            
            try:
                # Find topics that have broker subscriptions but zero reference count
                orphaned_topics = []
                
                for topic, is_broker_subscribed in self.broker_subscriptions.items():
                    if is_broker_subscribed and self.symbol_ref_count.get(topic, 0) <= 0:
                        orphaned_topics.append(topic)
                
                cleanup_info['orphaned_topics_found'] = orphaned_topics
                
                # Clean up orphaned topics
                for topic in orphaned_topics:
                    if topic in self.topic_metadata:
                        metadata = self.topic_metadata[topic]
                        cleanup_info['broker_unsubscriptions_needed'].append({
                            'symbol': metadata['symbol'],
                            'exchange': metadata['exchange'],
                            'mode': metadata['mode'],
                            'topic': topic
                        })
                    
                    # Clean up all references
                    self.broker_subscriptions[topic] = False
                    self.symbol_ref_count.pop(topic, None)
                    self.topic_metadata.pop(topic, None)
                    
                    cleanup_info['cleaned_count'] += 1
                    self.stats['orphaned_symbols_cleaned'] += 1
                
                self.stats['last_cleanup_time'] = cleanup_info['cleanup_time']
                
                if cleanup_info['cleaned_count'] > 0:
                    logger.info(f"Cleaned up {cleanup_info['cleaned_count']} orphaned symbols")
                else:
                    logger.debug("No orphaned symbols found during cleanup")
                
            except Exception as e:
                logger.error(f"Error in cleanup_orphaned_symbols: {e}")
                cleanup_info['error'] = str(e)
            
            return cleanup_info
    
    def force_cleanup_symbol(self, symbol: str, exchange: str, mode: int) -> bool:
        """
        Force cleanup of a specific symbol regardless of reference count
        Useful for manual cleanup or error recovery
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            mode: Subscription mode
            
        Returns:
            bool: True if symbol was cleaned up
        """
        with self._lock:
            try:
                topic = self._create_topic(symbol, exchange, mode)
                
                if topic not in self.broker_subscriptions:
                    logger.debug(f"Symbol {topic} not found in broker subscriptions")
                    return False
                
                # Remove from all tracking structures
                was_subscribed = self.broker_subscriptions.get(topic, False)
                
                self.broker_subscriptions.pop(topic, None)
                self.symbol_ref_count.pop(topic, None)
                self.topic_metadata.pop(topic, None)
                
                # Remove from all client subscriptions
                clients_affected = []
                for client_id, client_topics in self.client_subscriptions.items():
                    if topic in client_topics:
                        client_topics.discard(topic)
                        clients_affected.append(client_id)
                
                logger.info(f"Force cleaned symbol {topic} (was_broker_subscribed: {was_subscribed}, "
                           f"clients_affected: {len(clients_affected)})")
                
                self.stats['orphaned_symbols_cleaned'] += 1
                return was_subscribed
                
            except Exception as e:
                logger.error(f"Error in force_cleanup_symbol: {e}")
                return False
    
    def get_orphaned_symbols(self) -> List[Dict[str, Any]]:
        """
        Get list of orphaned symbols (broker subscribed but no clients)
        
        Returns:
            List of orphaned symbol information
        """
        with self._lock:
            orphaned = []
            
            for topic, is_broker_subscribed in self.broker_subscriptions.items():
                if is_broker_subscribed and self.symbol_ref_count.get(topic, 0) <= 0:
                    if topic in self.topic_metadata:
                        metadata = self.topic_metadata[topic]
                        orphaned.append({
                            'topic': topic,
                            'symbol': metadata['symbol'],
                            'exchange': metadata['exchange'],
                            'mode': metadata['mode'],
                            'ref_count': self.symbol_ref_count.get(topic, 0)
                        })
            
            return orphaned
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status for debugging"""
        with self._lock:
            return {
                'client_subscriptions': {k: list(v) for k, v in self.client_subscriptions.items()},
                'symbol_ref_count': dict(self.symbol_ref_count),
                'broker_subscriptions': dict(self.broker_subscriptions),
                'topic_metadata': dict(self.topic_metadata),
                'orphaned_symbols': self.get_orphaned_symbols(),
                'stats': self.get_stats()
            }