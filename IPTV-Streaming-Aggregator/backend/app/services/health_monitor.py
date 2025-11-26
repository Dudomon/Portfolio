"""
Stream Health Monitoring Service

Continuously monitors IPTV streams to ensure availability and quality.
Provides alerting and automatic failover capabilities.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import requests

logger = logging.getLogger(__name__)


class StreamStatus(Enum):
    """Stream health status"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class StreamHealthCheck:
    """Represents health check result for a stream"""

    def __init__(self, channel_id: str, status: StreamStatus,
                 response_time: Optional[float] = None,
                 error_message: Optional[str] = None):
        self.channel_id = channel_id
        self.status = status
        self.response_time = response_time
        self.error_message = error_message
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        return {
            'channel_id': self.channel_id,
            'status': self.status.value,
            'response_time_ms': round(self.response_time * 1000, 2) if self.response_time else None,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat()
        }


class StreamMonitor:
    """
    Monitors IPTV stream health in background
    Performs periodic checks and triggers alerts
    """

    def __init__(self, check_interval: int = 60, timeout: int = 5):
        """
        Args:
            check_interval: Seconds between health checks
            timeout: Request timeout in seconds
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self.channels: Dict[str, Dict] = {}
        self.health_history: Dict[str, List[StreamHealthCheck]] = {}
        self.alert_callbacks: List[Callable] = []
        self.running = False
        self.monitor_thread = None

    def add_channel(self, channel_id: str, stream_url: str, metadata: Optional[Dict] = None):
        """Register channel for monitoring"""
        self.channels[channel_id] = {
            'stream_url': stream_url,
            'metadata': metadata or {},
            'last_check': None,
            'current_status': StreamStatus.UNKNOWN,
            'consecutive_failures': 0,
            'uptime_percentage': 100.0
        }
        self.health_history[channel_id] = []
        logger.info(f"Added channel {channel_id} to monitoring")

    def remove_channel(self, channel_id: str):
        """Remove channel from monitoring"""
        if channel_id in self.channels:
            del self.channels[channel_id]
            del self.health_history[channel_id]
            logger.info(f"Removed channel {channel_id} from monitoring")

    def register_alert_callback(self, callback: Callable):
        """
        Register callback function for alerts
        Callback signature: callback(channel_id: str, status: StreamStatus, message: str)
        """
        self.alert_callbacks.append(callback)

    def check_stream(self, channel_id: str, stream_url: str) -> StreamHealthCheck:
        """
        Perform health check on single stream
        Tests if stream URL is accessible
        """
        try:
            start_time = time.time()

            # For HLS streams, check if m3u8 playlist is accessible
            if '.m3u8' in stream_url.lower():
                response = requests.head(stream_url, timeout=self.timeout, allow_redirects=True)
            else:
                # For other streams, try GET with small byte range
                response = requests.get(stream_url, timeout=self.timeout, stream=True)
                # Read first chunk to verify stream is actually working
                next(response.iter_content(chunk_size=1024), None)
                response.close()

            response_time = time.time() - start_time

            if response.status_code == 200:
                status = StreamStatus.ONLINE
                error_msg = None
            else:
                status = StreamStatus.DEGRADED
                error_msg = f"HTTP {response.status_code}"

            return StreamHealthCheck(channel_id, status, response_time, error_msg)

        except requests.exceptions.Timeout:
            return StreamHealthCheck(
                channel_id, StreamStatus.OFFLINE, None,
                "Connection timeout"
            )
        except requests.exceptions.ConnectionError:
            return StreamHealthCheck(
                channel_id, StreamStatus.OFFLINE, None,
                "Connection refused"
            )
        except Exception as e:
            return StreamHealthCheck(
                channel_id, StreamStatus.OFFLINE, None,
                str(e)
            )

    def _monitor_loop(self):
        """Background monitoring loop"""
        logger.info("Stream monitoring started")

        while self.running:
            check_start = time.time()

            # Check all channels
            for channel_id, channel_data in self.channels.items():
                stream_url = channel_data['stream_url']

                # Perform health check
                health_check = self.check_stream(channel_id, stream_url)

                # Update channel data
                channel_data['last_check'] = health_check.timestamp
                previous_status = channel_data['current_status']
                channel_data['current_status'] = health_check.status

                # Track consecutive failures
                if health_check.status == StreamStatus.OFFLINE:
                    channel_data['consecutive_failures'] += 1
                else:
                    channel_data['consecutive_failures'] = 0

                # Store in history (keep last 100 checks)
                self.health_history[channel_id].append(health_check)
                if len(self.health_history[channel_id]) > 100:
                    self.health_history[channel_id].pop(0)

                # Calculate uptime percentage
                channel_data['uptime_percentage'] = self._calculate_uptime(channel_id)

                # Trigger alerts on status change
                if previous_status != health_check.status:
                    self._trigger_alerts(channel_id, health_check)

                # Alert on consecutive failures
                if channel_data['consecutive_failures'] >= 3:
                    logger.warning(
                        f"Channel {channel_id} has {channel_data['consecutive_failures']} consecutive failures"
                    )

            # Calculate time to next check
            check_duration = time.time() - check_start
            sleep_time = max(0, self.check_interval - check_duration)

            logger.debug(f"Health check completed in {check_duration:.2f}s. Next check in {sleep_time:.2f}s")

            # Sleep until next check
            time.sleep(sleep_time)

        logger.info("Stream monitoring stopped")

    def _calculate_uptime(self, channel_id: str, hours: int = 24) -> float:
        """Calculate uptime percentage over last N hours"""
        history = self.health_history.get(channel_id, [])

        if not history:
            return 100.0

        # Filter checks from last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_checks = [
            check for check in history
            if check.timestamp >= cutoff_time
        ]

        if not recent_checks:
            return 100.0

        online_count = sum(
            1 for check in recent_checks
            if check.status == StreamStatus.ONLINE
        )

        return (online_count / len(recent_checks)) * 100.0

    def _trigger_alerts(self, channel_id: str, health_check: StreamHealthCheck):
        """Trigger registered alert callbacks"""
        channel_name = self.channels[channel_id]['metadata'].get('name', channel_id)

        if health_check.status == StreamStatus.OFFLINE:
            message = f"Stream '{channel_name}' went OFFLINE: {health_check.error_message}"
        elif health_check.status == StreamStatus.ONLINE:
            message = f"Stream '{channel_name}' is back ONLINE"
        elif health_check.status == StreamStatus.DEGRADED:
            message = f"Stream '{channel_name}' is DEGRADED: {health_check.error_message}"
        else:
            return

        logger.warning(message)

        # Call all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(channel_id, health_check.status, message)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def start(self):
        """Start background monitoring"""
        if self.running:
            logger.warning("Monitor already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Stream monitor started")

    def stop(self):
        """Stop background monitoring"""
        if not self.running:
            return

        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Stream monitor stopped")

    def get_channel_status(self, channel_id: str) -> Optional[Dict]:
        """Get current status for channel"""
        channel_data = self.channels.get(channel_id)
        if not channel_data:
            return None

        return {
            'channel_id': channel_id,
            'status': channel_data['current_status'].value,
            'last_check': channel_data['last_check'].isoformat() if channel_data['last_check'] else None,
            'consecutive_failures': channel_data['consecutive_failures'],
            'uptime_percentage': channel_data['uptime_percentage']
        }

    def get_all_statuses(self) -> List[Dict]:
        """Get status for all monitored channels"""
        return [
            self.get_channel_status(channel_id)
            for channel_id in self.channels.keys()
        ]

    def get_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        total_channels = len(self.channels)
        online_channels = sum(
            1 for ch in self.channels.values()
            if ch['current_status'] == StreamStatus.ONLINE
        )
        offline_channels = sum(
            1 for ch in self.channels.values()
            if ch['current_status'] == StreamStatus.OFFLINE
        )
        degraded_channels = sum(
            1 for ch in self.channels.values()
            if ch['current_status'] == StreamStatus.DEGRADED
        )

        avg_uptime = sum(
            ch['uptime_percentage'] for ch in self.channels.values()
        ) / total_channels if total_channels > 0 else 0

        return {
            'total_channels': total_channels,
            'online': online_channels,
            'offline': offline_channels,
            'degraded': degraded_channels,
            'average_uptime': round(avg_uptime, 2),
            'timestamp': datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create monitor
    monitor = StreamMonitor(check_interval=30, timeout=5)

    # Define alert callback
    def send_alert(channel_id: str, status: StreamStatus, message: str):
        print(f"[ALERT] {message}")
        # Here you would send email, SMS, webhook, etc.

    monitor.register_alert_callback(send_alert)

    # Add channels to monitor
    monitor.add_channel(
        "cnn_news",
        "https://example.com/cnn/stream.m3u8",
        metadata={'name': 'CNN News', 'category': 'News'}
    )

    monitor.add_channel(
        "bbc_world",
        "https://example.com/bbc/stream.m3u8",
        metadata={'name': 'BBC World', 'category': 'News'}
    )

    # Start monitoring
    monitor.start()

    try:
        # Keep running
        while True:
            time.sleep(60)

            # Print health report every minute
            report = monitor.get_health_report()
            print(f"\n=== Health Report ===")
            print(f"Total: {report['total_channels']}, "
                  f"Online: {report['online']}, "
                  f"Offline: {report['offline']}, "
                  f"Avg Uptime: {report['average_uptime']}%")

    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()
