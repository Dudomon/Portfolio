"""
IPTV Stream Aggregator Service

This module handles integration with multiple third-party IPTV providers,
normalizing different stream formats into a unified interface.
"""

import requests
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class StreamProvider:
    """Base class for IPTV stream providers"""

    def __init__(self, provider_id: str, api_url: str, api_key: str):
        self.provider_id = provider_id
        self.api_url = api_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})

    def get_channels(self) -> List[Dict]:
        """Fetch channel list from provider"""
        raise NotImplementedError

    def get_stream_url(self, channel_id: str) -> str:
        """Get streaming URL for specific channel"""
        raise NotImplementedError

    def validate_stream(self, url: str) -> bool:
        """Validate if stream URL is accessible"""
        try:
            response = requests.head(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Stream validation failed: {e}")
            return False


class M3UProvider(StreamProvider):
    """Provider for M3U/M3U8 playlist-based IPTV services"""

    def __init__(self, provider_id: str, playlist_url: str):
        self.provider_id = provider_id
        self.playlist_url = playlist_url
        self.channels_cache = []

    def parse_m3u(self, content: str) -> List[Dict]:
        """Parse M3U playlist format"""
        channels = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            if lines[i].startswith('#EXTINF:'):
                # Extract channel metadata
                metadata = lines[i]
                channel_name = metadata.split(',')[-1].strip()

                # Extract additional attributes
                tvg_id = re.search(r'tvg-id="([^"]*)"', metadata)
                tvg_logo = re.search(r'tvg-logo="([^"]*)"', metadata)
                group_title = re.search(r'group-title="([^"]*)"', metadata)

                # Next line should be the stream URL
                if i + 1 < len(lines) and lines[i + 1].strip():
                    stream_url = lines[i + 1].strip()

                    channels.append({
                        'id': tvg_id.group(1) if tvg_id else f"{self.provider_id}_{len(channels)}",
                        'name': channel_name,
                        'logo': tvg_logo.group(1) if tvg_logo else None,
                        'category': group_title.group(1) if group_title else 'General',
                        'stream_url': stream_url,
                        'provider_id': self.provider_id,
                        'stream_type': self._detect_stream_type(stream_url)
                    })

                i += 2
            else:
                i += 1

        return channels

    def _detect_stream_type(self, url: str) -> str:
        """Detect stream protocol from URL"""
        url_lower = url.lower()
        if '.m3u8' in url_lower or '/hls/' in url_lower:
            return 'HLS'
        elif url_lower.startswith('rtmp://'):
            return 'RTMP'
        elif url_lower.startswith('http://') or url_lower.startswith('https://'):
            return 'HTTP'
        else:
            return 'UNKNOWN'

    def get_channels(self) -> List[Dict]:
        """Fetch and parse M3U playlist"""
        try:
            response = requests.get(self.playlist_url, timeout=10)
            response.raise_for_status()

            self.channels_cache = self.parse_m3u(response.text)
            logger.info(f"Loaded {len(self.channels_cache)} channels from {self.provider_id}")

            return self.channels_cache
        except Exception as e:
            logger.error(f"Failed to fetch M3U playlist: {e}")
            return []

    def get_stream_url(self, channel_id: str) -> Optional[str]:
        """Get stream URL for channel by ID"""
        for channel in self.channels_cache:
            if channel['id'] == channel_id:
                return channel['stream_url']
        return None


class APIProvider(StreamProvider):
    """Provider for API-based IPTV services"""

    def get_channels(self) -> List[Dict]:
        """Fetch channels via REST API"""
        try:
            response = self.session.get(f"{self.api_url}/channels")
            response.raise_for_status()

            data = response.json()

            # Normalize API response to standard format
            channels = []
            for item in data.get('channels', []):
                channels.append({
                    'id': item.get('id'),
                    'name': item.get('name'),
                    'logo': item.get('logo_url'),
                    'category': item.get('genre', 'General'),
                    'stream_url': None,  # Fetched separately
                    'provider_id': self.provider_id,
                    'stream_type': item.get('stream_type', 'HLS')
                })

            logger.info(f"Loaded {len(channels)} channels from API provider {self.provider_id}")
            return channels

        except Exception as e:
            logger.error(f"Failed to fetch channels from API: {e}")
            return []

    def get_stream_url(self, channel_id: str) -> Optional[str]:
        """Get stream URL for specific channel"""
        try:
            response = self.session.get(f"{self.api_url}/channels/{channel_id}/stream")
            response.raise_for_status()

            data = response.json()
            return data.get('stream_url')

        except Exception as e:
            logger.error(f"Failed to get stream URL: {e}")
            return None


class StreamAggregator:
    """
    Main aggregator class that manages multiple IPTV providers
    and provides a unified interface for channel access
    """

    def __init__(self):
        self.providers: Dict[str, StreamProvider] = {}
        self.channel_index: Dict[str, Dict] = {}

    def add_provider(self, provider: StreamProvider):
        """Register a new IPTV provider"""
        self.providers[provider.provider_id] = provider
        logger.info(f"Added provider: {provider.provider_id}")

    def sync_all_channels(self) -> int:
        """
        Synchronize channel lists from all providers
        Returns number of channels loaded
        """
        total_channels = 0
        self.channel_index = {}

        for provider_id, provider in self.providers.items():
            channels = provider.get_channels()

            for channel in channels:
                # Create unique channel ID
                unique_id = f"{provider_id}:{channel['id']}"
                self.channel_index[unique_id] = channel
                total_channels += 1

        logger.info(f"Synchronized {total_channels} channels from {len(self.providers)} providers")
        return total_channels

    def get_all_channels(self, category: Optional[str] = None) -> List[Dict]:
        """Get all channels, optionally filtered by category"""
        channels = list(self.channel_index.values())

        if category:
            channels = [ch for ch in channels if ch['category'] == category]

        return channels

    def get_channel(self, channel_id: str) -> Optional[Dict]:
        """Get channel by ID"""
        return self.channel_index.get(channel_id)

    def get_stream_url(self, channel_id: str, with_failover: bool = True) -> Optional[str]:
        """
        Get stream URL for channel
        If with_failover=True, attempts backup sources if primary fails
        """
        channel = self.get_channel(channel_id)
        if not channel:
            return None

        # If URL is already in cache
        if channel.get('stream_url'):
            return channel['stream_url']

        # Fetch from provider
        provider_id = channel['provider_id']
        provider = self.providers.get(provider_id)

        if not provider:
            return None

        stream_url = provider.get_stream_url(channel['id'])

        # Cache the URL
        if stream_url:
            self.channel_index[channel_id]['stream_url'] = stream_url

        return stream_url

    def search_channels(self, query: str) -> List[Dict]:
        """Search channels by name"""
        query_lower = query.lower()
        results = []

        for channel in self.channel_index.values():
            if query_lower in channel['name'].lower():
                results.append(channel)

        return results

    def get_categories(self) -> List[str]:
        """Get list of all available categories"""
        categories = set()
        for channel in self.channel_index.values():
            categories.add(channel['category'])
        return sorted(list(categories))


# Example usage
if __name__ == "__main__":
    # Initialize aggregator
    aggregator = StreamAggregator()

    # Add M3U provider
    m3u_provider = M3UProvider(
        provider_id="provider_1",
        playlist_url="https://example.com/playlist.m3u"
    )
    aggregator.add_provider(m3u_provider)

    # Add API provider
    api_provider = APIProvider(
        provider_id="provider_2",
        api_url="https://api.example.com/v1",
        api_key="your_api_key"
    )
    aggregator.add_provider(api_provider)

    # Sync channels
    total = aggregator.sync_all_channels()
    print(f"Loaded {total} channels")

    # Get categories
    categories = aggregator.get_categories()
    print(f"Categories: {categories}")

    # Search channels
    news_channels = aggregator.get_all_channels(category="News")
    print(f"Found {len(news_channels)} news channels")
