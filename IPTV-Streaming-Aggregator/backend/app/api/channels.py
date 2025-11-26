"""
Channel Management API Endpoints

RESTful API for managing IPTV channels, streams, and metadata.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Create Blueprint
channels_bp = Blueprint('channels', __name__, url_prefix='/api/channels')


# Sample in-memory database (replace with actual DB in production)
class ChannelDatabase:
    def __init__(self):
        self.channels = {}
        self.next_id = 1

    def create_channel(self, data: Dict) -> Dict:
        channel_id = str(self.next_id)
        self.next_id += 1

        channel = {
            'id': channel_id,
            'name': data['name'],
            'logo_url': data.get('logo_url'),
            'category': data.get('category', 'General'),
            'stream_url': data['stream_url'],
            'stream_type': data.get('stream_type', 'HLS'),
            'provider_id': data.get('provider_id'),
            'enabled': data.get('enabled', True),
            'order': data.get('order', 0),
            'metadata': data.get('metadata', {})
        }

        self.channels[channel_id] = channel
        return channel

    def get_channel(self, channel_id: str) -> Dict:
        return self.channels.get(channel_id)

    def update_channel(self, channel_id: str, data: Dict) -> Dict:
        if channel_id not in self.channels:
            return None

        channel = self.channels[channel_id]
        channel.update(data)
        return channel

    def delete_channel(self, channel_id: str) -> bool:
        if channel_id in self.channels:
            del self.channels[channel_id]
            return True
        return False

    def list_channels(self, filters: Dict = None) -> List[Dict]:
        channels = list(self.channels.values())

        if filters:
            if 'category' in filters:
                channels = [ch for ch in channels if ch['category'] == filters['category']]
            if 'enabled' in filters:
                channels = [ch for ch in channels if ch['enabled'] == filters['enabled']]

        # Sort by order
        channels.sort(key=lambda x: x['order'])
        return channels


# Initialize database
db = ChannelDatabase()


@channels_bp.route('', methods=['GET'])
def list_channels():
    """
    GET /api/channels
    List all channels with optional filtering

    Query Parameters:
    - category: Filter by category
    - enabled: Filter by enabled status (true/false)
    - search: Search in channel names
    """
    try:
        # Get query parameters
        category = request.args.get('category')
        enabled = request.args.get('enabled')
        search = request.args.get('search', '').lower()

        filters = {}
        if category:
            filters['category'] = category
        if enabled is not None:
            filters['enabled'] = enabled.lower() == 'true'

        # Get channels
        channels = db.list_channels(filters)

        # Apply search filter
        if search:
            channels = [
                ch for ch in channels
                if search in ch['name'].lower()
            ]

        return jsonify({
            'success': True,
            'count': len(channels),
            'channels': channels
        }), 200

    except Exception as e:
        logger.error(f"Error listing channels: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@channels_bp.route('/<channel_id>', methods=['GET'])
def get_channel(channel_id: str):
    """
    GET /api/channels/:id
    Get single channel by ID
    """
    channel = db.get_channel(channel_id)

    if not channel:
        return jsonify({
            'success': False,
            'error': 'Channel not found'
        }), 404

    return jsonify({
        'success': True,
        'channel': channel
    }), 200


@channels_bp.route('', methods=['POST'])
@jwt_required()
def create_channel():
    """
    POST /api/channels
    Create new channel (requires authentication)

    Request Body:
    {
        "name": "Channel Name",
        "stream_url": "https://...",
        "category": "News",
        "logo_url": "https://...",
        "stream_type": "HLS"
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        if not data.get('name'):
            return jsonify({
                'success': False,
                'error': 'Channel name is required'
            }), 400

        if not data.get('stream_url'):
            return jsonify({
                'success': False,
                'error': 'Stream URL is required'
            }), 400

        # Create channel
        channel = db.create_channel(data)

        logger.info(f"Channel created: {channel['id']} - {channel['name']}")

        return jsonify({
            'success': True,
            'channel': channel
        }), 201

    except Exception as e:
        logger.error(f"Error creating channel: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@channels_bp.route('/<channel_id>', methods=['PUT'])
@jwt_required()
def update_channel(channel_id: str):
    """
    PUT /api/channels/:id
    Update existing channel (requires authentication)
    """
    try:
        data = request.get_json()

        # Update channel
        channel = db.update_channel(channel_id, data)

        if not channel:
            return jsonify({
                'success': False,
                'error': 'Channel not found'
            }), 404

        logger.info(f"Channel updated: {channel_id}")

        return jsonify({
            'success': True,
            'channel': channel
        }), 200

    except Exception as e:
        logger.error(f"Error updating channel: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@channels_bp.route('/<channel_id>', methods=['DELETE'])
@jwt_required()
def delete_channel(channel_id: str):
    """
    DELETE /api/channels/:id
    Delete channel (requires authentication)
    """
    try:
        success = db.delete_channel(channel_id)

        if not success:
            return jsonify({
                'success': False,
                'error': 'Channel not found'
            }), 404

        logger.info(f"Channel deleted: {channel_id}")

        return jsonify({
            'success': True,
            'message': 'Channel deleted successfully'
        }), 200

    except Exception as e:
        logger.error(f"Error deleting channel: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@channels_bp.route('/<channel_id>/stream', methods=['GET'])
def get_stream_url(channel_id: str):
    """
    GET /api/channels/:id/stream
    Get streaming URL for channel

    This endpoint can implement additional logic like:
    - Token-based URL generation
    - Geographic restrictions
    - Bandwidth selection
    """
    channel = db.get_channel(channel_id)

    if not channel:
        return jsonify({
            'success': False,
            'error': 'Channel not found'
        }), 404

    if not channel['enabled']:
        return jsonify({
            'success': False,
            'error': 'Channel is disabled'
        }), 403

    # In production, you might generate a time-limited signed URL here
    stream_url = channel['stream_url']

    return jsonify({
        'success': True,
        'stream_url': stream_url,
        'stream_type': channel['stream_type'],
        'expires_in': 3600  # URL valid for 1 hour
    }), 200


@channels_bp.route('/<channel_id>/health', methods=['GET'])
def check_channel_health(channel_id: str):
    """
    GET /api/channels/:id/health
    Check stream health status
    """
    channel = db.get_channel(channel_id)

    if not channel:
        return jsonify({
            'success': False,
            'error': 'Channel not found'
        }), 404

    # In production, this would query the health monitoring service
    # For now, return sample data
    return jsonify({
        'success': True,
        'health': {
            'channel_id': channel_id,
            'status': 'online',
            'uptime_percentage': 99.5,
            'last_checked': '2024-11-26T10:30:00Z',
            'response_time_ms': 145
        }
    }), 200


@channels_bp.route('/categories', methods=['GET'])
def get_categories():
    """
    GET /api/channels/categories
    Get list of all available categories
    """
    channels = db.list_channels()
    categories = set(ch['category'] for ch in channels)

    return jsonify({
        'success': True,
        'categories': sorted(list(categories))
    }), 200


@channels_bp.route('/bulk', methods=['POST'])
@jwt_required()
def bulk_import_channels():
    """
    POST /api/channels/bulk
    Import multiple channels at once (requires authentication)

    Request Body:
    {
        "channels": [
            {"name": "...", "stream_url": "...", ...},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        channels_data = data.get('channels', [])

        if not channels_data:
            return jsonify({
                'success': False,
                'error': 'No channels provided'
            }), 400

        created_channels = []
        errors = []

        for idx, channel_data in enumerate(channels_data):
            try:
                channel = db.create_channel(channel_data)
                created_channels.append(channel)
            except Exception as e:
                errors.append({
                    'index': idx,
                    'error': str(e)
                })

        logger.info(f"Bulk import: {len(created_channels)} channels created, {len(errors)} errors")

        return jsonify({
            'success': True,
            'created': len(created_channels),
            'errors': errors,
            'channels': created_channels
        }), 201

    except Exception as e:
        logger.error(f"Error in bulk import: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


# Error handlers
@channels_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@channels_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
