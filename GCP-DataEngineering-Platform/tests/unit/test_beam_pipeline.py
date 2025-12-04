"""
Unit tests for Apache Beam streaming pipeline components.

Tests cover message parsing, validation, enrichment, and data quality checks
without requiring actual GCP infrastructure.
"""

import unittest
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pipelines/beam')))

from streaming_pipeline import (
    ParsePubSubMessage,
    EnrichEvent,
    ValidateDataQuality,
    FormatForBigQuery
)


class TestParsePubSubMessage(unittest.TestCase):
    """Test message parsing and validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = ParsePubSubMessage()

    def test_parse_valid_message(self):
        """Test parsing a valid JSON message."""
        valid_message = {
            'event_id': 'evt_123',
            'user_id': 'user_456',
            'session_id': 'sess_789',
            'event_type': 'page_view',
            'event_timestamp': '2024-01-15T10:30:00Z'
        }

        message_bytes = json.dumps(valid_message).encode('utf-8')

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([message_bytes])
                | beam.ParDo(self.parser)
            )

            assert_that(output, equal_to([valid_message]))

    def test_parse_invalid_json(self):
        """Test handling of malformed JSON."""
        invalid_json = b'{invalid json'

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([invalid_json])
                | beam.ParDo(self.parser).with_outputs('errors', main='valid')
            )

            # Should produce error, not valid output
            assert_that(output.valid, equal_to([]))

    def test_missing_required_field(self):
        """Test validation of missing required fields."""
        incomplete_message = {
            'event_id': 'evt_123',
            # Missing user_id, session_id, event_type, event_timestamp
        }

        message_bytes = json.dumps(incomplete_message).encode('utf-8')

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([message_bytes])
                | beam.ParDo(self.parser).with_outputs('errors', main='valid')
            )

            # Should produce error due to missing fields
            assert_that(output.valid, equal_to([]))

    def test_invalid_timestamp_format(self):
        """Test handling of invalid timestamp format."""
        message_with_bad_timestamp = {
            'event_id': 'evt_123',
            'user_id': 'user_456',
            'session_id': 'sess_789',
            'event_type': 'page_view',
            'event_timestamp': 'not-a-timestamp'
        }

        message_bytes = json.dumps(message_with_bad_timestamp).encode('utf-8')

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([message_bytes])
                | beam.ParDo(self.parser).with_outputs('errors', main='valid')
            )

            assert_that(output.valid, equal_to([]))


class TestEnrichEvent(unittest.TestCase):
    """Test event enrichment logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.enricher = EnrichEvent()

    def test_enrich_adds_metadata(self):
        """Test that enrichment adds required metadata fields."""
        event = {
            'event_id': 'evt_123',
            'user_id': 'user_456',
            'event_type': 'page_view',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([event])
                | beam.ParDo(self.enricher)
            )

            def check_enrichment(results):
                assert len(results) == 1
                enriched = results[0]
                assert 'ingestion_timestamp' in enriched
                assert 'pipeline_version' in enriched
                assert enriched['pipeline_version'] == '1.0.0'

            assert_that(output, check_enrichment)

    def test_device_type_detection_mobile(self):
        """Test mobile device detection from user agent."""
        event = {
            'event_id': 'evt_123',
            'user_id': 'user_456',
            'event_type': 'page_view',
            'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)'
        }

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([event])
                | beam.ParDo(self.enricher)
            )

            def check_device_type(results):
                assert len(results) == 1
                assert results[0]['device_type'] == 'mobile'

            assert_that(output, check_device_type)

    def test_device_type_detection_tablet(self):
        """Test tablet device detection from user agent."""
        event = {
            'event_id': 'evt_123',
            'user_id': 'user_456',
            'event_type': 'page_view',
            'user_agent': 'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X)'
        }

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([event])
                | beam.ParDo(self.enricher)
            )

            def check_device_type(results):
                assert len(results) == 1
                assert results[0]['device_type'] == 'tablet'

            assert_that(output, check_device_type)

    def test_device_type_detection_desktop(self):
        """Test desktop device detection from user agent."""
        event = {
            'event_id': 'evt_123',
            'user_id': 'user_456',
            'event_type': 'page_view',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([event])
                | beam.ParDo(self.enricher)
            )

            def check_device_type(results):
                assert len(results) == 1
                assert results[0]['device_type'] == 'desktop'

            assert_that(output, check_device_type)


class TestValidateDataQuality(unittest.TestCase):
    """Test data quality validation logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ValidateDataQuality()

    def test_valid_event_passes(self):
        """Test that valid events pass quality checks."""
        valid_event = {
            'user_id': 'user_12345',
            'session_id': 'sess_abcdefghij',
            'event_type': 'page_view',
            'event_id': 'evt_123'
        }

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([valid_event])
                | beam.ParDo(self.validator).with_outputs('errors', main='valid')
            )

            assert_that(output.valid, equal_to([valid_event]))

    def test_invalid_user_id_fails(self):
        """Test that invalid user_id fails quality check."""
        invalid_event = {
            'user_id': 'ab',  # Too short
            'session_id': 'sess_abcdefghij',
            'event_type': 'page_view',
            'event_id': 'evt_123'
        }

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([invalid_event])
                | beam.ParDo(self.validator).with_outputs('errors', main='valid')
            )

            assert_that(output.valid, equal_to([]))

    def test_invalid_event_type_fails(self):
        """Test that invalid event_type fails quality check."""
        invalid_event = {
            'user_id': 'user_12345',
            'session_id': 'sess_abcdefghij',
            'event_type': 'invalid_type',  # Not in allowed list
            'event_id': 'evt_123'
        }

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([invalid_event])
                | beam.ParDo(self.validator).with_outputs('errors', main='valid')
            )

            assert_that(output.valid, equal_to([]))

    def test_invalid_session_id_fails(self):
        """Test that invalid session_id fails quality check."""
        invalid_event = {
            'user_id': 'user_12345',
            'session_id': 'short',  # Too short
            'event_type': 'page_view',
            'event_id': 'evt_123'
        }

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([invalid_event])
                | beam.ParDo(self.validator).with_outputs('errors', main='valid')
            )

            assert_that(output.valid, equal_to([]))


class TestFormatForBigQuery(unittest.TestCase):
    """Test BigQuery formatting logic."""

    def test_format_complete_event(self):
        """Test formatting of complete event for BigQuery."""
        event = {
            'event_id': 'evt_123',
            'user_id': 'user_456',
            'session_id': 'sess_789',
            'event_type': 'page_view',
            'event_properties': {'page': '/home'},
            'device_type': 'desktop',
            'user_agent': 'Mozilla/5.0',
            'ip_address': '192.168.1.1',
            'event_timestamp': '2024-01-15T10:30:00Z'
        }

        formatter = FormatForBigQuery()

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([event])
                | beam.ParDo(formatter)
            )

            def check_format(results):
                assert len(results) == 1
                formatted = results[0]

                # Check all required fields present
                assert 'event_id' in formatted
                assert 'user_id' in formatted
                assert 'session_id' in formatted
                assert 'event_type' in formatted
                assert 'event_timestamp' in formatted

                # Check event_properties is JSON string
                assert isinstance(formatted['event_properties'], str)
                parsed = json.loads(formatted['event_properties'])
                assert parsed['page'] == '/home'

            assert_that(output, check_format)

    def test_format_minimal_event(self):
        """Test formatting of minimal event with optional fields missing."""
        event = {
            'event_id': 'evt_123',
            'user_id': 'user_456',
            'session_id': 'sess_789',
            'event_type': 'page_view',
            'event_timestamp': '2024-01-15T10:30:00Z'
        }

        formatter = FormatForBigQuery()

        with TestPipeline() as p:
            output = (
                p
                | beam.Create([event])
                | beam.ParDo(formatter)
            )

            def check_format(results):
                assert len(results) == 1
                formatted = results[0]

                # Required fields present
                assert formatted['event_id'] == 'evt_123'
                assert formatted['user_id'] == 'user_456'

                # Optional fields handled correctly
                assert formatted.get('device_type') is None
                assert formatted.get('user_agent') is None

            assert_that(output, check_format)


if __name__ == '__main__':
    unittest.main()
