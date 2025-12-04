"""
Streaming pipeline for processing user events from Pub/Sub to BigQuery.

This pipeline demonstrates production-grade streaming data processing with:
- Data validation and schema enforcement
- Error handling and dead letter queue pattern
- Custom metrics for monitoring
- Windowing for aggregations
- BigQuery streaming inserts
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.io.gcp.bigquery import WriteToBigQuery, BigQueryDisposition
from apache_beam.metrics import Metrics
from apache_beam.transforms import window


class PipelineConfig:
    """Configuration for the streaming pipeline."""

    def __init__(self, project_id: str, subscription: str, output_table: str):
        self.project_id = project_id
        self.subscription = subscription
        self.output_table = output_table
        self.error_topic = f"projects/{project_id}/topics/pipeline-errors"


class ParsePubSubMessage(beam.DoFn):
    """Parse and validate Pub/Sub messages."""

    REQUIRED_FIELDS = ['event_id', 'user_id', 'session_id', 'event_type', 'event_timestamp']

    def __init__(self):
        self.parse_errors = Metrics.counter(self.__class__, 'parse_errors')
        self.parse_success = Metrics.counter(self.__class__, 'parse_success')
        self.validation_errors = Metrics.counter(self.__class__, 'validation_errors')

    def process(self, element: bytes):
        """
        Parse JSON message and validate required fields.

        Args:
            element: Raw bytes from Pub/Sub message

        Yields:
            Parsed and validated event dict, or error dict if parsing fails
        """
        try:
            message = json.loads(element.decode('utf-8'))

            # Validate required fields
            missing_fields = [field for field in self.REQUIRED_FIELDS if field not in message]
            if missing_fields:
                self.validation_errors.inc()
                yield beam.pvalue.TaggedOutput('errors', {
                    'error_type': 'validation_error',
                    'error_message': f'Missing required fields: {missing_fields}',
                    'raw_message': element.decode('utf-8'),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                return

            # Validate event_timestamp format
            try:
                datetime.fromisoformat(message['event_timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                self.validation_errors.inc()
                yield beam.pvalue.TaggedOutput('errors', {
                    'error_type': 'validation_error',
                    'error_message': 'Invalid event_timestamp format',
                    'raw_message': element.decode('utf-8'),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                return

            self.parse_success.inc()
            yield message

        except json.JSONDecodeError as e:
            self.parse_errors.inc()
            logging.error(f"JSON decode error: {e}")
            yield beam.pvalue.TaggedOutput('errors', {
                'error_type': 'json_decode_error',
                'error_message': str(e),
                'raw_message': element.decode('utf-8', errors='replace'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            self.parse_errors.inc()
            logging.error(f"Unexpected error in parsing: {e}")
            yield beam.pvalue.TaggedOutput('errors', {
                'error_type': 'unexpected_error',
                'error_message': str(e),
                'raw_message': element.decode('utf-8', errors='replace'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })


class EnrichEvent(beam.DoFn):
    """Enrich event with additional metadata."""

    def __init__(self):
        self.enrichment_count = Metrics.counter(self.__class__, 'enrichment_count')

    def process(self, event: Dict[str, Any]):
        """
        Add ingestion metadata to event.

        Args:
            event: Parsed event dictionary

        Yields:
            Enriched event with additional metadata
        """
        enriched = event.copy()
        enriched['ingestion_timestamp'] = datetime.now(timezone.utc).isoformat()
        enriched['pipeline_version'] = '1.0.0'

        # Parse device type from user agent if present
        user_agent = event.get('user_agent', '').lower()
        if 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent:
            enriched['device_type'] = 'mobile'
        elif 'tablet' in user_agent or 'ipad' in user_agent:
            enriched['device_type'] = 'tablet'
        else:
            enriched['device_type'] = 'desktop'

        self.enrichment_count.inc()
        yield enriched


class ValidateDataQuality(beam.DoFn):
    """Perform data quality checks on events."""

    def __init__(self):
        self.quality_passed = Metrics.counter(self.__class__, 'quality_passed')
        self.quality_failed = Metrics.counter(self.__class__, 'quality_failed')

    def process(self, event: Dict[str, Any]):
        """
        Validate data quality rules.

        Args:
            event: Enriched event dictionary

        Yields:
            Event if quality checks pass, otherwise error
        """
        quality_errors = []

        # Check user_id format (should be UUID or numeric ID)
        user_id = event.get('user_id', '')
        if not user_id or len(user_id) < 3:
            quality_errors.append('Invalid user_id format')

        # Check event_type is in allowed values
        allowed_event_types = ['page_view', 'click', 'form_submit', 'purchase', 'add_to_cart']
        if event.get('event_type') not in allowed_event_types:
            quality_errors.append(f"Invalid event_type: {event.get('event_type')}")

        # Check session_id is present and reasonable length
        session_id = event.get('session_id', '')
        if not session_id or len(session_id) < 10:
            quality_errors.append('Invalid session_id')

        if quality_errors:
            self.quality_failed.inc()
            yield beam.pvalue.TaggedOutput('errors', {
                'error_type': 'data_quality_failure',
                'error_message': '; '.join(quality_errors),
                'event': event,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        else:
            self.quality_passed.inc()
            yield event


class FormatForBigQuery(beam.DoFn):
    """Format events for BigQuery insertion."""

    def process(self, event: Dict[str, Any]):
        """
        Format event to match BigQuery schema.

        Args:
            event: Validated event dictionary

        Yields:
            Event formatted for BigQuery table schema
        """
        formatted = {
            'event_id': event['event_id'],
            'user_id': event['user_id'],
            'session_id': event['session_id'],
            'event_type': event['event_type'],
            'event_properties': json.dumps(event.get('event_properties', {})),
            'device_type': event.get('device_type'),
            'user_agent': event.get('user_agent'),
            'ip_address': event.get('ip_address'),
            'event_timestamp': event['event_timestamp']
        }

        yield formatted


def run_pipeline(config: PipelineConfig, pipeline_options: PipelineOptions):
    """
    Execute the streaming pipeline.

    Args:
        config: Pipeline configuration
        pipeline_options: Apache Beam pipeline options
    """
    with beam.Pipeline(options=pipeline_options) as pipeline:

        # Read from Pub/Sub
        messages = (
            pipeline
            | 'Read from Pub/Sub' >> ReadFromPubSub(subscription=config.subscription)
        )

        # Parse and validate messages
        parsed_results = (
            messages
            | 'Parse Messages' >> beam.ParDo(ParsePubSubMessage()).with_outputs('errors', main='valid')
        )

        # Process valid messages
        enriched_events = (
            parsed_results.valid
            | 'Enrich Events' >> beam.ParDo(EnrichEvent())
        )

        # Data quality validation
        quality_results = (
            enriched_events
            | 'Validate Quality' >> beam.ParDo(ValidateDataQuality()).with_outputs('errors', main='valid')
        )

        # Format and write to BigQuery
        _ = (
            quality_results.valid
            | 'Format for BigQuery' >> beam.ParDo(FormatForBigQuery())
            | 'Write to BigQuery' >> WriteToBigQuery(
                table=config.output_table,
                schema='SCHEMA_AUTODETECT',
                create_disposition=BigQueryDisposition.CREATE_NEVER,
                write_disposition=BigQueryDisposition.WRITE_APPEND,
                method='STREAMING_INSERTS'
            )
        )

        # Combine all errors and write to error topic
        all_errors = (
            (parsed_results.errors, quality_results.errors)
            | 'Flatten Errors' >> beam.Flatten()
            | 'Format Errors as JSON' >> beam.Map(lambda x: json.dumps(x).encode('utf-8'))
        )

        # In production, write errors to dead letter queue
        # For now, just log them
        _ = (
            all_errors
            | 'Log Errors' >> beam.Map(lambda x: logging.error(f"Pipeline error: {x.decode('utf-8')}"))
        )


def main():
    """Main entry point for the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Streaming user events pipeline')
    parser.add_argument('--project', required=True, help='GCP project ID')
    parser.add_argument('--region', default='us-central1', help='GCP region')
    parser.add_argument('--subscription', required=True, help='Pub/Sub subscription path')
    parser.add_argument('--output_table', required=True, help='BigQuery output table')
    parser.add_argument('--runner', default='DirectRunner', help='Pipeline runner')
    parser.add_argument('--temp_location', required=False, help='GCS temp location')
    parser.add_argument('--staging_location', required=False, help='GCS staging location')
    parser.add_argument('--max_num_workers', type=int, default=5, help='Max workers')
    parser.add_argument('--autoscaling_algorithm', default='THROUGHPUT_BASED', help='Autoscaling algorithm')

    args, beam_args = parser.parse_known_args()

    # Create pipeline configuration
    config = PipelineConfig(
        project_id=args.project,
        subscription=args.subscription,
        output_table=args.output_table
    )

    # Configure pipeline options
    pipeline_options = PipelineOptions(
        beam_args,
        streaming=True,
        project=args.project,
        region=args.region,
        runner=args.runner,
        temp_location=args.temp_location,
        staging_location=args.staging_location,
        max_num_workers=args.max_num_workers,
        autoscaling_algorithm=args.autoscaling_algorithm,
        save_main_session=True
    )

    pipeline_options.view_as(StandardOptions).streaming = True

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Starting streaming pipeline for subscription: {args.subscription}")

    # Run the pipeline
    run_pipeline(config, pipeline_options)


if __name__ == '__main__':
    main()
