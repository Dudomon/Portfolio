"""
Generate sample data for testing the data engineering platform locally.

This script creates realistic sample data for:
- Transaction records
- User events
- Test scenarios for data quality validation
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict
import argparse


class SampleDataGenerator:
    """Generate realistic sample data for testing."""

    CURRENCIES = ['USD', 'EUR', 'GBP', 'BRL']
    TRANSACTION_TYPES = ['purchase', 'refund', 'chargeback', 'adjustment']
    EVENT_TYPES = ['page_view', 'click', 'form_submit', 'purchase', 'add_to_cart']
    DEVICE_TYPES = ['mobile', 'desktop', 'tablet']

    def __init__(self, num_users: int = 100, num_merchants: int = 20):
        """
        Initialize generator.

        Args:
            num_users: Number of unique users to generate
            num_merchants: Number of unique merchants
        """
        self.num_users = num_users
        self.num_merchants = num_merchants
        self.user_ids = [f"user_{uuid.uuid4().hex[:8]}" for _ in range(num_users)]
        self.merchant_ids = [f"merchant_{uuid.uuid4().hex[:8]}" for _ in range(num_merchants)]

    def generate_transaction(self, timestamp: datetime) -> Dict:
        """
        Generate a single transaction record.

        Args:
            timestamp: Transaction timestamp

        Returns:
            Transaction dictionary
        """
        transaction_type = random.choice(self.TRANSACTION_TYPES)

        # Refunds are typically smaller amounts
        if transaction_type == 'refund':
            amount = round(random.uniform(10, 200), 2)
        else:
            amount = round(random.uniform(5, 2000), 2)

        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'user_id': random.choice(self.user_ids),
            'transaction_type': transaction_type,
            'amount': amount,
            'currency': random.choice(self.CURRENCIES),
            'merchant_id': random.choice(self.merchant_ids) if random.random() > 0.1 else None,
            'metadata': json.dumps({
                'payment_method': random.choice(['credit_card', 'debit_card', 'paypal', 'bank_transfer']),
                'category': random.choice(['electronics', 'clothing', 'food', 'services', 'other'])
            }),
            'transaction_timestamp': timestamp.isoformat(),
            'ingestion_timestamp': datetime.utcnow().isoformat()
        }

        return transaction

    def generate_user_event(self, timestamp: datetime) -> Dict:
        """
        Generate a single user event record.

        Args:
            timestamp: Event timestamp

        Returns:
            User event dictionary
        """
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        event_type = random.choice(self.EVENT_TYPES)
        device_type = random.choice(self.DEVICE_TYPES)

        user_agents = {
            'mobile': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15',
            'desktop': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'tablet': 'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15'
        }

        event = {
            'event_id': str(uuid.uuid4()),
            'user_id': random.choice(self.user_ids),
            'session_id': session_id,
            'event_type': event_type,
            'event_properties': json.dumps({
                'page_url': f'/page/{random.randint(1, 100)}',
                'referrer': random.choice(['google', 'facebook', 'direct', 'email'])
            }),
            'device_type': device_type,
            'user_agent': user_agents[device_type],
            'ip_address': f'{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}',
            'event_timestamp': timestamp.isoformat()
        }

        return event

    def generate_transactions_batch(
        self,
        count: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Generate a batch of transactions.

        Args:
            count: Number of transactions to generate
            start_date: Start of time range
            end_date: End of time range

        Returns:
            List of transaction dictionaries
        """
        transactions = []
        time_range = (end_date - start_date).total_seconds()

        for _ in range(count):
            random_seconds = random.randint(0, int(time_range))
            timestamp = start_date + timedelta(seconds=random_seconds)
            transactions.append(self.generate_transaction(timestamp))

        return transactions

    def generate_events_batch(
        self,
        count: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Generate a batch of user events.

        Args:
            count: Number of events to generate
            start_date: Start of time range
            end_date: End of time range

        Returns:
            List of event dictionaries
        """
        events = []
        time_range = (end_date - start_date).total_seconds()

        for _ in range(count):
            random_seconds = random.randint(0, int(time_range))
            timestamp = start_date + timedelta(seconds=random_seconds)
            events.append(self.generate_user_event(timestamp))

        return events

    def generate_data_quality_test_cases(self) -> Dict[str, List[Dict]]:
        """
        Generate test cases for data quality validation.

        Returns:
            Dictionary with test case categories
        """
        now = datetime.utcnow()

        test_cases = {
            'valid_transactions': self.generate_transactions_batch(10, now - timedelta(hours=1), now),
            'invalid_transactions': [
                # Missing required field
                {
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': None,  # NULL user_id
                    'transaction_type': 'purchase',
                    'amount': 100.00,
                    'currency': 'USD',
                    'transaction_timestamp': now.isoformat()
                },
                # Negative amount
                {
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': random.choice(self.user_ids),
                    'transaction_type': 'purchase',
                    'amount': -50.00,  # Negative amount
                    'currency': 'USD',
                    'transaction_timestamp': now.isoformat()
                },
                # Invalid currency
                {
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': random.choice(self.user_ids),
                    'transaction_type': 'purchase',
                    'amount': 100.00,
                    'currency': 'INVALID',  # Invalid currency code
                    'transaction_timestamp': now.isoformat()
                },
                # Invalid transaction type
                {
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': random.choice(self.user_ids),
                    'transaction_type': 'invalid_type',  # Invalid type
                    'amount': 100.00,
                    'currency': 'USD',
                    'transaction_timestamp': now.isoformat()
                }
            ]
        }

        return test_cases


def save_to_json_files(data: Dict[str, List[Dict]], output_dir: str):
    """
    Save generated data to JSON files.

    Args:
        data: Dictionary of data categories
        output_dir: Output directory path
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    for category, records in data.items():
        filename = os.path.join(output_dir, f"{category}.json")
        with open(filename, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} records to {filename}")


def save_to_ndjson(records: List[Dict], output_file: str):
    """
    Save records to newline-delimited JSON format.

    Args:
        records: List of record dictionaries
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    print(f"Saved {len(records)} records to {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate sample data for testing')
    parser.add_argument('--num-transactions', type=int, default=1000,
                        help='Number of transactions to generate')
    parser.add_argument('--num-events', type=int, default=5000,
                        help='Number of user events to generate')
    parser.add_argument('--num-users', type=int, default=100,
                        help='Number of unique users')
    parser.add_argument('--num-merchants', type=int, default=20,
                        help='Number of unique merchants')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days of historical data')
    parser.add_argument('--output-dir', default='./sample_data',
                        help='Output directory for generated files')
    parser.add_argument('--format', choices=['json', 'ndjson'], default='ndjson',
                        help='Output format')

    args = parser.parse_args()

    # Initialize generator
    generator = SampleDataGenerator(
        num_users=args.num_users,
        num_merchants=args.num_merchants
    )

    # Generate data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    print(f"Generating {args.num_transactions} transactions...")
    transactions = generator.generate_transactions_batch(
        args.num_transactions,
        start_date,
        end_date
    )

    print(f"Generating {args.num_events} user events...")
    events = generator.generate_events_batch(
        args.num_events,
        start_date,
        end_date
    )

    print("Generating data quality test cases...")
    test_cases = generator.generate_data_quality_test_cases()

    # Save data
    if args.format == 'json':
        data = {
            'transactions': transactions,
            'user_events': events,
            **test_cases
        }
        save_to_json_files(data, args.output_dir)
    else:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        save_to_ndjson(transactions, os.path.join(args.output_dir, 'transactions.ndjson'))
        save_to_ndjson(events, os.path.join(args.output_dir, 'user_events.ndjson'))
        save_to_json_files(test_cases, os.path.join(args.output_dir, 'test_cases'))

    print("\nData generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"\nSummary:")
    print(f"  Transactions: {len(transactions)}")
    print(f"  User Events: {len(events)}")
    print(f"  Test Cases: {sum(len(v) for v in test_cases.values())}")


if __name__ == '__main__':
    main()
