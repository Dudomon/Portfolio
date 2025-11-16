#!/usr/bin/env python3
"""
🔍 FIND MAGIC NUMBERS IN ROBOT LOGS
Localiza logs específicos por magic number
"""

import os
import glob
import re
from datetime import datetime

# Magic numbers to find
MAGIC_NUMBERS = {
    '777844': 'Seventeen SEM filtro de horário',
    '777712': 'Seventeen com filtro ANTIGO',
    '777344': 'Seventeen com filtro ATUAL',
    '777528': 'Eighteen SEM filtro'
}

log_dir = "D:/Projeto/Modelo PPO Trader/logs"

print("=" * 100)
print("🔍 SEARCHING FOR MAGIC NUMBERS IN ROBOT LOGS")
print("=" * 100)
print()

# Get all log files
log_files = glob.glob(f"{log_dir}/trading_session_*.txt")
print(f"📂 Found {len(log_files)} total log files")
print()

# Search for each magic number
results = {}

for magic, description in MAGIC_NUMBERS.items():
    print(f"🔎 Searching for {magic} ({description})...")

    found_files = []

    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)  # Read first 5000 chars

                # Search for magic number pattern
                if magic in content or f"magic={magic}" in content or f"Magic: {magic}" in content:
                    # Get file stats
                    file_size = os.path.getsize(log_file) / 1024  # KB
                    mod_time = os.path.getmtime(log_file)
                    mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')

                    found_files.append({
                        'path': log_file,
                        'name': os.path.basename(log_file),
                        'size_kb': file_size,
                        'mod_date': mod_date,
                        'mod_timestamp': mod_time
                    })
        except Exception as e:
            pass

    if found_files:
        # Sort by modification time (most recent first)
        found_files.sort(key=lambda x: x['mod_timestamp'], reverse=True)

        print(f"   ✅ Found {len(found_files)} log(s)")
        for f in found_files[:5]:  # Show max 5 most recent
            print(f"      - {f['name']} ({f['size_kb']:.1f} KB) - {f['mod_date']}")

        results[magic] = found_files
    else:
        print(f"   ❌ Not found")

    print()

# Summary
print("=" * 100)
print("📊 SUMMARY")
print("=" * 100)

for magic, description in MAGIC_NUMBERS.items():
    if magic in results and results[magic]:
        most_recent = results[magic][0]
        print(f"\n✅ {description} (Magic: {magic})")
        print(f"   📁 File: {most_recent['name']}")
        print(f"   📏 Size: {most_recent['size_kb']:.1f} KB")
        print(f"   📅 Date: {most_recent['mod_date']}")
        print(f"   📂 Path: {most_recent['path']}")
    else:
        print(f"\n❌ {description} (Magic: {magic}) - NOT FOUND")

print()
print("=" * 100)

# Save results to file
output_file = "D:/Projeto/magic_numbers_logs.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("MAGIC NUMBERS LOGS\n")
    f.write("=" * 100 + "\n\n")

    for magic, description in MAGIC_NUMBERS.items():
        f.write(f"{description} (Magic: {magic})\n")
        if magic in results and results[magic]:
            most_recent = results[magic][0]
            f.write(f"Path: {most_recent['path']}\n")
        else:
            f.write("NOT FOUND\n")
        f.write("\n")

print(f"💾 Results saved to: {output_file}")
