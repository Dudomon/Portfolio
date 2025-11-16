#!/usr/bin/env python3
"""
Quick convergence status check - single run
"""

import os
from datetime import datetime

def quick_check():
    """Quick convergence status check"""
    print("=" * 60)
    print("    QUICK CONVERGENCE CHECK")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Find latest debug file
    debug_files = [f for f in os.listdir('.') if f.startswith('debug_zeros_report_step_') and f.endswith('.txt')]
    
    if not debug_files:
        print("No debug files found - training not started yet")
        return
    
    latest_debug = sorted(debug_files, key=lambda x: int(x.split('_')[4].split('.')[0]))[-1]
    step = int(latest_debug.split('_')[4].split('.')[0])
    
    print(f"Latest data from: {latest_debug}")
    print(f"Training step: {step:,}")
    print()
    
    # Read and parse debug file
    try:
        with open(latest_debug, 'r', encoding='utf-8') as f:
            content = f.read()
        
        gradient_zeros = 0.0
        alert_count = 0
        
        for line in content.split('\n'):
            if 'Recent avg zeros:' in line:
                try:
                    gradient_zeros = float(line.split('Recent avg zeros: ')[1].split('%')[0])
                except:
                    pass
            if 'Alert count:' in line:
                try:
                    alert_count = int(line.split('Alert count: ')[1])
                except:
                    pass
        
        # Status assessment
        print("GRADIENT HEALTH:")
        print("-" * 20)
        
        if gradient_zeros < 2.0:
            status = "EXCELLENT"
            status_icon = "[+++]"
        elif gradient_zeros < 5.0:
            status = "HEALTHY" 
            status_icon = "[++]"
        elif gradient_zeros < 10.0:
            status = "WARNING"
            status_icon = "[+]"
        else:
            status = "CRITICAL"
            status_icon = "[!!!]"
        
        print(f"Gradient Zeros: {gradient_zeros:.2f}% {status_icon} {status}")
        print(f"Alert Count: {alert_count}")
        
        if alert_count == 0:
            print("Alert Status: [OK] NO ACTIVE ALERTS")
        else:
            print(f"Alert Status: [!] {alert_count} ACTIVE ALERTS")
            
        print()
        
        # Recommendations
        print("RECOMMENDATIONS:")
        print("-" * 20)
        
        if gradient_zeros < 5.0:
            print("+ System is healthy - continue training")
        elif gradient_zeros < 10.0:
            print("! Monitor closely - consider intervention if worsens")
        else:
            print("!!! Gradient death detected - check transformer fix")
            
        if step < 10000:
            print("i Early training - gradients still stabilizing")
        elif step > 50000:
            print("i Extended training - system should be stable")
            
    except Exception as e:
        print(f"Error reading debug file: {e}")
    
    print("=" * 60)
    print("For continuous monitoring, run: python convergence_monitor_working.py")
    print("=" * 60)

if __name__ == "__main__":
    quick_check()