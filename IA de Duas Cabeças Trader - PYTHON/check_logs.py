#!/usr/bin/env python3
import json
import os

log_file = "D:/Projeto/avaliacoes/performance_20250819_111931_73748_4261d2f5.jsonl"

print("📊 ÚLTIMOS 10 SNAPSHOTS DE PERFORMANCE:")
print("=" * 70)

with open(log_file, 'r') as f:
    lines = [line for line in f.readlines()[1:] if json.loads(line).get('type') == 'performance_metrics']
    
    for line in lines[-10:]:
        data = json.loads(line)
        step = data['step']
        portfolio = data['portfolio_value']
        dd = data['drawdown']
        trades = data['trades_count']
        winrate = data['win_rate'] * 100
        
        print(f"Step {step:>6}: Portfolio=${portfolio:>7.2f}, DD={dd:>5.1f}%, Trades={trades:>3}, WinRate={winrate:>4.1f}%")

# Training metrics
training_file = "D:/Projeto/avaliacoes/training_20250819_111931_73748_4261d2f5.jsonl" 
print(f"\n📈 TRAINING METRICS:")
print("=" * 70)

with open(training_file, 'r') as f:
    lines = [line for line in f.readlines()[1:] if json.loads(line).get('type') == 'training_step']
    
    for line in lines[-5:]:
        data = json.loads(line)
        step = data['step']
        loss = data['loss']
        policy_loss = data['policy_loss']
        value_loss = data['value_loss'] 
        entropy_loss = data['entropy_loss']
        explained_var = data['explained_variance']
        
        print(f"Step {step:>6}: Loss={loss:>8.3f}, Policy={policy_loss:>7.3f}, Value={value_loss:>7.3f}, ExpVar={explained_var:>5.1%}")