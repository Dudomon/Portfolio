import time
import pandas as pd
from pescador import PescadorEnv
import numpy as np


def load_df():
    # Align with silus loader
    path = 'data/GC=F_YAHOO_20250821_161220.csv'
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['time'])
    df.set_index('timestamp', inplace=True)
    df = df.rename(columns={
        'open': 'open_5m',
        'high': 'high_5m',
        'low': 'low_5m',
        'close': 'close_5m',
        'tick_volume': 'volume_5m'
    })
    return df


def run_sim(steps=10000):
    df = load_df()
    env = PescadorEnv(df, window_size=20, is_training=True, initial_balance=500)
    obs = env.reset()
    total_reward = 0.0
    attempts = 0
    passes = 0
    for i in range(steps):
        # Try to enter frequently: alternate LONG/SHORT decisions
        entry = 1.0 if (i % 2 == 0) else 2.0
        action = np.array([entry, 0.9, 0.0, 0.0], dtype=np.float32)
        prev_attempts = len(env._gate_recent_attempts)
        obs, rew, done, info = env.step(action)
        total_reward += float(rew)
        if len(env._gate_recent_attempts) > prev_attempts:
            attempts += 1
            passes += 1 if env._gate_recent_attempts[-1] else 0
        if done:
            obs = env.reset()
    pr = env.get_pescador_pass_rate()
    print(f"Sim steps={steps} | reward_sum={total_reward:.3f} | attempts={attempts} | pass={passes} | pass_rate_window={pr*100:.1f}% | current_gate: slope>={env.gate_min_slope:.3f}, atr>={env.gate_min_atr:.3f}, sess={env.gate_session_start}-{env.gate_session_end}")


if __name__ == '__main__':
    run_sim(steps=10000)

