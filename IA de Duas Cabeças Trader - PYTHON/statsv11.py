#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StatsV11 - Visualizador simples dos componentes de reward do V4-Selective

Lê os arquivos JSONL gerados pelo RealTimeLogger (padrão em D:/Projeto/avaliacoes)
e plota séries de tempo de métricas como quality_score, entry_confidence e flags de padrões.

Dependências: pandas, matplotlib
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_reward_jsonl(folder: str) -> pd.DataFrame:
    base = Path(folder)
    files = sorted(base.glob('rewards_*.jsonl'))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo rewards_*.jsonl em {folder}")
    # Pega o mais recente
    path = files[-1]
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get('type') in ('reward_info', 'rewards_metrics'):
                    rows.append(obj)
            except Exception:
                continue
    if not rows:
        raise RuntimeError("Nenhuma linha de reward encontrada no JSONL")
    df = pd.DataFrame(rows)
    return df


def plot_reward_components(df: pd.DataFrame, out: str | None = None):
    # Selecionar colunas relevantes se existirem
    cols = []
    for c in [
        'step', 'base_reward', 'quality_score', 'entry_confidence', 'position_time_ratio',
        'ma_cross_long', 'ma_cross_short', 'double_bottom', 'double_top'
    ]:
        if c in df.columns:
            cols.append(c)
    d = df[cols].copy()
    d.sort_values('step', inplace=True)

    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    if 'base_reward' in d:
        d.plot(x='step', y='base_reward', ax=ax1, label='base_reward', color='tab:blue')
    if 'quality_score' in d:
        d.plot(x='step', y='quality_score', ax=ax1, label='quality_score', color='tab:green')
    if 'entry_confidence' in d:
        d.plot(x='step', y='entry_confidence', ax=ax1, label='entry_confidence', color='tab:orange')
    if 'position_time_ratio' in d:
        d.plot(x='step', y='position_time_ratio', ax=ax1, label='position_time_ratio', color='tab:red')

    # Flags como pontos
    if 'ma_cross_long' in d:
        m = d[d['ma_cross_long'] == True]
        plt.scatter(m['step'], [0]*len(m), marker='^', color='green', alpha=0.6, label='ma_cross_long')
    if 'ma_cross_short' in d:
        m = d[d['ma_cross_short'] == True]
        plt.scatter(m['step'], [0]*len(m), marker='v', color='red', alpha=0.6, label='ma_cross_short')
    if 'double_bottom' in d:
        m = d[d['double_bottom'] == True]
        plt.scatter(m['step'], [0]*len(m), marker='o', facecolors='none', edgecolors='green', alpha=0.6, label='double_bottom')
    if 'double_top' in d:
        m = d[d['double_top'] == True]
        plt.scatter(m['step'], [0]*len(m), marker='o', facecolors='none', edgecolors='red', alpha=0.6, label='double_top')

    plt.title('V4-Selective: Reward e Padrões ao longo do treino')
    plt.xlabel('Step')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if out:
        plt.savefig(out, bbox_inches='tight')
        print(f"Gráfico salvo em: {out}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', default='D:/Projeto/avaliacoes', help='Pasta com rewards_*.jsonl')
    ap.add_argument('--out', default=None, help='Se fornecido, salva em arquivo PNG')
    args = ap.parse_args()

    df = load_reward_jsonl(args.folder)
    plot_reward_components(df, args.out)


if __name__ == '__main__':
    main()
