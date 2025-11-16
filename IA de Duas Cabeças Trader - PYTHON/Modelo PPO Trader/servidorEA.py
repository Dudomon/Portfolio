#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor Flask para comunicação com Expert Advisor do MetaTrader 5
Recebe dados do RobotV3.py e serve análise técnica profissional para o EA
"""

from flask import Flask, request, jsonify
import json
import os
import logging
import time
from datetime import datetime, timedelta
import numpy as np

# Configuração do Flask
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Configuração de logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ea_webhook.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Armazenar dados recebidos do robô
robot_data = {
    'action': 'HOLD',
    'step': 0,
    'timestamp': datetime.now().isoformat(),
    'symbol': 'GOLD',
    'price': 3330.0,
    'confidence': 0.5,
    'rsi': 50.0,
    'bb_position': 0.5,
    'volatility': 0.01,
    'momentum': 0.0,
    'trend': 'LATERAL',
    'portfolio_value': 500.0,
    'positions': [],
    'status': 'active'
}

# Histórico de preços para análise
price_history = []
MAX_HISTORY = 50

def calculate_breakeven(current_price, positions_data, portfolio_value):
    """Calcular breakeven baseado nas posições atuais"""
    if not positions_data or len(positions_data) == 0:
        # Se não tem posições, usar análise técnica
        return current_price  # Breakeven = preço atual
    
    total_volume = 0
    weighted_price = 0
    
    for pos in positions_data:
        if isinstance(pos, dict):
            volume = pos.get('volume', 0.01)
            price = pos.get('price', current_price)
            total_volume += volume
            weighted_price += price * volume
    
    if total_volume > 0:
        return weighted_price / total_volume
    else:
        return current_price

def calculate_target_prediction(current_price, action, confidence, rsi, bb_position, volatility, momentum):
    """Calcular previsão de preço e direção baseado na análise do modelo"""
    
    # Calcular amplitude baseada na volatilidade
    amplitude = max(10.0, volatility * 1000)  # Mínimo 10 pontos
    
    # Determinar direção baseada na ação do modelo
    if action == 'BUY' or action == 'LONG':
        direction = "UP"
        # Alvo baseado na confiança e RSI
        target_multiplier = 1.0 + (confidence * 0.5)  # 1.0 a 1.5
        if rsi < 30:  # RSI oversold = maior potencial de alta
            target_multiplier *= 1.3
        target_price = current_price + (amplitude * target_multiplier)
        
    elif action == 'SELL' or action == 'SHORT':
        direction = "DOWN"
        # Alvo baseado na confiança e RSI
        target_multiplier = 1.0 + (confidence * 0.5)  # 1.0 a 1.5
        if rsi > 70:  # RSI overbought = maior potencial de queda
            target_multiplier *= 1.3
        target_price = current_price - (amplitude * target_multiplier)
        
    else:  # HOLD
        direction = "SIDEWAYS"
        # Alvo = breakeven ou preço atual
        target_price = current_price
        
    return target_price, direction

def detect_formations(price_history, current_price):
    """Detectar formações gráficas baseado no histórico de preços"""
    if len(price_history) < 10:
        return []
    
    # Pegar últimos 20 preços
    recent_prices = price_history[-20:] if len(price_history) >= 20 else price_history
    
    # Encontrar máximas e mínimas locais
    highs = []
    lows = []
    
    for i in range(1, len(recent_prices) - 1):
        if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
            highs.append((i, recent_prices[i]))
        elif recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
            lows.append((i, recent_prices[i]))
    
    # Tentar detectar formações
    current_time = int(time.time())
    
    # Triângulo (padrão mais comum)
    if len(highs) >= 2 and len(lows) >= 2:
        # Pegar os 2 últimos topos e 2 últimos fundos
        last_high = highs[-1]
        prev_high = highs[-2]
        last_low = lows[-1]
        
        return [
            'triangle',
            current_time - (len(recent_prices) - prev_high[0]) * 60,  # Tempo do primeiro ponto
            prev_high[1],  # Preço do primeiro ponto
            current_time - (len(recent_prices) - last_high[0]) * 60,  # Tempo do segundo ponto
            last_high[1],  # Preço do segundo ponto
            current_time - (len(recent_prices) - last_low[0]) * 60,   # Tempo do terceiro ponto
            last_low[1]    # Preço do terceiro ponto
        ]
    
    # Fundo duplo
    if len(lows) >= 2:
        last_low = lows[-1]
        prev_low = lows[-2]
        
        # Verificar se os fundos são similares (diferença < 5%)
        if abs(last_low[1] - prev_low[1]) / prev_low[1] < 0.05:
            return [
                'double_bottom',
                current_time - (len(recent_prices) - prev_low[0]) * 60,
                prev_low[1],
                current_time - (len(recent_prices) - last_low[0]) * 60,
                last_low[1],
                current_time,
                current_price
            ]
    
    # Topo duplo
    if len(highs) >= 2:
        last_high = highs[-1]
        prev_high = highs[-2]
        
        # Verificar se os topos são similares (diferença < 5%)
        if abs(last_high[1] - prev_high[1]) / prev_high[1] < 0.05:
            return [
                'double_top',
                current_time - (len(recent_prices) - prev_high[0]) * 60,
                prev_high[1],
                current_time - (len(recent_prices) - last_high[0]) * 60,
                last_high[1],
                current_time,
                current_price
            ]
    
    return []

@app.route('/receber', methods=['POST'])
def receber_dados():
    """Recebe dados do RobotV3.py"""
    global robot_data, price_history
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Dados JSON inválidos'}), 400
        
        # Atualizar dados do robô
        robot_data.update(data)
        robot_data['timestamp'] = datetime.now().isoformat()
        
        # Atualizar histórico de preços
        current_price = data.get('price', robot_data['price'])
        price_history.append(current_price)
        
        # Manter apenas os últimos MAX_HISTORY preços
        if len(price_history) > MAX_HISTORY:
            price_history = price_history[-MAX_HISTORY:]
        
        # Log dos dados recebidos
        logging.info(f"📡 Dados recebidos: Action={data.get('action', 'N/A')}, Price={current_price}, Confidence={data.get('confidence', 'N/A')}")
        
        return jsonify({'status': 'success', 'message': 'Dados recebidos com sucesso'})
        
    except Exception as e:
        logging.error(f"❌ Erro ao processar dados: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/dados', methods=['GET'])
def obter_dados():
    """Serve dados de análise técnica profissional para o EA"""
    try:
        current_price = robot_data.get('price', 3330.0)
        action = robot_data.get('action', 'HOLD')
        confidence = robot_data.get('confidence', 0.5)
        rsi = robot_data.get('rsi', 50.0)
        bb_position = robot_data.get('bb_position', 0.5)
        volatility = robot_data.get('volatility', 0.01)
        momentum = robot_data.get('momentum', 0.0)
        positions = robot_data.get('positions', [])
        portfolio_value = robot_data.get('portfolio_value', 500.0)
        
        # 1. CALCULAR BREAKEVEN
        breakeven = calculate_breakeven(current_price, positions, portfolio_value)
        
        # 2. CALCULAR PREVISÃO DE PREÇO E DIREÇÃO
        target_price, target_direction = calculate_target_prediction(
            current_price, action, confidence, rsi, bb_position, volatility, momentum
        )
        
        # 3. DETECTAR FORMAÇÕES GRÁFICAS
        formations = detect_formations(price_history, current_price)
        
        # 4. MAPEAR SINAL DO MODELO
        signal_mapping = {
            'BUY': 'BUY',
            'LONG': 'BUY', 
            'SELL': 'SELL',
            'SHORT': 'SELL',
            'HOLD': 'HOLD'
        }
        signal = signal_mapping.get(action, 'HOLD')
        
        # 5. DADOS COMPLETOS PARA O EA
        ea_data = {
            # DADOS PRINCIPAIS QUE O EA PRECISA
            'breakeven': str(breakeven),
            'signal': signal,
            'target_price': str(target_price),
            'target_direction': target_direction,
            'formations': formations,
            
            # DADOS ADICIONAIS
            'price': current_price,
            'confidence': confidence,
            'rsi': rsi,
            'bb_position': bb_position,
            'volatility': volatility,
            'momentum': momentum,
            'timestamp': robot_data['timestamp'],
            'symbol': robot_data.get('symbol', 'GOLD'),
            'portfolio_value': portfolio_value,
            'positions_count': len(positions),
            
            # DADOS PARA COMPATIBILIDADE
            'action': action,
            'step': robot_data.get('step', 0),
            'status': robot_data.get('status', 'active')
        }
        
        logging.info(f"📊 EA Data: Signal={signal}, Breakeven={breakeven:.2f}, Target={target_price:.2f} {target_direction}, Formations={len(formations) > 0}")
        return jsonify(ea_data)
        
    except Exception as e:
        logging.error(f"❌ Erro ao gerar análise: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/status', methods=['GET'])
def status():
    """Status do servidor"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'robot_data': robot_data,
        'price_history_count': len(price_history)
    })

if __name__ == '__main__':
    print("🚀 Servidor Flask EA Profissional iniciado!")
    print("📡 Endpoint para receber dados: http://127.0.0.1:5000/receber")
    print("🎯 Endpoint para servir análise: http://127.0.0.1:5000/dados")
    print("📊 Endpoint de status: http://127.0.0.1:5000/status")
    print()
    print("🔧 DADOS SERVIDOS PARA O EA:")
    print("   • breakeven: Preço de equilíbrio calculado")
    print("   • signal: BUY/SELL/HOLD baseado na decisão do modelo")
    print("   • target_price: Preço alvo previsto")
    print("   • target_direction: UP/DOWN/SIDEWAYS")
    print("   • formations: Formações gráficas detectadas")
    
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        threaded=True
    ) 