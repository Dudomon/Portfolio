#!/usr/bin/env python3
"""
CropLink - Ponto de entrada da aplicação
========================================

Este arquivo inicializa e executa a aplicação CropLink.
"""

from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)