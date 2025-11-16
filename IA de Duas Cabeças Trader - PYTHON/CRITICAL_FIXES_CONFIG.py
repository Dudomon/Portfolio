
# 🔧 CRITICAL FIXES FOR ZERO WEIGHTS PROBLEM
# Apply these changes immediately to prevent recurrence:

LEARNING_RATE_FIX = {
    "current": 5e-3,           # PROBLEM: Too high, causing weight explosion
    "fixed": 1e-4,             # SOLUTION: 50x reduction to stable range
    "rationale": "High LR causes weight explosion → clipping → zeros"
}

GRADIENT_CLIPPING_FIX = {
    "recommended": {
        "max_grad_norm": 1.0,   # Gentle clipping to prevent explosion
        "clip_grad_value": None # Don't use value clipping
    },
    "avoid": {
        "max_grad_norm": 0.1,   # Too aggressive, causes zeros
        "clip_grad_value": 0.5  # Value clipping can zero weights
    }
}

OPTIMIZER_SETTINGS = {
    "type": "Adam",
    "lr": 1e-4,                # Critical: Use fixed learning rate
    "betas": (0.9, 0.999),     # Standard Adam betas
    "eps": 1e-8,               # Standard epsilon
    "weight_decay": 1e-5,      # Light regularization only
    "amsgrad": False           # Keep simple
}

INITIALIZATION_CHECK = {
    "bias_initialization": {
        "lstm_forget_bias": 1.0,     # Forget gate bias = 1
        "gru_biases": "uniform(-0.01, 0.01)",
        "linear_biases": "uniform(-0.01, 0.01)",
        "layernorm_biases": 0.0
    },
    "weight_initialization": {
        "lstm_weights": "xavier_uniform + orthogonal",
        "linear_weights": "xavier_uniform", 
        "embedding_weights": "normal(0, 0.01)"
    }
}

MONITORING_REQUIRED = [
    "gradient_norms",           # Watch for explosion
    "weight_zero_percentages",  # Monitor for recurrence  
    "bias_magnitudes",          # Ensure biases stay healthy
    "loss_stability"            # Check for training instability
]
    