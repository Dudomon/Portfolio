import sys
import torch
import gym
from stable_baselines3.common.utils import get_schedule_fn
from trading_framework.policies.two_head_v7_simple import TwoHeadV7Simple, get_v7_kwargs, _validate_v7_policy

# Função utilitária para filtrar não-ASCII
def ascii_only(text):
    return text.encode('ascii', errors='ignore').decode('ascii')


def main():
    print(ascii_only("[TESTE] Iniciando teste dos Gates V7Simple..."))
    try:
        # Inicializar policy com kwargs padrão
        kwargs = get_v7_kwargs()
        features_dim = kwargs['v7_features_dim']
        lstm_hidden_size = kwargs.get('v7_shared_lstm_hidden', 256)
        # Espaços dummy para teste
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(features_dim,), dtype=float)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=float)  # Ajuste se necessário
        lr_schedule = get_schedule_fn(1e-4)
        policy = TwoHeadV7Simple(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )
        # Capturar prints da validação e filtrar
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _validate_v7_policy(policy)
        print(ascii_only(buf.getvalue()))

        # Corrigir shape do dummy_input para bater com observation_space
        dummy_input = torch.randn(1, *observation_space.shape)
        # Inicializar lstm_states corretamente (h, c) para batch size 1
        lstm_states = (
            torch.zeros(1, lstm_hidden_size),  # h
            torch.zeros(1, lstm_hidden_size)   # c
        )
        episode_starts = torch.ones(1, dtype=torch.bool)

        # Forward actor para garantir funcionamento
        with torch.no_grad():
            print(ascii_only("Executando forward_actor..."))
            try:
                actor_out, new_lstm_states, gate_info = policy.forward_actor(dummy_input, lstm_states, episode_starts)
                print(ascii_only(f"actor_out type: {type(actor_out)}, shape: {getattr(actor_out, 'shape', 'N/A')}"))
                print(ascii_only(f"new_lstm_states type: {type(new_lstm_states)}"))
                print(ascii_only(f"gate_info type: {type(gate_info)}"))
                if actor_out is None:
                    print(ascii_only("[ERRO] actor_out retornou None!"))
                    sys.exit(1)
                if new_lstm_states is None:
                    print(ascii_only("[ERRO] new_lstm_states retornou None!"))
                    sys.exit(1)
                if gate_info is None:
                    print(ascii_only("[ERRO] gate_info retornou None!"))
                    sys.exit(1)
                if hasattr(actor_out, 'shape'):
                    print(ascii_only(f"[OK] Forward actor executado com sucesso. Saida shape: {actor_out.shape}"))
                else:
                    print(ascii_only("[ERRO] actor_out nao tem atributo shape!"))
                    sys.exit(1)
            except Exception as e:
                import traceback
                print(ascii_only(f"[DEBUG] Erro detalhado: {e}"))
                print(ascii_only(f"[DEBUG] Traceback: {traceback.format_exc()}"))
                raise

        print(ascii_only("[SUCESSO] Todos os testes dos Gates V7Simple passaram!"))
        sys.exit(0)
    except Exception as e:
        print(ascii_only(f"[ERRO] Falha no teste dos Gates V7Simple: {e}"))
        sys.exit(1)

if __name__ == "__main__":
    main()