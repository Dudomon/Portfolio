#!/usr/bin/env python3
"""
Memória Watchdog Callback
-------------------------

Callback utilitário para monitorar o uso de RAM e VRAM durante o treinamento.
Ajuda a detectar vazamentos e aciona limpezas leves (gc / empty_cache)
quando os limiares configurados são ultrapassados.
"""

from dataclasses import dataclass
from typing import Optional
import os
import gc
import time

import psutil
import torch
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class MemorySnapshot:
    """Representa um ponto de medição de recursos."""
    timestamp: float
    ram_gb: float
    ram_percent: float
    gpu_alloc_gb: float
    gpu_reserved_gb: float
    gpu_total_gb: float

    @property
    def gpu_alloc_frac(self) -> float:
        if self.gpu_total_gb <= 0:
            return 0.0
        return self.gpu_alloc_gb / self.gpu_total_gb

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "ram_gb": self.ram_gb,
            "ram_percent": self.ram_percent,
            "gpu_alloc_gb": self.gpu_alloc_gb,
            "gpu_reserved_gb": self.gpu_reserved_gb,
            "gpu_total_gb": self.gpu_total_gb,
            "gpu_alloc_frac": self.gpu_alloc_frac,
        }


class MemoryWatchdogCallback(BaseCallback):
    """
    Callback que monitora e registra uso de memória/VRAM durante o treinamento.

    Parameters
    ----------
    log_freq : int
        Intervalo (em timesteps) para registrar métricas.
    warn_gpu_fraction : float
        Fração do total da GPU a partir da qual um alerta é emitido.
    warn_ram_gb : float
        Uso absoluto de RAM (em GB) a partir do qual um alerta é emitido.
    auto_release_cuda : bool
        Se True, executa torch.cuda.empty_cache() quando o limite é ultrapassado.
    auto_gc : bool
        Se True, chama gc.collect() quando o limite de RAM é ultrapassado.
    verbose : int
        Nível de verbosidade (0 silencioso, 1 logs resumidos, 2 logs completos).
    """

    def __init__(
        self,
        log_freq: int = 5000,
        warn_gpu_fraction: float = 0.85,
        warn_ram_gb: float = 28.0,
        auto_release_cuda: bool = True,
        auto_gc: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.log_freq = max(1, int(log_freq))
        self.warn_gpu_fraction = warn_gpu_fraction
        self.warn_ram_gb = warn_ram_gb
        self.auto_release_cuda = auto_release_cuda
        self.auto_gc = auto_gc

        self._process = psutil.Process(os.getpid())
        self._last_log_step = -1
        self._last_snapshot: Optional[MemorySnapshot] = None

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _collect_snapshot(self) -> MemorySnapshot:
        """Coleta métricas de RAM e (se disponível) VRAM."""
        ram_bytes = self._process.memory_info().rss
        ram_gb = ram_bytes / (1024 ** 3)
        ram_percent = psutil.virtual_memory().percent

        gpu_alloc_gb = 0.0
        gpu_reserved_gb = 0.0
        gpu_total_gb = 0.0

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            gpu_reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
            gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

        return MemorySnapshot(
            timestamp=time.time(),
            ram_gb=ram_gb,
            ram_percent=ram_percent,
            gpu_alloc_gb=gpu_alloc_gb,
            gpu_reserved_gb=gpu_reserved_gb,
            gpu_total_gb=gpu_total_gb,
        )

    def _log_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Imprime snapshot em formato legível."""
        if self.verbose == 0:
            return

        msg = (
            f"[MEM] step={self.num_timesteps:,} | "
            f"RAM={snapshot.ram_gb:.2f}GB ({snapshot.ram_percent:.1f}%)"
        )
        if snapshot.gpu_total_gb > 0:
            msg += (
                f" | GPU alloc={snapshot.gpu_alloc_gb:.2f}GB "
                f"(reserved={snapshot.gpu_reserved_gb:.2f}GB, "
                f"total={snapshot.gpu_total_gb:.1f}GB)"
            )
        print(msg)

        if self.verbose > 1 and self._last_snapshot is not None:
            delta_ram = snapshot.ram_gb - self._last_snapshot.ram_gb
            delta_gpu = snapshot.gpu_alloc_gb - self._last_snapshot.gpu_alloc_gb
            print(
                f"[MEM] ΔRAM={delta_ram:+.2f}GB | "
                f"ΔGPU={delta_gpu:+.2f}GB desde última medição"
            )

    def _emit_warnings(self, snapshot: MemorySnapshot) -> None:
        """Emite alertas e aciona limpezas leves se necessário."""
        warnings = []

        if snapshot.ram_gb >= self.warn_ram_gb:
            warnings.append(
                f"RAM alta: {snapshot.ram_gb:.2f}GB (limite {self.warn_ram_gb:.1f}GB)"
            )
            if self.auto_gc:
                gc.collect()
                warnings.append("gc.collect() executado")

        if snapshot.gpu_total_gb > 0 and snapshot.gpu_alloc_frac >= self.warn_gpu_fraction:
            warnings.append(
                f"GPU alloc alta: {snapshot.gpu_alloc_gb:.2f}/{snapshot.gpu_total_gb:.1f}GB "
                f"({snapshot.gpu_alloc_frac*100:.1f}%)"
            )
            if self.auto_release_cuda:
                torch.cuda.empty_cache()
                warnings.append("torch.cuda.empty_cache() executado")

        if warnings and self.verbose > 0:
            print("[MEM][WARN] " + " | ".join(warnings))

        # Usar RealTimeLogger, se disponível
        try:
            from avaliacoes.real_time_logger import get_logger

            logger = get_logger()
            if logger and hasattr(logger, 'log_resource_snapshot'):
                logger.log_resource_snapshot(snapshot.as_dict())
        except Exception:
            # Logger opcional; ignore erros silenciosamente
            pass

    # ------------------------------------------------------------------ #
    # BaseCallback overrides                                             #
    # ------------------------------------------------------------------ #
    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log_step >= self.log_freq:
            snapshot = self._collect_snapshot()
            self._log_snapshot(snapshot)
            self._emit_warnings(snapshot)
            self._last_snapshot = snapshot
            self._last_log_step = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        """Registra snapshot final ao término do treinamento."""
        snapshot = self._collect_snapshot()
        self._log_snapshot(snapshot)
        self._last_snapshot = snapshot

        if self.verbose > 0:
            print("[MEM] Watchdog finalizado.")
