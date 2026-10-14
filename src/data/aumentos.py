"""Aumentos moderados, aplicados SOLO a entrenamiento y juntos a las tres fases."""

import math

import torch
import torch.nn.functional as F


class AumentosDCE:
    """Reflejo horizontal (50 %) y giro uniforme entre -10 y +10 grados.

    Una sola rejilla por corte mantiene la alineación PRE/EARLY/LATE. No
    modifica el contraste relativo de las fases ni aprende de validación.
    Usa el generador de PyTorch, que DataLoader inicializa también en workers.
    """

    def __call__(self, imagen):
        if imagen.ndim != 3 or imagen.shape[0] != 3:
            raise ValueError("Se espera un tensor (3, alto, ancho) PRE/EARLY/LATE.")
        if torch.rand(()).item() < 0.5:
            imagen = torch.flip(imagen, dims=(-1,))
        angulo = (torch.rand(()).item() * 20 - 10) * math.pi / 180
        c, s = math.cos(angulo), math.sin(angulo)
        matriz = imagen.new_tensor([[c, -s, 0], [s, c, 0]]).unsqueeze(0)
        lote = imagen.unsqueeze(0)
        rejilla = F.affine_grid(matriz, lote.shape, align_corners=False)
        return F.grid_sample(
            lote, rejilla, mode="bilinear", padding_mode="zeros", align_corners=False,
        ).squeeze(0)
