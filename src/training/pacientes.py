"""Aprendizaje con todos los cortes de cada paciente y media de probabilidades."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def grupos_pacientes(filas):
    """Índices posicionales y etiqueta única; admite filas intercaladas."""
    requeridas = ["sample_id", "patient_id", "pCR"]
    if not set(requeridas).issubset(filas.columns) or filas.empty:
        raise ValueError("Se necesitan cortes con sample_id, patient_id y pCR.")
    if filas[requeridas].isna().any().any() or filas.sample_id.duplicated().any():
        raise ValueError("Hay metadatos ausentes o cortes duplicados.")
    if not filas.pCR.isin([0, 1]).all():
        raise ValueError("pCR debe ser binaria (0/1).")
    grupos = filas.reset_index(drop=True).groupby("patient_id", sort=False)
    if (grupos.pCR.nunique() != 1).any():
        raise ValueError("Hay etiquetas contradictorias dentro de una paciente.")
    indices = list(grupos.indices.values())
    etiquetas = [float(filas.iloc[idx[0]].pCR) for idx in indices]
    return indices, etiquetas


class CortesPorPaciente(Dataset):
    """Una muestra contiene TODOS los cortes de una paciente, sin rellenar."""

    def __init__(self, cortes, filas):
        if len(cortes) != len(filas):
            raise ValueError("Los cortes y sus metadatos no tienen la misma longitud.")
        self.cortes = cortes
        self.indices, self.etiquetas = grupos_pacientes(filas)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        muestras = [self.cortes[int(j)] for j in self.indices[i]]
        etiqueta = self.etiquetas[i]
        if any(float(y) != etiqueta for _, y in muestras):
            raise ValueError("Las etiquetas del dataset no coinciden con los metadatos.")
        return torch.stack([x for x, _ in muestras]), etiqueta


def juntar_pacientes(lote):
    """Concatena cortes, conservando cuántos pertenecen a cada paciente."""
    imagenes, etiquetas = zip(*lote)
    return (torch.cat(imagenes), torch.tensor(etiquetas, dtype=torch.float32),
            [len(x) for x in imagenes])


def logits_media_paciente(logits, tamanos):
    """Logit de la MEDIA de sigmoides, no media de logits.

    log(p/(1-p)) = log(sum(sigmoid(z))) - log(sum(sigmoid(-z))).
    Los divisores n se cancelan. log-sigmoid/logsumexp evita saturación y
    mantiene gradientes incluso cuando todos los logits son muy extremos.
    """
    if logits.ndim != 1 or not tamanos or min(tamanos) < 1 or sum(tamanos) != len(logits):
        raise ValueError("Los tamaños deben repartir todos los logits en pacientes no vacías.")
    return torch.stack([
        torch.logsumexp(F.logsigmoid(z), dim=0)
        - torch.logsumexp(F.logsigmoid(-z), dim=0)
        for z in torch.split(logits, tamanos)
    ])


def perdida_pacientes(logits, filas, pos_weight=None):
    """BCE media por paciente sobre logits por corte en el orden de filas."""
    if len(logits) != len(filas):
        raise ValueError("Los logits y los metadatos no tienen la misma longitud.")
    indices, etiquetas = grupos_pacientes(filas)
    ordenados = torch.cat([logits[torch.as_tensor(idx, device=logits.device)] for idx in indices])
    agregados = logits_media_paciente(ordenados, [len(idx) for idx in indices])
    peso = logits.new_tensor(pos_weight) if pos_weight is not None else None
    return F.binary_cross_entropy_with_logits(agregados, logits.new_tensor(etiquetas), pos_weight=peso)
