"""Entrenamiento 4+1 por paciente, sin evaluación externa ni test final."""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import utils_caso as uc
from src.data.aumentos import AumentosDCE
from src.training.comparar_cv import MODELOS, elegir_umbral, metricas, validar_samples
from src.training.entrenar import entrenar, guardar_json


def particiones(samples, fold_val=0):
    desarrollo = validar_samples(samples)
    train, interna = uc.particion(desarrollo, fold_val=fold_val)
    if set(train.patient_id) & set(interna.patient_id):
        raise ValueError('Solapamiento de pacientes.')
    return train.reset_index(drop=True), interna.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datos', type=Path, default=uc.RAIZ)
    p.add_argument('--salida', type=Path, required=True)
    p.add_argument('--modelo', choices=list(MODELOS), required=True)
    p.add_argument('--fold-val', type=int, choices=range(5), default=0)
    p.add_argument('--perdida', choices=['normal', 'ponderada', 'ambas'], default='normal')
    p.add_argument('--dispositivo', choices=['cpu', 'cuda', 'mps'], default='cuda')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epocas', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--hilos', type=int, default=4)
    p.add_argument('--semilla', type=int, default=42)
    p.add_argument('--solo-plan', action='store_true')
    args = p.parse_args()
    if min(args.epocas, args.batch_size, args.hilos) < 1 or args.num_workers < 0:
        p.error('Épocas, batch e hilos deben ser positivos; workers >= 0.')
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error('lr debe ser finito y positivo.')
    train, interna = particiones(uc.cargar_samples(args.datos), args.fold_val)
    print(f'final_train: {train.patient_id.nunique()} pacientes de entrenamiento / '
          f'{interna.patient_id.nunique()} de validación interna. Externo: 0. Test excluido.', flush=True)
    if args.solo_plan:
        return
    if args.salida.exists():
        p.error('La salida ya existe; usa una carpeta nueva para conservar los resultados anteriores.')
    if args.dispositivo == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA/ROCm no disponible; no se cambia automáticamente a CPU.')
    args.salida.mkdir(parents=True)
    particion = pd.concat([train.assign(uso='train'), interna.assign(uso='seleccion_interna')])
    particion[['patient_id', 'pCR', 'fold', 'uso']].drop_duplicates().to_csv(
        args.salida / 'particion_pacientes.csv', index=False)
    particion[['sample_id', 'patient_id', 'pCR', 'fold', 'uso']].to_csv(
        args.salida / 'particion.csv', index=False)
    torch.set_num_threads(args.hilos)
    perdidas = ['normal', 'ponderada'] if args.perdida == 'ambas' else [args.perdida]
    for perdida in perdidas:
        random.seed(args.semilla)
        np.random.seed(args.semilla)
        torch.manual_seed(args.semilla)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.semilla)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        modelo = MODELOS[args.modelo]()
        destino = args.salida / perdida
        destino.mkdir()
        config = dict(vars(args), modo='final_train', fold_interno=args.fold_val,
                      perdida=perdida, paciencia=9, min_delta=0., weight_decay=.001,
                      optimizador='adamw', seleccion='auc', agregacion='mean', umbral=.5,
                      aumentos=True, reducir_lr=True, evaluar_train=True, prueba=False,
                      normalizacion='PRE, EARLY, LATE / 255', python_torch=str(torch.__version__))
        config.update(datos=str(args.datos.resolve()), salida=str(destino.resolve()))
        guardar_json(destino / 'configuracion.json', config)
        hist, _, mejor_epoca, _ = entrenar(
            modelo, uc.BreastDCEDataset(train, args.datos, transform=AumentosDCE()),
            uc.BreastDCEDataset(interna, args.datos), interna,
            epocas=args.epocas, lr=args.lr, batch_size=args.batch_size,
            pos_weight=uc.pos_weight(train) if perdida == 'ponderada' else None,
            paciencia=9, dispositivo=args.dispositivo, salida=destino, semilla=args.semilla,
            num_workers=args.num_workers, optimizador='adamw', weight_decay=.001,
            reducir_lr=True, seleccion='auc', min_delta=0.,
            train_eval_ds=uc.BreastDCEDataset(train, args.datos), train_samples=train,
            nombre=f'final_train / {args.modelo} / {perdida}',
        )
        predicciones = pd.read_csv(destino / 'predicciones_validacion.csv')
        umbral = elegir_umbral(predicciones)
        guardar_json(destino / 'seleccion_congelada.json', dict(
            mejor_epoca=mejor_epoca, epocas_completadas=len(hist['epoca']), umbral=umbral,
            fold_interno=args.fold_val, criterio='Máximo F1 macro en validación interna; no criterio clínico'))
        guardar_json(destino / 'metricas_internas_umbral.json', metricas(predicciones, umbral))
    print(f'Entrenamiento terminado: {args.salida}. Resultados internos; no son evaluación externa.')


if __name__ == '__main__':
    main()
