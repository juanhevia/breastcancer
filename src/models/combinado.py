"""Complemento tabular sobre la puntuación de una CNN congelada. Sin pickle."""
import json
from pathlib import Path

import numpy as np

SUBTIPOS = ('HRposHER2neg', 'TripleNeg', 'HER2pos', 'No disponible')


def validar_variables(variables):
    if not isinstance(variables, dict) or set(variables) - {'age', 'HR_HER2_STATUS'}:
        raise ValueError('Solo se admiten edad y subtipo; el volumen aún no está habilitado.')
    edad = variables.get('age')
    if edad is not None and (isinstance(edad, bool) or not isinstance(edad, (int, float))
                             or not np.isfinite(edad) or not 0 < edad <= 120):
        raise ValueError('La edad debe estar entre 0 y 120 años, o quedar vacía.')
    subtipo = variables.get('HR_HER2_STATUS') or 'No disponible'
    if subtipo not in SUBTIPOS:
        raise ValueError('Subtipo no reconocido.')
    if edad is None and subtipo == 'No disponible':
        raise ValueError('Aporta al menos edad o subtipo para el análisis complementario.')
    return edad, subtipo


def matriz(edades, subtipos, probabilidades, mediana, con_imagen):
    edad = np.asarray(edades, dtype=float)
    ausente = np.isnan(edad)
    columnas = [np.where(ausente, mediana, edad), ausente.astype(float)]
    columnas.extend((np.asarray(subtipos) == s).astype(float) for s in SUBTIPOS)
    if con_imagen:
        p = np.clip(np.asarray(probabilidades, dtype=float), 1e-6, 1 - 1e-6)
        columnas.append(np.log(p / (1 - p)))
    return np.column_stack(columnas)


def probabilidad_json(modelo, x):
    media, escala = np.asarray(modelo['media']), np.asarray(modelo['escala'])
    coef = np.asarray(modelo['coeficientes'])
    if media.shape != (x.shape[1],) or escala.shape != media.shape or coef.shape != media.shape:
        raise ValueError('Dimensiones incompatibles en el complemento.')
    if not np.isfinite(np.r_[media, escala, coef, modelo['intercepto']]).all() or np.any(escala <= 0):
        raise ValueError('Parámetros tabulares inválidos.')
    z = np.clip(((x - media) / escala) @ coef + modelo['intercepto'], -700, 700)
    return 1 / (1 + np.exp(-z))


class Complemento:
    def __init__(self, ruta, sha_cnn):
        self.config = json.loads(Path(ruta).read_text())
        if self.config.get('version') != 1 or self.config.get('sha256_cnn') != sha_cnn:
            raise ValueError('El complemento no corresponde a la CNN conectada.')
        if self.config.get('variables') != ['age', 'HR_HER2_STATUS']:
            raise ValueError('Variables del complemento incompatibles.')
        for nombre, con_imagen in [('solo_variables', False), ('combinado', True)]:
            m = self.config['modelos'][nombre]
            if not np.isfinite(m['umbral']) or not 0 <= m['umbral'] <= 1:
                raise ValueError('Umbral del complemento inválido.')
            probabilidad_json(m, matriz([50.], [SUBTIPOS[0]], [.5], 50., con_imagen))
        if not np.isfinite(self.config['mediana_edad']):
            raise ValueError('Imputación inválida.')

    def predecir(self, prob_cnn, variables):
        if not np.isfinite(prob_cnn) or not 0 <= prob_cnn <= 1:
            raise ValueError('Puntuación CNN inválida.')
        edad, subtipo = validar_variables(variables)
        salidas = {}
        for nombre, con_imagen in [('solo_variables', False), ('combinado', True)]:
            x = matriz([np.nan if edad is None else edad], [subtipo], [prob_cnn], self.config['mediana_edad'], con_imagen)
            m = self.config['modelos'][nombre]
            p = float(probabilidad_json(m, x)[0])
            salidas[nombre] = dict(probabilidad=p, umbral=m['umbral'], clase=int(p >= m['umbral']))
        return dict(**salidas, variables=dict(age=edad, HR_HER2_STATUS=subtipo),
                    edad_imputada=edad is None,
                    alcance='Complemento experimental posterior al test de v8. Validación interna de desarrollo; sin evaluación independiente.')
