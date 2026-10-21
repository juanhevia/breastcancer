import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.training.diagnostico import auditar_metadatos, seleccionar_memorizacion, resumir_predicciones
from src.training.entrenar import EXPERIMENTOS, entrenar


class DiagnosticoTests(unittest.TestCase):
    def test_control_lr_solo_cambia_lr_y_medicion_de_train(self):
        original, candidato = EXPERIMENTOS["v2"], EXPERIMENTOS["v2_lr"]
        cambios = {k for k in original if original[k] != candidato[k]}
        self.assertEqual(cambios, {"lr", "evaluar_train"})
        self.assertEqual(candidato["lr"], 3e-4)
        self.assertEqual(candidato["modelo"], "v2")

    def datos(self):
        filas, pacientes = [], []
        for i in range(10):
            split, fold = ("test", -1) if i >= 8 else ("train", 0 if i >= 6 else 1)
            pid = f"p{i}"
            pacientes.append(dict(pid=pid, pCR=i%2, split=split, dataset="a" if i<5 else "b"))
            for j in range(3):
                filas.append(dict(patient_id=pid, sample_id=f"{pid}_{j}", pCR=i%2,
                                  split=split, fold=fold, slice_index=j))
        return pd.DataFrame(filas), pd.DataFrame(pacientes)

    def test_auditoria_y_seleccion_excluyen_validacion_y_test(self):
        s, p = self.datos()
        auditoria = auditar_metadatos(s, p)
        self.assertEqual(auditoria["errores"], [])
        seleccion = seleccionar_memorizacion(s, pacientes=4)
        self.assertEqual(len(seleccion), 4)
        self.assertEqual(seleccion.patient_id.nunique(), 4)
        self.assertTrue((seleccion.split == "train").all())
        self.assertTrue((seleccion.fold == 1).all())
        self.assertTrue((seleccion.slice_index == 1).all())
        self.assertEqual(seleccion.pCR.sum(), 2)
        pd.testing.assert_frame_equal(seleccion, seleccionar_memorizacion(s, pacientes=4))

    def test_auditoria_detecta_etiquetas_y_particiones_contradictorias(self):
        s, p = self.datos()
        s.loc[0, "pCR"] = 1
        self.assertTrue(auditar_metadatos(s, p)["errores"])
        s, p = self.datos()
        p.loc[0, "pCR"] = 1
        self.assertIn("pCR no coincide entre los dos CSV.", auditar_metadatos(s, p)["errores"])
        s.loc[0, "fold"] = 0
        self.assertTrue(auditar_metadatos(s, p)["errores"])

    def test_cohortes_agregan_por_paciente_y_rechazan_predicciones_incompletas(self):
        s, p = self.datos()
        va = s[s.fold == 0]
        pred = va[["sample_id", "patient_id", "pCR"]].assign(probabilidad=va.pCR*.6+.2)
        r = resumir_predicciones(pred, p, va)
        self.assertEqual(r[0]["pacientes"], 2)
        self.assertEqual(r[0]["auc"], 1)
        with self.assertRaisesRegex(ValueError, "todos los cortes"):
            resumir_predicciones(pred.iloc[:-1], p, va)
        with self.assertRaisesRegex(ValueError, "cohorte"):
            resumir_predicciones(pred, p[p.pid != "p6"], va)

    def test_cohorte_con_una_clase_no_inventa_auc(self):
        s, p = self.datos()
        p.loc[p.pid == "p6", "dataset"] = "c"
        va = s[s.fold == 0]
        pred = va[["sample_id", "patient_id", "pCR"]].assign(probabilidad=.3)
        r = resumir_predicciones(pred, p, va)
        self.assertTrue(all(x["auc"] is None for x in r if x["cohorte"] != "todas"))

    def test_interrumpir_despues_de_una_epoca_conserva_predicciones_y_metricas(self):
        torch.set_num_threads(2)
        class Interrumpible(torch.nn.Sequential):
            pasos_train = 0

            def forward(self, x):
                if self.training:
                    self.pasos_train += 1
                    if self.pasos_train == 2:
                        raise KeyboardInterrupt
                return super().forward(x)

        modelo = Interrumpible(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
        x = torch.tensor([[0.], [1.], [2.], [3.]])
        y = torch.tensor([0., 0., 1., 1.])
        ds = TensorDataset(x, y)
        filas = pd.DataFrame({"sample_id": list("abcd"), "patient_id": list("abcd"), "pCR": y.numpy()})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                entrenar(modelo, ds, ds, filas, epocas=3, batch_size=4, dispositivo="cpu", salida=tmp)
            p = Path(tmp)
            met = json.loads((p / "metricas.json").read_text())
            self.assertEqual(met["mejor_epoca"], 1)
            self.assertEqual(len(pd.read_csv(p / "historial.csv")), 1)
            modelo.load_state_dict(torch.load(p / "mejor_modelo.pt", weights_only=True))
            modelo.eval()
            with torch.no_grad(): prob = modelo(x).sigmoid().numpy()
            np.testing.assert_allclose(pd.read_csv(p / "predicciones_validacion.csv").probabilidad,
                                       prob, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
