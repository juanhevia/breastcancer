"""Diagnóstico de desarrollo: promedio fijo 50/50 de v2 y v7 normales.

Usa exclusivamente predicciones internas/externas de la CV ya completada.
No entrena, no cambia checkpoints, no abre test y no busca pesos de mezcla.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/importados_chat_20261027/revision/models/comparacion_v2_v7"
DEST = ROOT / "models/diagnostico_combinacion_v2_v7_20261027"


def combinar(fold, interna):
    name = ("seleccion_interna/predicciones_validacion.csv" if interna
            else "predicciones_externas.csv")
    a, b = [pd.read_csv(SOURCE / f"fold_{fold}" / model / "normal" / name)
            for model in ("v2", "v7")]
    merged = a.merge(b, on="sample_id", suffixes=("_v2", "_v7"),
                     how="outer", validate="one_to_one", indicator=True)
    assert merged._merge.eq("both").all(), "Muestras distintas"
    for col in ("patient_id", "pCR"):
        assert merged[col + "_v2"].eq(merged[col + "_v7"]).all(), col
    merged = merged.rename(columns={"patient_id_v2": "patient_id", "pCR_v2": "pCR"})
    assert merged.groupby("patient_id").pCR.nunique().eq(1).all()
    p = merged.groupby("patient_id").agg(
        pCR=("pCR", "first"), v2=("probabilidad_v2", "mean"),
        v7=("probabilidad_v7", "mean"))
    assert np.isfinite(p[["v2", "v7"]]).all().all()
    assert p[["v2", "v7"]].ge(0).all().all() and p[["v2", "v7"]].le(1).all().all()
    p["promedio"] = (p.v2 + p.v7) / 2
    return p


def umbral(p):
    return float(max(np.unique(np.r_[0., .5, 1., p.promedio.values]), key=lambda t: (
        f1_score(p.pCR, p.promedio >= t, average="macro", zero_division=0),
        -abs(t - .5), t)))


def main():
    records = []
    patients_seen = set()
    for k in range(5):
        internal, external = combinar(k, True), combinar(k, False)
        assert not set(internal.index) & set(external.index)
        assert not patients_seen & set(external.index)
        patients_seen.update(external.index)
        threshold = umbral(internal)
        tn, fp, fn, tp = confusion_matrix(external.pCR, external.promedio >= threshold,
                                          labels=[0, 1]).ravel()
        records.append(dict(
            fold=k, pacientes=len(external),
            auc_v2=roc_auc_score(external.pCR, external.v2),
            auc_v7=roc_auc_score(external.pCR, external.v7),
            auc_promedio=roc_auc_score(external.pCR, external.promedio),
            auc_interno_promedio=roc_auc_score(internal.pCR, internal.promedio),
            umbral_interno=threshold,
            f1_macro=f1_score(external.pCR, external.promedio >= threshold, average="macro"),
            sensibilidad=tp / (tp + fn), especificidad=tn / (tn + fp),
            correlacion_probabilidades=float(external[["v2", "v7"]].corr().iloc[0, 1]),
        ))
    assert len(patients_seen) == 1097
    table = pd.DataFrame(records)
    reference = pd.read_csv(SOURCE / "resultados_por_fold.csv")
    for model in ("v2", "v7"):
        expected = reference[reference.modelo.eq(model) & reference.perdida.eq("normal")].set_index("fold")
        assert np.allclose(table.set_index("fold")["auc_" + model], expected.auc.reindex(range(5)), atol=1e-7)
    DEST.mkdir(exist_ok=True)
    table.to_csv(DEST / "comparacion.csv", index=False)
    summary = {
        "protocolo": "Promedio fijo 0.5*v2 normal + 0.5*v7 normal; umbral F1 macro interno",
        "aviso": "Análisis exploratorio posterior sobre desarrollo; no test independiente ni modelo desplegado",
        "medias": table.drop(columns=["fold", "pacientes"]).mean().to_dict(),
        "desviaciones_auc": table[["auc_v2", "auc_v7", "auc_promedio"]].std(ddof=1).to_dict(),
    }
    (DEST / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(table.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
