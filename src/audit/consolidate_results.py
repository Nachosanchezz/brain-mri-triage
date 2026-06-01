"""
consolidate_results.py
----------------------
Junta en un solo JSON+MD los resultados de:
  - confounded baseline (cnn3d_test_results.json)
  - LODO A y B (outputs/evaluation/lodo_A|B/cnn3d_test_results.json)
  - audit_leakage.json (tiny baseline mezcla aleatoria)
  - audit_lodo.json   (tiny baseline LODO)
  - btc_intradomain_tinybaseline.json
  - btc_intradomain/cnn_kfold_results.json

Salida:
  docs/audit/resumen_consolidado.json
  docs/audit/resumen_consolidado.md
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "audit"


def load_json_safe(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  no se pudo leer {p}: {exc}")
        return None


def fmt(v, fmt_spec=".4f"):
    if v is None:
        return "—"
    try:
        return format(float(v), fmt_spec)
    except Exception:
        return str(v)


def main() -> None:
    paths = {
        "confound_cnn":       REPO_ROOT / "outputs" / "evaluation" / "cnn3d_test_results.json",
        "lodo_a_cnn":         REPO_ROOT / "outputs" / "evaluation" / "lodo_A" / "cnn3d_test_results.json",
        "lodo_b_cnn":         REPO_ROOT / "outputs" / "evaluation" / "lodo_B" / "cnn3d_test_results.json",
        "tiny_mix":           REPO_ROOT / "docs" / "audit" / "audit_leakage.json",
        "tiny_lodo":          REPO_ROOT / "docs" / "audit" / "audit_lodo.json",
        "btc_tiny":           REPO_ROOT / "docs" / "audit" / "btc_intradomain_tinybaseline.json",
        "btc_cnn":            REPO_ROOT / "outputs" / "evaluation" / "btc_intradomain" / "cnn_kfold_results.json",
    }
    data = {k: load_json_safe(p) for k, p in paths.items()}

    # extraer numeros
    summary = {}

    if data["confound_cnn"]:
        summary["confound_cnn_test_auc"] = data["confound_cnn"].get("auc")
    if data["lodo_a_cnn"]:
        summary["lodo_A_cnn_test_auc"] = data["lodo_a_cnn"].get("auc")
        summary["lodo_A_cnn_test_sen"] = data["lodo_a_cnn"].get("metrics_at_threshold", {}).get("sensitivity")
        summary["lodo_A_cnn_test_spe"] = data["lodo_a_cnn"].get("metrics_at_threshold", {}).get("specificity")
    if data["lodo_b_cnn"]:
        summary["lodo_B_cnn_test_auc"] = data["lodo_b_cnn"].get("auc")
        summary["lodo_B_cnn_test_sen"] = data["lodo_b_cnn"].get("metrics_at_threshold", {}).get("sensitivity")
        summary["lodo_B_cnn_test_spe"] = data["lodo_b_cnn"].get("metrics_at_threshold", {}).get("specificity")
    if data["tiny_mix"]:
        summary["tiny_mix_logreg_auc"] = data["tiny_mix"].get("tiny_baseline_label", {}).get("logreg_test_auc")
        summary["tiny_mix_rf_auc"]     = data["tiny_mix"].get("tiny_baseline_label", {}).get("rf_test_auc")
    if data["tiny_lodo"]:
        for k, v in data["tiny_lodo"].items():
            if isinstance(v, dict) and "logreg_test_auc" in v:
                summary[f"tiny_lodo:{k}:logreg"] = v.get("logreg_test_auc")
                summary[f"tiny_lodo:{k}:rf"]     = v.get("rf_test_auc")
    if data["btc_tiny"]:
        summary["btc_tiny_logreg_auc"] = data["btc_tiny"].get("logreg_auc_overall")
        summary["btc_tiny_logreg_ci95"] = data["btc_tiny"].get("logreg_auc_ci95")
        summary["btc_tiny_rf_auc"]     = data["btc_tiny"].get("rf_auc_overall")
        summary["btc_tiny_rf_ci95"]    = data["btc_tiny"].get("rf_auc_ci95")
    if data["btc_cnn"]:
        summary["btc_cnn_auc"] = data["btc_cnn"].get("overall_auc")
        summary["btc_cnn_ci95"] = data["btc_cnn"].get("overall_auc_ci95")
        summary["btc_cnn_sen"] = data["btc_cnn"].get("metrics_at_thr_0.5", {}).get("sensitivity")
        summary["btc_cnn_spe"] = data["btc_cnn"].get("metrics_at_thr_0.5", {}).get("specificity")

    # JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "resumen_consolidado.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # MD
    md = []
    md.append("# Resumen consolidado de la auditoria\n")
    md.append("Tabla unica con todos los resultados ordenados de mas confundido a mas honesto.\n")
    md.append("## Tabla maestra\n")
    md.append("| Experimento | Modelo | AUC | IC95% | Notas |")
    md.append("|---|---|---|---|---|")
    md.append(f"| Mixed (4 datasets, split aleatorio) | Tiny baseline LogReg | **{fmt(summary.get('tiny_mix_logreg_auc'))}** | — | atajo trivial de intensidad |")
    md.append(f"| Mixed (4 datasets, split aleatorio) | Tiny baseline RF     | **{fmt(summary.get('tiny_mix_rf_auc'))}** | — | idem |")
    md.append(f"| Mixed (4 datasets, split aleatorio) | CNN 3D 2-canales     | **{fmt(summary.get('confound_cnn_test_auc'))}** | — | confounded baseline |")

    md.append(f"| LODO A: BraTS+IXI -> UPENN+NKI | Tiny baseline LogReg | {fmt(summary.get('tiny_lodo:train BraTS+IXI -> test UPENN+NKI:logreg'))} | — | |")
    md.append(f"| LODO A: BraTS+IXI -> UPENN+NKI | **CNN 3D 2-canales** | **{fmt(summary.get('lodo_A_cnn_test_auc'))}** | — | sen={fmt(summary.get('lodo_A_cnn_test_sen'))}  spe={fmt(summary.get('lodo_A_cnn_test_spe'))} |")
    md.append(f"| LODO B: UPENN+NKI -> BraTS+IXI | Tiny baseline LogReg | {fmt(summary.get('tiny_lodo:train UPENN+NKI -> test BraTS+IXI:logreg'))} | — | |")
    md.append(f"| LODO B: UPENN+NKI -> BraTS+IXI | **CNN 3D 2-canales** | **{fmt(summary.get('lodo_B_cnn_test_auc'))}** | — | sen={fmt(summary.get('lodo_B_cnn_test_sen'))}  spe={fmt(summary.get('lodo_B_cnn_test_spe'))} |")

    btc_lr_ci = summary.get("btc_tiny_logreg_ci95") or [None, None]
    btc_rf_ci = summary.get("btc_tiny_rf_ci95") or [None, None]
    btc_cnn_ci = summary.get("btc_cnn_ci95") or [None, None]
    md.append(f"| **Ghent intra-dominio (n=36, k-fold)** | Tiny baseline LogReg | **{fmt(summary.get('btc_tiny_logreg_auc'))}** | [{fmt(btc_lr_ci[0])}, {fmt(btc_lr_ci[1])}] | T1-only |")
    md.append(f"| **Ghent intra-dominio (n=36, k-fold)** | Tiny baseline RF     | **{fmt(summary.get('btc_tiny_rf_auc'))}** | [{fmt(btc_rf_ci[0])}, {fmt(btc_rf_ci[1])}] | T1-only |")
    md.append(f"| **Ghent intra-dominio (n=36, k-fold)** | **CNN 3D 1-canal**   | **{fmt(summary.get('btc_cnn_auc'))}** | [{fmt(btc_cnn_ci[0])}, {fmt(btc_cnn_ci[1])}] | sen={fmt(summary.get('btc_cnn_sen'))}  spe={fmt(summary.get('btc_cnn_spe'))} |")

    # Embeddings (E1)
    sil = load_json_safe(REPO_ROOT / "docs" / "audit" / "embeddings_silhouette.json")
    if sil:
        md.append("\n## Embeddings (E1) — espacio latente de la CNN\n")
        md.append(f"- silhouette global: por etiqueta = {fmt(sil.get('silhouette_by_label'), '.3f')}  |  "
                  f"por dataset = {fmt(sil.get('silhouette_by_dataset'), '.3f')} (diluido)")
    intra = load_json_safe(REPO_ROOT / "docs" / "audit" / "embeddings_intraclass.json")
    if intra:
        s = intra.get("intraclass_sanos_IXI_vs_NKI") or {}
        t = intra.get("intraclass_tumor_BraTS_vs_UPENN") or {}
        dc = intra.get("dataset_classifier_from_embeddings") or {}
        md.append(f"- **intra-clase (huella de procedencia aislada):** IXI vs NKI (sanos) "
                  f"LogReg CV-AUC = {fmt(s.get('logreg_cv_auc'), '.3f')}; "
                  f"BraTS vs UPENN (tumor) = {fmt(t.get('logreg_cv_auc'), '.3f')}")
        md.append(f"- clasificador de dataset (4 clases) desde embeddings: "
                  f"acc = {fmt(dc.get('cv_accuracy'), '.3f')} (azar {dc.get('chance_level')})")
        md.append("- Conclusion: el latente identifica el centro de origen incluso entre "
                  "sujetos de la MISMA clase -> codifica procedencia, no solo tumor. "
                  "Ver `figures/embeddings_tsne.png`.")

    md.append("\n## Lectura sugerida\n")
    md.append("- **Mixed AUC ~ 1.0** (CNN y baseline lineal): confound trivial.")
    md.append("- **LODO**: caos asimetrico, AUC < 0.5 en algunos casos = predicciones invertidas. Firma de dataset, no tumor.")
    md.append("- **Ghent intra-dominio**: con dominio controlado, AUC honesto con IC95%. ESE es el numero entregable.")
    md.append("- **Embeddings**: el espacio latente codifica origen del dato mas alla de la clase.")
    md.append("\n## Figuras (docs/audit/figures/)\n")
    md.append("auc_summary, roc_curves, score_hist_confound, score_hist_lodo, "
              "confusion_matrices, btc_kfold_bars, embeddings_tsne, embeddings_pca, "
              "intensity_by_dataset, gradcam/{confound,lodo_A,lodo_B}/*")
    md.append("\n_Generado por `src/audit/consolidate_results.py`_\n")

    (OUT_DIR / "resumen_consolidado.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Resumen guardado: {OUT_DIR/'resumen_consolidado.md'}")


if __name__ == "__main__":
    main()
