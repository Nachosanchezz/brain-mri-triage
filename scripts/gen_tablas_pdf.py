#!/usr/bin/env python
"""Genera un PDF de 2 paginas con (1) matriz dataset/clase y (2) tabla resumen
de resultados, para enviar al tutor. Cifras verificadas contra los JSON del repo."""
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["font.family"] = "DejaVu Sans"

HEADER_BG = "#2c3e50"
HEADER_FG = "white"
ROW_ALT = "#f4f6f8"
RED = "#f8d7da"
GREEN = "#d4edda"
YELLOW = "#fff3cd"


def wrap(s, w):
    return "\n".join(textwrap.wrap(str(s), w)) if s else ""


def draw_table(ax, col_labels, rows, col_widths, wrap_widths,
               cell_colors=None, fontsize=7.4, header_fs=7.8, title=None,
               line_h=0.022):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=14, loc="left")

    # Ajusta el texto de cada celda
    wrapped = []
    for r in rows:
        wrapped.append([wrap(c, wrap_widths[j]) for j, c in enumerate(r)])

    tbl = ax.table(cellText=wrapped, colLabels=col_labels,
                   colWidths=col_widths, loc="upper left", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    n_cols = len(col_labels)
    # altura por linea de texto (fraccion de la altura del axes)

    # Cabecera
    for j in range(n_cols):
        c = tbl[0, j]
        c.set_facecolor(HEADER_BG)
        c.set_text_props(color=HEADER_FG, fontweight="bold", fontsize=header_fs)
        c.set_height(line_h * 2.2)
        c.set_edgecolor("white")

    # Filas
    for i, r in enumerate(wrapped, start=1):
        n_lines = max(s.count("\n") + 1 for s in r)
        h = line_h * (n_lines + 0.8)
        for j in range(n_cols):
            c = tbl[i, j]
            c.set_height(h)
            c.set_edgecolor("#cfd6dc")
            c.PAD = 0.02
            # color de fondo
            bg = ROW_ALT if i % 2 == 0 else "white"
            if cell_colors and (i - 1, j) in cell_colors:
                bg = cell_colors[(i - 1, j)]
            c.set_facecolor(bg)
            c.set_text_props(va="center")
    return tbl


# --------------------------------------------------------------------------
# PAGINA 1 — Matriz dataset / clase
# --------------------------------------------------------------------------
ds_cols = ["Dataset", "Procedencia / dominio", "Clase", "n proc.",
           "Modalidades", "Uso en el proyecto", "Riesgo metodologico"]
ds_rows = [
    ["BraTS 2021", "Multicentrico (challenge BraTS)", "Tumor (1)", "580",
     "T1 + T2", "Entrenamiento (pool)",
     "Aporta SOLO positivos -> clase ligada al dominio"],
    ["UPENN-GBM", "Hosp. Univ. Pensilvania (TCIA)", "Tumor (1)", "587",
     "T1 + T2", "Entrenamiento (pool); fue test externo",
     "Aporta SOLO positivos; firma de intensidad muy marcada"],
    ["IXI", "3 hospitales de Londres (sanos)", "Sano (0)", "577",
     "T1 + T2", "Entrenamiento (pool)", "Aporta SOLO negativos"],
    ["NKI Rockland", "Nathan Kline Institute (sanos)", "Sano (0)", "523",
     "T1 + T2", "Entrenamiento (pool); reequilibrio neg.",
     "Aporta SOLO negativos"],
    ["BTC_preop\n(OpenNeuro ds001226)", "Ghent University Hospital (1 centro)",
     "Tumor (1) y Sano (0)", "36\n(25/11)", "T1-only",
     "Validacion intra-dominio (unica medicion honesta)",
     "n pequeno; T1-only"],
]
ds_widths = [0.11, 0.17, 0.10, 0.06, 0.09, 0.20, 0.27]
ds_wrap = [16, 24, 14, 8, 12, 28, 38]
# colorea la fila de BTC (la validacion honesta) en verde claro y resalta clase
ds_colors = {}
for j in range(len(ds_cols)):
    ds_colors[(4, j)] = GREEN
# resalta columna "Clase" del pool en rojo claro (confound)
for i in range(4):
    ds_colors[(i, 2)] = RED

# matriz 2x2 simple
mini_cols = ["", "Tumor (1)", "Sano (0)"]
mini_rows = [
    ["Pool principal", "BraTS, UPENN", "IXI, NKI"],
    ["Validacion honesta", "BTC_preop (Ghent)", "BTC_preop (Ghent)"],
]

fig1 = plt.figure(figsize=(11.69, 8.27))  # A4 horizontal
fig1.suptitle("Tabla 1 — Composicion del problema: matriz dataset / clase",
              fontsize=13, fontweight="bold", x=0.06, ha="left", y=0.97)

ax_main = fig1.add_axes([0.04, 0.40, 0.92, 0.48])
draw_table(ax_main, ds_cols, ds_rows, ds_widths, ds_wrap, cell_colors=ds_colors)

ax_mini = fig1.add_axes([0.04, 0.13, 0.54, 0.22])
ax_mini.set_title("Resumen: ningun dataset del pool aporta las dos clases",
                  fontsize=10, fontweight="bold", loc="left", pad=10)
mini_colors = {(0, 1): RED, (0, 2): RED, (1, 1): GREEN, (1, 2): GREEN}
draw_table(ax_mini, mini_cols, mini_rows, [0.30, 0.35, 0.35], [22, 22, 22],
           cell_colors=mini_colors, fontsize=8.5, header_fs=8.5, line_h=0.16)

fig1.text(0.04, 0.07,
          "Confusion clase<->dominio del 100%: P(tumor|BraTS)=P(tumor|UPENN)=1 y "
          "P(tumor|IXI)=P(tumor|NKI)=0. 'Detectar tumor' y 'reconocer el centro de "
          "origen' son indistinguibles en el pool.\n"
          "Fuentes: data/processed/preprocessing_summary.json ; "
          "docs/audit/audit_leakage.json (label_equals_dataset).",
          fontsize=7.5, color="#444", va="top")

# --------------------------------------------------------------------------
# PAGINA 2 — Tabla resumen de resultados (cascada)
# --------------------------------------------------------------------------
rs_cols = ["#", "Regimen", "Modelo / prueba", "Particion",
           "AUC", "Sen", "Spe", "IC95%", "Interpretacion", "Validez"]
rs_rows = [
    ["1", "Pool multi-fuente (split aleatorio)", "CNN 3D (2 canales)",
     "test 340 (175+/165-)", "0.99997", "0.994", "1.000", "—",
     "Rendimiento aparente casi perfecto", "NO VALIDO (confound)"],
    ["2", "Pool multi-fuente", "Tiny baseline LogReg (16 stats)",
     "test 340 (train n=600)", "1.000", "—", "—", "—",
     "Etiqueta decodificable sin red ni anatomia", "NO VALIDO (atajo)"],
    ["3", "Pool multi-fuente", "Tiny baseline RF", "test 340",
     "0.9989", "—", "—", "—", "Idem", "NO VALIDO (atajo)"],
    ["4", "Identificabilidad de dominio", "Clf dataset (intensidad, 4 clases)",
     "test 340", "acc 0.985", "—", "—", "—",
     "Centro de origen trivialmente separable (azar 0.25)", "Evidencia confound"],
    ["5", "Identificabilidad de dominio", "Clf dataset desde embeddings (4 cl.)",
     "latente 96-d", "acc 0.982", "—", "—", "—",
     "El latente de la CNN codifica procedencia", "Evidencia confound"],
    ["6", "LODO A: BraTS+IXI -> UPENN+NKI", "CNN 3D (2 canales)", "test n=1110",
     "0.6236", "0.676", "0.463", "—", "Degrada hacia el azar fuera de dominio",
     "NO VALIDO (no generaliza)"],
    ["7", "LODO B: UPENN+NKI -> BraTS+IXI", "CNN 3D (2 canales)", "test n=1157",
     "0.2012", "0.010", "0.965", "—",
     "Regla invertida / colapso de sensibilidad", "NO VALIDO (no generaliza)"],
    ["8", "LODO A", "Tiny baseline LogReg", "held-out", "0.9952", "—", "—", "—",
     "El atajo transfiere en una direccion", "Evidencia confound"],
    ["9", "LODO B", "Tiny baseline LogReg", "held-out", "0.3184", "—", "—", "—",
     "El atajo no transfiere en la otra", "Evidencia confound"],
    ["10", "BTC_preop intra-dominio", "CNN 3D (1 canal)", "k-fold, n=36",
     "0.4036", "0.60", "0.364", "[0.213, 0.623]",
     "Compatible con azar (IC cruza 0.5)", "VALIDA (medicion honesta)"],
    ["11", "BTC_preop intra-dominio", "Tiny baseline LogReg", "k-fold, n=36",
     "0.5491", "—", "—", "[0.319, 0.788]", "Compatible con azar",
     "VALIDA (medicion honesta)"],
    ["12", "BTC_preop intra-dominio", "Tiny baseline RF", "k-fold, n=36",
     "0.4055", "—", "—", "[0.215, 0.616]", "Compatible con azar",
     "VALIDA (medicion honesta)"],
    ["13", "Embeddings intra-clase", "LogReg: IXI vs NKI (ambos sanos)", "n=1100",
     "0.998", "—", "—", "—",
     "Separa cohortes dentro de una clase -> codifica origen",
     "Evidencia confound"],
]
rs_widths = [0.025, 0.135, 0.135, 0.085, 0.05, 0.04, 0.04, 0.075, 0.205, 0.135]
rs_wrap = [3, 22, 22, 14, 8, 6, 6, 12, 33, 18]

rs_colors = {}
for i, r in enumerate(rs_rows):
    v = r[-1]
    col = (len(rs_cols) - 1)
    if v.startswith("NO VALIDO"):
        rs_colors[(i, col)] = RED
    elif v.startswith("VALIDA"):
        rs_colors[(i, col)] = GREEN
    else:
        rs_colors[(i, col)] = YELLOW

fig2 = plt.figure(figsize=(11.69, 8.27))  # A4 horizontal
fig2.suptitle("Tabla 2 — Analisis progresivo de validez del rendimiento aparente "
              "(narrativa en cascada)",
              fontsize=12.5, fontweight="bold", x=0.04, ha="left", y=0.975)
ax2 = fig2.add_axes([0.02, 0.10, 0.96, 0.82])
draw_table(ax2, rs_cols, rs_rows, rs_widths, rs_wrap,
           cell_colors=rs_colors, fontsize=6.9, header_fs=7.2)

fig2.text(0.02, 0.055,
          "Rojo = aparentemente bueno pero NO valido (inflado por confound).  "
          "Verde = medicion metodologicamente honesta (clase y dominio desacoplados).  "
          "Amarillo = evidencia del confound.\n"
          "Fuentes: outputs/evaluation/cnn3d_test_results.json ; lodo_{A,B}/cnn3d_test_results.json ; "
          "btc_intradomain/cnn_kfold_results.json ; docs/audit/{audit_leakage,audit_lodo,"
          "btc_intradomain_tinybaseline,embeddings_intraclass}.json",
          fontsize=7, color="#444", va="top")

# --------------------------------------------------------------------------
out = "docs/01_memoria/tablas_resultados_tutor.pdf"
with PdfPages(out) as pdf:
    pdf.savefig(fig1)
    pdf.savefig(fig2)
print("PDF generado:", out)
