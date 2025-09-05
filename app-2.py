
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="XRD Peak Marker", layout="wide")

# ---------- Utilities ----------
def load_xrd_from_excel_like(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith(".csv"):
        try:
            raw = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            raw = pd.read_csv(io.BytesIO(file_bytes), sep=";")
    else:
        raw = pd.read_excel(io.BytesIO(file_bytes))
    col_theta, col_int, header_row = None, None, None
    for idx, row in raw.iterrows():
        vals = [str(x).strip() for x in row.values]
        if "<2Theta>" in vals and "I" in vals:
            header_row = idx
            s = row.astype(str).str.strip()
            col_theta = s[s == "<2Theta>"].index[0]
            col_int = s[s == "I"].index[0]
            break
    if header_row is not None:
        data = raw.loc[header_row+1:, [col_theta, col_int]].copy()
        data.columns = ["2theta", "intensity"]
        data = data.dropna().astype(float).reset_index(drop=True)
        return data
    numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
    if len(numeric_cols) >= 2:
        df = pd.DataFrame({"2theta": raw[numeric_cols[0]], "intensity": raw[numeric_cols[1]]}).dropna()
        return df.astype(float)
    for i in range(len(raw.columns)-1):
        try:
            x = pd.to_numeric(raw.iloc[:, i], errors="coerce")
            y = pd.to_numeric(raw.iloc[:, i+1], errors="coerce")
            df = pd.DataFrame({"2theta": x, "intensity": y}).dropna()
            if len(df) > 3:
                return df.astype(float)
        except Exception:
            continue
    raise ValueError("Não foi possível localizar as colunas de 2θ e intensidade.")

def smooth_signal(y, window=9):
    if window < 3:
        return y.copy()
    w = window if window % 2 == 1 else window + 1
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kern = np.ones(w) / w
    return np.convolve(ypad, kern, mode="valid")

def detect_peaks(x, y, window_pts=35, prominence_rel=0.015, min_distance_pts=12):
    y = np.asarray(y); x = np.asarray(x)
    n = len(y)
    candidates = []
    for i in range(1, n-1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            i0 = max(0, i-window_pts)
            i1 = min(n-1, i+window_pts)
            left_min = y[i0:i].min() if i > i0 else y[i]
            right_min = y[i+1:i1+1].min() if i1 > i else y[i]
            prominence = y[i] - max(left_min, right_min)
            candidates.append((i, prominence))
    if not candidates:
        return np.array([], dtype=int)
    dynamic_range = y.max() - y.min() + 1e-9
    thr = prominence_rel * dynamic_range
    peaks = [i for i, prom in candidates if prom >= thr]
    peaks = sorted(peaks, key=lambda i: y[i], reverse=True)
    selected = []
    for idx in peaks:
        if all(abs(idx - j) >= min_distance_pts for j in selected):
            selected.append(idx)
    return np.array(sorted(selected), dtype=int)

def nearest_peak_y(xvec, yvec, x_val):
    idx = int(np.clip(np.searchsorted(xvec, x_val), 1, len(xvec)-1))
    i = idx-1 if yvec[idx-1] >= yvec[idx] else idx
    return float(yvec[i])

def default_labels_from_peaks(x, y, peak_idx):
    px = x[peak_idx]; py = y[peak_idx]
    df = pd.DataFrame({
        "use": True,
        "two_theta": np.round(px, 2),
        "y": py,
        "offset_x": 0,     # points
        "offset_y": 12,    # points
        "rotation": 60,
        "text": [f"{v:.2f}" for v in px],
        "color": "#000000",
    })
    return df

def anti_overlap(df, min_dx=0.25, step_y=6, max_levels=4, mode="stagger"):
    """
    Resolve sobreposição simples baseada em proximidade em 2θ.
    - min_dx: separação mínima em 2θ para considerar que há conflito
    - step_y: incremento de offset_y por nível (points)
    - max_levels: número de degraus antes de voltar ao nível 0
    - mode: 'stagger' alterna níveis; 'stair' acumula
    """
    if df is None or len(df)==0:
        return df
    df = df.copy()
    df.sort_values("two_theta", inplace=True, ignore_index=True)
    level = 0
    prev_x = None
    for i in range(len(df)):
        x = float(df.loc[i, "two_theta"])
        if prev_x is not None and (x - prev_x) < min_dx:
            if mode == "stair":
                level = min(level + 1, max_levels-1)
            else:  # stagger: 0,1,2,3,0,1,...
                level = (level + 1) % max_levels
        else:
            level = 0
        base_oy = float(df.loc[i, "offset_y"]) if not pd.isna(df.loc[i, "offset_y"]) else 12.0
        df.loc[i, "offset_y"] = base_oy + level * step_y
        # pequeno deslocamento alternado no X para ajudar
        base_ox = float(df.loc[i, "offset_x"]) if not pd.isna(df.loc[i, "offset_x"]) else 0.0
        df.loc[i, "offset_x"] = base_ox + (level % 2) * 2.0 - 1.0 if mode == "stagger" else base_ox
        prev_x = x
    return df

def draw_xrd(data, settings, labels_df, figure_size, dpi):
    x = data["2theta"].to_numpy()
    y = data["intensity"].to_numpy()
    y_s = smooth_signal(y, window=settings["smooth_window"])
    peaks = detect_peaks(
        x, y_s,
        window_pts=settings["window_pts"],
        prominence_rel=settings["prominence_rel"],
        min_distance_pts=settings["min_distance_pts"],
    )
    if len(peaks) > settings["max_peaks_auto"]:
        order = np.argsort(y_s[peaks])[::-1][:settings["max_peaks_auto"]]
        peaks = np.sort(peaks[order])

    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    ax.plot(x, y, linewidth=settings["line_width"], color=settings["line_color"])

    # tick marks (auto peaks)
    if settings["show_ticks"]:
        px = x[peaks]; py = y[peaks]
        for xi, yi in zip(px, py):
            if settings["tick_mode"] == "percent":
                tick_len = yi * settings["tick_height"]
            else:
                tick_len = settings["tick_abs"]
            y0 = max(0, yi - tick_len)
            ax.plot([xi, xi], [y0, yi],
                    linewidth=settings["tick_width"],
                    color=settings["tick_color"])

    # labels
    if labels_df is not None and len(labels_df) > 0:
        for _, row in labels_df.iterrows():
            if not bool(row.get("use", True)):
                continue
            xi = float(row["two_theta"])
            yi = float(row["y"]) if pd.notnull(row["y"]) else nearest_peak_y(x, y, xi)
            txt = str(row["text"]) if pd.notnull(row["text"]) else f"{xi:.2f}"
            ox = float(row.get("offset_x", 0))
            oy = float(row.get("offset_y", 12))
            rot = float(row.get("rotation", 60))
            color = str(row.get("color", settings["label_color"]))
            ax.annotate(
                txt, xy=(xi, yi), xytext=(ox, oy), textcoords="offset points",
                ha="center", va="bottom", fontsize=settings["font_size"],
                rotation=rot,
                arrowprops=dict(arrowstyle="-", lw=settings["leader_width"],
                                color=color, shrinkA=0, shrinkB=0)
            )

    ax.set_xlabel("2θ (°)")
    ax.set_ylabel("Intensidade (u.a.)")

    if settings["auto_y"]:
        ymin = max(0, float(np.nanmin(y)) - 0.02*(float(np.nanmax(y))-float(np.nanmin(y))))
        ax.set_ylim(ymin, None)
    else:
        ax.set_ylim(settings["ymin"], settings["ymax"])

    if settings["xlim"]:
        ax.set_xlim(*settings["xlim"])

    if settings["title"]:
        ax.set_title(settings["title"], loc=settings["title_loc"])
    ax.margins(x=0.01)
    fig.tight_layout()
    return fig, peaks

# ---------- UI ----------
st.title("DRX — Marcação e Rotulagem de Picos (Web)")

with st.sidebar:
    st.header("Detecção")
    smooth_window = st.slider("Suavização (janela média móvel)", 1, 21, 9, step=2)
    prominence_rel = st.number_input("Proeminência relativa (0–1)", value=0.015, min_value=0.0, step=0.001, format="%.3f")
    window_pts = st.slider("Janela local p/ proeminência (pontos)", 5, 200, 35, 1)
    min_distance_pts = st.slider("Distância mínima entre picos (pontos)", 1, 100, 12, 1)
    max_peaks_auto = st.slider("Máx. picos (auto)", 5, 200, 60, 1)

    st.markdown("---")
    st.header("Curva")
    line_color = st.color_picker("Cor da curva", "#1f77b4")
    line_width = st.slider("Espessura da curva (px)", 0.1, 5.0, 2.0, 0.1)

    st.markdown("---")
    st.header("Ticks e Rótulos")
    show_ticks = st.checkbox("Mostrar ticks", value=True)
    tick_mode = st.radio("Modo do tamanho do tick", ["percent", "absoluto"], index=0,
                         format_func=lambda s: "Percentual do pico" if s=="percent" else "Unidades de intensidade")
    if tick_mode == "percent":
        tick_height = st.slider("Altura do tick (% do pico)", 0.1, 10.0, 2.0, 0.1) / 100.0
        tick_abs = 20.0
    else:
        tick_abs = st.number_input("Altura do tick (unidades de intensidade)", value=20.0, min_value=1.0, step=1.0)
        tick_height = 0.02
    tick_width = st.slider("Espessura do tick (px)", 0.1, 4.0, 1.0, 0.1)
    tick_color = st.color_picker("Cor do tick", "#000000")

    label_color = st.color_picker("Cor padrão do rótulo", "#000000")
    font_size = st.slider("Tamanho da fonte", 4, 16, 6, 1)
    leader_width = st.slider("Espessura da linha-guia", 0.1, 3.0, 1.0, 0.1)

    st.markdown("---")
    st.header("Título e Faixa dos Eixos")
    title = st.text_input("Título (opcional)", "")
    title_loc = st.selectbox("Posição do título", ["left", "center", "right"], index=2)

    st.markdown("---")
    st.header("Eixo Y")
    auto_y = st.checkbox("Auto-escalar Y", value=True)
    ymin = st.number_input("Y mínimo", value=0.0, step=10.0, disabled=auto_y)
    ymax = st.number_input("Y máximo", value=2500.0, step=10.0, disabled=auto_y)

    st.markdown("---")
    st.header("Exportação")
    dpi = st.slider("DPI do PNG/SVG/PDF", 100, 1200, 400, 50)
    width_px = st.number_input("Largura (px)", value=3200, min_value=800, max_value=20000, step=100)
    height_px = st.number_input("Altura (px)", value=800, min_value=300, max_value=3000, step=50)
    fmt = st.selectbox("Formato", ["PNG", "SVG", "PDF"])
    scale_x = st.slider("Escala lateral (X)", 0.5, 4.0, 1.0, 0.1)
    st.caption("Dica: aumente a escala lateral para espaçar os rótulos sem aumentar a altura.")

uploaded = st.file_uploader("Escolha um arquivo (.xlsx ou .csv)", type=["xlsx", "csv"])

if uploaded is not None:
    try:
        raw_bytes = uploaded.read()
        data = load_xrd_from_excel_like(raw_bytes, uploaded.name)

        xmin = float(np.nanmin(data["2theta"]))
        xmax = float(np.nanmax(data["2theta"]))
        x_range = st.slider("Faixa de 2θ (°)", xmin, xmax, (max(xmin, xmin), min(xmax, xmax)))

        settings = {
            "smooth_window": smooth_window,
            "prominence_rel": float(prominence_rel),
            "window_pts": window_pts,
            "min_distance_pts": min_distance_pts,
            "max_peaks_auto": max_peaks_auto,
            "tick_mode": tick_mode,
            "show_ticks": show_ticks,
            "tick_height": float(tick_height),
            "tick_abs": float(tick_abs),
            "tick_width": float(tick_width),
            "tick_color": tick_color,
            "label_color": label_color,
            "font_size": int(font_size),
            "leader_width": float(leader_width),
            "xlim": x_range,
            "title": title,
            "title_loc": title_loc,
            "line_color": line_color,
            "line_width": float(line_width),
            "auto_y": auto_y,
            "ymin": float(ymin),
            "ymax": float(ymax),
        }

        if "labels_df" not in st.session_state or st.button("Resetar rótulos (auto)"):
            st.session_state["labels_df"] = None

        eff_width_px = int(width_px * scale_x)
        st.caption(f"Largura efetiva: **{eff_width_px}px** (escala {scale_x:.1f}×)")
        fig_size = (eff_width_px / dpi, height_px / dpi)

        # Desenho preliminar para obter picos
        fig_tmp, peaks = draw_xrd(
            data, settings,
            labels_df=st.session_state.get("labels_df") if st.session_state.get("labels_df") is not None else pd.DataFrame(),
            figure_size=fig_size, dpi=dpi
        )
        plt.close(fig_tmp)

        x = data["2theta"].to_numpy()
        y = data["intensity"].to_numpy()
        if st.session_state.get("labels_df") is None:
            st.session_state["labels_df"] = default_labels_from_peaks(x, y, peaks)

        with st.expander("Editar rótulos (adicione, remova, ajuste offsets, texto e cor)", expanded=True):
            st.caption("Dica: deixe 'y' em branco para 'snap' ao pico mais próximo. Você pode adicionar linhas.")
            edited = st.data_editor(
                st.session_state["labels_df"],
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "use": st.column_config.CheckboxColumn("usar", default=True),
                    "two_theta": st.column_config.NumberColumn("2θ (°)", step=0.01, format="%.2f"),
                    "y": st.column_config.NumberColumn("Intensidade (opcional)", step=1),
                    "offset_x": st.column_config.NumberColumn("Offset X (pt)", step=1),
                    "offset_y": st.column_config.NumberColumn("Offset Y (pt)", step=1),
                    "rotation": st.column_config.NumberColumn("Rotação (°)", step=1),
                    "text": st.column_config.TextColumn("Texto"),
                    "color": st.column_config.TextColumn("Cor (#hex)", help="Ex.: #ff0000"),
                },
                hide_index=True,
            )

            # Bulk tools
            st.subheader("Ferramentas em lote")
            col1, col2, col3 = st.columns(3)
            with col1:
                rot_all = st.number_input("Rotação para todos (°)", value=60, step=1)
                if st.button("Aplicar rotação a todos"):
                    edited["rotation"] = rot_all
            with col2:
                base = st.number_input("Base (°)", value=60, step=1)
            with col3:
                step = st.number_input("Passo (°)", value=8, step=1)
            if st.button("Auto rotação (padrão leque: base±step)"):
                order_idx = np.argsort(edited["two_theta"].to_numpy())
                rotations = []
                pattern = [-2, -1, 0, 1, 2]
                for i in range(len(edited)):
                    rotations.append(base + pattern[i % len(pattern)]*step)
                new_rot = edited["rotation"].copy()
                for rank, idx in enumerate(order_idx):
                    new_rot.iloc[idx] = rotations[rank]
                edited["rotation"] = new_rot

            st.markdown("—")
            st.subheader("Anti-sobreposição (beta)")
            min_dx = st.number_input("Separação mínima em 2θ (°)", value=0.25, step=0.01, format="%.2f")
            step_y = st.number_input("Incremento de offset Y (pt)", value=6, step=1)
            max_levels = st.number_input("Nº de níveis", value=4, step=1, min_value=2, max_value=10)
            mode = st.selectbox("Modo", ["stagger", "stair"], index=0,
                                help="stagger alterna níveis; stair acumula níveis até max_levels-1")
            if st.button("Aplicar anti-sobreposição"):
                edited = anti_overlap(edited, min_dx=min_dx, step_y=step_y, max_levels=int(max_levels), mode=mode)

            st.session_state["labels_df"] = edited

        fig, _ = draw_xrd(
            data, settings,
            labels_df=st.session_state["labels_df"],
            figure_size=fig_size, dpi=dpi
        )

        # Export buffers
        png_buf = io.BytesIO()
        svg_buf = io.BytesIO()
        pdf_buf = io.BytesIO()
        if fmt == "PNG":
            fig.savefig(png_buf, format="png", bbox_inches="tight", dpi=dpi)
            png_buf.seek(0)
            st.download_button("Baixar gráfico (PNG)", png_buf, file_name="drx_peaks.png", mime="image/png")
            st.image(png_buf)  # Preview from the same buffer
        elif fmt == "SVG":
            fig.savefig(svg_buf, format="svg", bbox_inches="tight", dpi=dpi)
            svg_buf.seek(0)
            st.download_button("Baixar gráfico (SVG vetorial)", svg_buf, file_name="drx_peaks.svg", mime="image/svg+xml")
            st.pyplot(fig, clear_figure=False)
        else:
            fig.savefig(pdf_buf, format="pdf", bbox_inches="tight", dpi=dpi)
            pdf_buf.seek(0)
            st.download_button("Baixar gráfico (PDF vetorial)", pdf_buf, file_name="drx_peaks.pdf", mime="application/pdf")
            st.pyplot(fig, clear_figure=False)

        # CSV de rótulos
        labels_csv = st.session_state["labels_df"].to_csv(index=False).encode("utf-8")
        st.download_button("Baixar rótulos atuais (CSV)", labels_csv, file_name="rotulos_personalizados.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Ocorreu um erro ao ler/plotar o arquivo: {e}")
else:
    st.info("Carregue um arquivo para gerar o gráfico.")
