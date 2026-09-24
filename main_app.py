"""
Laboratorio de LLMs con Groq + Streamlit
----------------------------------------
- API key de Groq ingresada por el usuario
- Lista de LLMs disponibles (prioriza los modelos GPT-OSS de OpenAI)
- Generación de texto con control de modelo, temperatura y demás parámetros
- Comparación de temperaturas / modelos
- Tokens y Token IDs (tiktoken)
- Bag of Words / TF-IDF
- Métricas de similitud (coseno, euclidiana, Manhattan, producto punto, Jaccard, Levenshtein)
- Embeddings locales (sentence-transformers o LSA) con PCA y búsqueda semántica

Ejecutar:  streamlit run app.py
"""

import html
import re
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)

# ----------------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Laboratorio LLM · Groq", page_icon="🧪", layout="wide")

MODELOS_FALLBACK = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
EXCLUIDOS = ("whisper", "tts", "guard", "orpheus", "playai", "embed")
EMB_OPCIONES = [
    "Sentence-Transformers: paraphrase-multilingual-MiniLM-L12-v2",
    "Sentence-Transformers: all-MiniLM-L6-v2",
    "LSA (TF-IDF + SVD, sin descargas)",
]
CORPUS_DEFECTO = "\n".join(
    [
        "El gato duerme sobre el sofá de la sala.",
        "Un felino descansa en el sillón de la casa.",
        "La inteligencia artificial está transformando la educación.",
        "Los modelos de lenguaje generan texto a partir de probabilidades.",
        "El perro corre por el parque en la mañana.",
        "Los LLM predicen el siguiente token dado un contexto.",
    ]
)

st.session_state.setdefault("corpus", CORPUS_DEFECTO)
st.session_state.setdefault("historial", [])
st.session_state.setdefault("conectado", False)


def es_llm(model_id: str) -> bool:
    return not any(x in model_id.lower() for x in EXCLUIDOS)


def es_gpt(model_id: str) -> bool:
    return "gpt-oss" in model_id.lower()


# ----------------------------------------------------------------------------
# Groq
# ----------------------------------------------------------------------------
def conectar():
    key = st.session_state.get("api_key", "").strip()
    if not key:
        st.sidebar.warning("Ingresa tu API key de Groq.")
        return
    try:
        client = Groq(api_key=key)
        data = client.models.list().data
        metas = []
        for m in data:
            try:
                metas.append(m.model_dump())
            except Exception:
                metas.append({"id": getattr(m, "id", str(m))})
        ids = sorted(m["id"] for m in metas if es_llm(m["id"]))
        gpt = [i for i in ids if es_gpt(i)]
        otros = [i for i in ids if i not in gpt]
        st.session_state.modelos = gpt + otros
        st.session_state.modelos_meta = metas
        st.session_state.conectado = True
    except Exception as e:
        st.session_state.conectado = False
        st.sidebar.error(f"No se pudo conectar: {e}")


def _uso(u):
    if u is None:
        return None, None, None
    return (
        getattr(u, "prompt_tokens", None),
        getattr(u, "completion_tokens", None),
        getattr(u, "total_tokens", None),
    )


def llamar(client, modelo, mensajes, p, stream=False, placeholder=None):
    """Llama al chat de Groq y devuelve texto + métricas."""
    kwargs = dict(
        model=modelo,
        messages=mensajes,
        temperature=p["temperature"],
        top_p=p["top_p"],
        max_completion_tokens=p["max_tokens"],
    )
    if p.get("seed", -1) >= 0:
        kwargs["seed"] = p["seed"]
    if p.get("stop"):
        kwargs["stop"] = p["stop"]
    if es_gpt(modelo) and p.get("reasoning") and p["reasoning"] != "por defecto":
        kwargs["extra_body"] = {"reasoning_effort": p["reasoning"]}

    t0 = time.perf_counter()
    texto, usage = "", None
    if stream:
        for chunk in client.chat.completions.create(stream=True, **kwargs):
            if chunk.choices and chunk.choices[0].delta.content:
                texto += chunk.choices[0].delta.content
                if placeholder is not None:
                    placeholder.markdown(texto + "▌")
            xg = getattr(chunk, "x_groq", None)
            if xg is not None and getattr(xg, "usage", None) is not None:
                usage = xg.usage
    else:
        resp = client.chat.completions.create(**kwargs)
        texto = resp.choices[0].message.content or ""
        usage = resp.usage
    lat = time.perf_counter() - t0

    pt, ct, tt = _uso(usage)
    estimado = False
    if ct is None:  # respaldo: estimación con tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        ct = len(enc.encode(texto, disallowed_special=()))
        pt = sum(len(enc.encode(m["content"], disallowed_special=())) for m in mensajes)
        tt = pt + ct
        estimado = True
    return {
        "texto": texto,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "latencia": lat,
        "tps": (ct / lat) if lat > 0 and ct else None,
        "estimado": estimado,
    }


def mostrar_metricas(r):
    c = st.columns(4)
    sufijo = " (≈)" if r.get("estimado") else ""
    c[0].metric("Tokens prompt" + sufijo, r["prompt_tokens"])
    c[1].metric("Tokens respuesta" + sufijo, r["completion_tokens"])
    c[2].metric("Latencia", f"{r['latencia']:.2f} s")
    c[3].metric("Tokens/s", f"{r['tps']:.0f}" if r["tps"] else "—")


# ----------------------------------------------------------------------------
# Tokenización
# ----------------------------------------------------------------------------
@st.cache_resource
def get_encoding(nombre):
    return tiktoken.get_encoding(nombre)


def encodings_disponibles():
    preferidos = ["o200k_harmony", "o200k_base", "cl100k_base", "p50k_base", "r50k_base"]
    disp = set(tiktoken.list_encoding_names())
    return [e for e in preferidos if e in disp]


def tokens_html(piezas):
    colores = ["#ffd54f66", "#4fc3f766", "#81c78466", "#f0629266", "#ba68c866"]
    spans = []
    for i, p in enumerate(piezas):
        t = html.escape(p).replace("\n", "↵")
        spans.append(
            f'<span style="background:{colores[i % 5]};padding:2px 1px;'
            f'border-radius:3px;white-space:pre-wrap;">{t}</span>'
        )
    return (
        '<div style="line-height:2.1;font-family:monospace;font-size:15px">'
        + "".join(spans)
        + "</div>"
    )


# ----------------------------------------------------------------------------
# Representaciones, similitud y embeddings
# ----------------------------------------------------------------------------
def obtener_corpus():
    lineas = [l.strip() for l in st.session_state.corpus.split("\n") if l.strip()]
    textos = list(lineas)
    etiquetas = [f"T{i + 1}" for i in range(len(lineas))]
    if st.session_state.get("incluir_gen"):
        gen = [h["texto"] for h in st.session_state.historial if h["texto"].strip()]
        textos += gen
        etiquetas += [f"G{i + 1}" for i in range(len(gen))]
    return textos, etiquetas


def embeddings_lsa(textos):
    X = TfidfVectorizer(strip_accents="unicode").fit_transform(textos)
    k = max(1, min(50, X.shape[0] - 1, X.shape[1] - 1))
    return TruncatedSVD(n_components=k, random_state=0).fit_transform(X)


@st.cache_resource(show_spinner="Cargando modelo de embeddings…")
def cargar_st(nombre):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nombre)


@st.cache_data(show_spinner="Calculando embeddings…")
def embeddings_st(nombre, textos):
    return cargar_st(nombre).encode(list(textos), normalize_embeddings=True)


def obtener_embeddings(textos):
    metodo = st.session_state.get("emb_metodo", EMB_OPCIONES[0])
    if metodo.startswith("LSA"):
        return embeddings_lsa(textos)
    nombre = metodo.split(": ", 1)[1]
    try:
        return np.asarray(embeddings_st(nombre, tuple(textos)))
    except Exception as e:
        st.warning(
            f"No se pudo usar sentence-transformers ({type(e).__name__}). "
            "Se usa LSA como alternativa."
        )
        return embeddings_lsa(textos)


def representar(textos, rep):
    if rep.startswith("BoW"):
        return CountVectorizer(strip_accents="unicode").fit_transform(textos).toarray().astype(float)
    if rep == "TF-IDF":
        return TfidfVectorizer(strip_accents="unicode").fit_transform(textos).toarray()
    return obtener_embeddings(textos)


def palabras(t):
    return set(re.findall(r"\w+", t.lower()))


def jaccard(a, b):
    A, B = palabras(a), palabras(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def levenshtein(a, b):
    a, b = a[:2000], b[:2000]
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lev_sim(a, b):
    m = max(len(a), len(b), 1)
    return 1 - levenshtein(a, b) / m


METRICAS_VECTORIALES = ["Coseno", "Euclidiana", "Manhattan", "Producto punto"]
METRICAS_TEXTO = ["Jaccard (palabras)", "Levenshtein (normalizada)"]


def matriz_metrica(textos, metrica, rep):
    if metrica in METRICAS_VECTORIALES:
        X = representar(textos, rep)
        if metrica == "Coseno":
            return cosine_similarity(X)
        if metrica == "Euclidiana":
            return euclidean_distances(X)
        if metrica == "Manhattan":
            return manhattan_distances(X)
        return X @ X.T
    f = jaccard if metrica.startswith("Jaccard") else lev_sim
    n = len(textos)
    return np.array([[f(textos[i], textos[j]) for j in range(n)] for i in range(n)])


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuración")
    st.text_input(
        "Groq API Key",
        type="password",
        key="api_key",
        placeholder="gsk_...",
        on_change=lambda: st.session_state.update(conectado=False),
        help="Se guarda solo en la sesión del navegador.",
    )
    if st.button("Conectar", type="primary"):
        conectar()
    if st.session_state.conectado:
        st.success(f"Conectado · {len(st.session_state.modelos)} LLMs disponibles")

    st.divider()
    st.subheader("Parámetros de generación")
    modelos = st.session_state.get("modelos", MODELOS_FALLBACK)
    idx = modelos.index("openai/gpt-oss-120b") if "openai/gpt-oss-120b" in modelos else 0
    modelo = st.selectbox("Modelo", modelos, index=idx)
    temperature = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.05, 1.0, 1.0, 0.05)
    max_tokens = st.slider("Máx. tokens de salida", 64, 8192, 1024, 64)
    seed = st.number_input("Seed (-1 = aleatoria)", min_value=-1, value=-1, step=1)
    stop_txt = st.text_input("Secuencias de parada (separadas por coma)")
    reasoning = "por defecto"
    if es_gpt(modelo):
        reasoning = st.selectbox("Reasoning effort (GPT-OSS)", ["por defecto", "low", "medium", "high"])
    stream = st.toggle("Streaming", value=True)

    st.divider()
    with st.expander("Embeddings (locales)"):
        st.selectbox("Método", EMB_OPCIONES, key="emb_metodo")
        st.caption("Groq no ofrece endpoint de embeddings, por eso se calculan localmente.")

params = {
    "temperature": temperature,
    "top_p": top_p,
    "max_tokens": max_tokens,
    "seed": int(seed),
    "stop": [s.strip() for s in stop_txt.split(",") if s.strip()][:4],
    "reasoning": reasoning,
}
conectado = st.session_state.conectado

# ----------------------------------------------------------------------------
# Cuerpo principal
# ----------------------------------------------------------------------------
st.title("🧪 Laboratorio de LLMs con Groq")
st.caption("Generación de texto, tokens, Bag of Words, métricas de similitud y embeddings.")

with st.expander("📝 Corpus de textos (para Bag of Words, similitud y embeddings)"):
    st.text_area("Un texto por línea", key="corpus", height=150)
    st.checkbox("Incluir respuestas generadas por los LLM (etiquetas G1, G2…)", key="incluir_gen")
    _t, _e = obtener_corpus()
    st.dataframe(pd.DataFrame({"Etiqueta": _e, "Texto": _t}), hide_index=True)

tab_mod, tab_gen, tab_cmp, tab_tok, tab_bow, tab_sim, tab_emb = st.tabs(
    [
        "🤖 Modelos",
        "✍️ Generación",
        "🌡️ Comparar",
        "🔤 Tokens e IDs",
        "🎒 Bag of Words",
        "📐 Similitud",
        "🧭 Embeddings",
    ]
)

# ------------------------------- Modelos -------------------------------------
with tab_mod:
    st.subheader("LLMs disponibles en Groq")
    if not conectado:
        st.info("Conéctate con tu API key para ver la lista real de modelos. Referencia:")
        st.write(pd.DataFrame({"id": MODELOS_FALLBACK}))
    else:
        df = pd.DataFrame(st.session_state.modelos_meta)
        df["tipo"] = df["id"].apply(lambda i: "LLM" if es_llm(i) else "Audio / Guard")
        df["familia"] = df["id"].apply(lambda i: "GPT-OSS (OpenAI)" if es_gpt(i) else "Otros")
        cols = [c for c in ["id", "owned_by", "context_window", "max_completion_tokens", "active", "tipo", "familia"] if c in df.columns]
        df = df[cols].sort_values(["familia", "id"], ascending=[False, True])
        st.dataframe(df, hide_index=True)
        if "context_window" in df.columns:
            d = df[(df["tipo"] == "LLM")].dropna(subset=["context_window"]).sort_values("context_window")
            if not d.empty:
                st.plotly_chart(
                    px.bar(d, x="context_window", y="id", color="familia", orientation="h",
                           title="Ventana de contexto por modelo (tokens)"),
                )

# ------------------------------ Generación -----------------------------------
with tab_gen:
    st.subheader("Generación de texto")
    if not conectado:
        st.info("Ingresa tu API key en la barra lateral y pulsa **Conectar**.")
    sistema = st.text_area("System prompt", "Eres un asistente útil y conciso. Responde en español.", height=80)
    prompt = st.text_area("Prompt", height=140, placeholder="Escribe aquí tu instrucción…")
    if st.button("Generar", type="primary", disabled=not (conectado and prompt.strip())):
        client = Groq(api_key=st.session_state.api_key.strip())
        msgs = [{"role": "system", "content": sistema}, {"role": "user", "content": prompt}]
        zona = st.empty()
        try:
            r = llamar(client, modelo, msgs, params, stream=stream, placeholder=zona)
            zona.empty()
            r.update(modelo=modelo, temperatura=temperature, top_p=top_p, max_tokens=max_tokens, prompt=prompt)
            st.session_state.ultima = r
            st.session_state.historial.append(r)
        except Exception as e:
            zona.empty()
            st.error(f"Error en la generación: {e}")

    if "ultima" in st.session_state:
        r = st.session_state.ultima
        st.caption(f"Modelo: `{r['modelo']}` · temperatura {r['temperatura']} · top-p {r['top_p']}")
        with st.container(border=True):
            if r["texto"].strip():
                st.markdown(r["texto"])
            else:
                st.warning("La respuesta llegó vacía. En modelos GPT-OSS el razonamiento consume tokens: sube el máximo de tokens o baja el reasoning effort.")
        mostrar_metricas(r)

    if st.session_state.historial:
        with st.expander(f"Historial ({len(st.session_state.historial)})"):
            resumen = pd.DataFrame(
                [
                    {
                        "modelo": h["modelo"],
                        "temp": h["temperatura"],
                        "top_p": h["top_p"],
                        "tokens_salida": h["completion_tokens"],
                        "latencia_s": round(h["latencia"], 2),
                        "respuesta": h["texto"][:120],
                    }
                    for h in st.session_state.historial
                ]
            )
            st.dataframe(resumen, hide_index=True)
            if st.button("Borrar historial"):
                st.session_state.historial = []
                st.session_state.pop("ultima", None)
                st.rerun()

# ------------------------------- Comparar ------------------------------------
with tab_cmp:
    st.subheader("Comparar temperaturas o modelos")
    if not conectado:
        st.info("Conéctate con tu API key para usar esta sección.")
    modo = st.radio("Comparar", ["Temperaturas (mismo modelo)", "Modelos (misma temperatura)"], horizontal=True)
    prompt_c = st.text_area("Prompt", value="Escribe una frase creativa sobre el océano.", height=90, key="prompt_cmp")
    if modo.startswith("Temperaturas"):
        temps = st.multiselect("Temperaturas (máx. 4)", [0.0, 0.3, 0.7, 1.0, 1.3, 1.6, 2.0], default=[0.0, 0.7, 1.5])
        combos = [(modelo, t) for t in temps]
    else:
        sel = st.multiselect("Modelos (máx. 4)", modelos, default=modelos[:2])
        combos = [(m, temperature) for m in sel]
    combos = combos[:4]

    if st.button("Ejecutar comparación", disabled=not (conectado and prompt_c.strip() and combos)):
        client = Groq(api_key=st.session_state.api_key.strip())
        msgs = [{"role": "user", "content": prompt_c}]
        res = []
        with st.spinner("Generando…"):
            for m, t in combos:
                try:
                    r = llamar(client, m, msgs, {**params, "temperature": t})
                    r.update(modelo=m, temperatura=t, top_p=top_p, max_tokens=max_tokens, prompt=prompt_c)
                    res.append(r)
                    st.session_state.historial.append(r)
                except Exception as e:
                    st.error(f"{m} (T={t}): {e}")
        st.session_state.cmp_res = res

    res = st.session_state.get("cmp_res", [])
    if res:
        cols = st.columns(len(res))
        for col, r in zip(cols, res):
            with col.container(border=True):
                st.markdown(f"**{r['modelo']}**  \nT = {r['temperatura']}")
                st.write(r["texto"] or "_(vacía)_")
                st.caption(f"{r['completion_tokens']} tokens · {r['latencia']:.2f} s")
        textos_r = [r["texto"] for r in res if r["texto"].strip()]
        if len(textos_r) >= 2:
            try:
                S = cosine_similarity(TfidfVectorizer(strip_accents="unicode").fit_transform(textos_r))
                media = (S.sum() - len(textos_r)) / (len(textos_r) * (len(textos_r) - 1))
                st.metric("Similitud media entre respuestas (coseno TF-IDF)", f"{media:.2f}")
                st.caption("Más cercano a 1 = respuestas más parecidas (menos diversidad).")
            except ValueError:
                pass

# ------------------------------ Tokens e IDs ---------------------------------
with tab_tok:
    st.subheader("Tokens y Token IDs")
    st.caption(
        "Tokenizadores de OpenAI vía tiktoken. Para los modelos GPT-OSS el más cercano es "
        "`o200k_harmony` / `o200k_base`; para otros modelos (Llama, etc.) el conteo es aproximado."
    )
    encs = encodings_disponibles()
    c1, c2 = st.columns([1, 3])
    enc_nombre = c1.selectbox("Codificación", encs)
    texto_tok = c2.text_area("Texto", "Los modelos de lenguaje convierten el texto en tokens.", height=100, key="txt_tok")
    enc = get_encoding(enc_nombre)
    ids = enc.encode(texto_tok, disallowed_special=())
    partes = enc.decode_tokens_bytes(ids) if hasattr(enc, "decode_tokens_bytes") else [enc.decode_single_token_bytes(i) for i in ids]
    piezas = [b.decode("utf-8", errors="replace") for b in partes]

    m = st.columns(4)
    m[0].metric("Tokens", len(ids))
    m[1].metric("Caracteres", len(texto_tok))
    m[2].metric("Palabras", len(texto_tok.split()))
    m[3].metric("Caracteres / token", f"{len(texto_tok) / len(ids):.2f}" if ids else "—")

    st.markdown("**Tokens coloreados**")
    st.markdown(tokens_html(piezas), unsafe_allow_html=True)
    st.markdown("**Token IDs**")
    st.code(str(ids))
    st.dataframe(
        pd.DataFrame(
            {"#": range(len(ids)), "Token ID": ids, "Token": [repr(p) for p in piezas], "Bytes (hex)": [b.hex(" ") for b in partes]}
        ),
        hide_index=True,
    )
    st.caption("Algunos caracteres (tildes, emojis) se dividen en varios tokens; por eso pueden verse como �.")

    with st.expander("Comparar tokenizadores"):
        filas = []
        for e in encs:
            en = get_encoding(e)
            t = en.encode(texto_tok, disallowed_special=())
            filas.append({"Codificación": e, "Vocabulario": en.n_vocab, "Tokens": len(t)})
        st.dataframe(pd.DataFrame(filas), hide_index=True)

    with st.expander("Decodificar IDs → texto"):
        ids_txt = st.text_input("IDs separados por coma", value=", ".join(map(str, ids[:8])))
        try:
            ids_in = [int(x) for x in ids_txt.split(",") if x.strip()]
            st.code(enc.decode(ids_in))
        except Exception as e:
            st.error(f"IDs inválidos: {e}")

# ------------------------------ Bag of Words ---------------------------------
with tab_bow:
    st.subheader("Bag of Words")
    textos, et = obtener_corpus()
    if not textos:
        st.info("Agrega textos al corpus.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        modo_bow = c1.selectbox("Representación", ["Conteo", "Binario", "TF-IDF"])
        ngmax = c2.slider("n-gramas (máx.)", 1, 3, 1)
        sin_acentos = c3.checkbox("Quitar acentos", value=True)
        top = c4.slider("Top términos", 5, 30, 15)
        kw = dict(ngram_range=(1, ngmax), strip_accents="unicode" if sin_acentos else None)
        if modo_bow == "TF-IDF":
            vec = TfidfVectorizer(**kw)
        else:
            vec = CountVectorizer(binary=(modo_bow == "Binario"), **kw)
        try:
            X = vec.fit_transform(textos)
            dfb = pd.DataFrame(X.toarray(), index=et, columns=vec.get_feature_names_out())
            st.caption(f"Matriz documento-término: {dfb.shape[0]} textos × {dfb.shape[1]} términos")
            st.dataframe(dfb.round(3))
            tot = dfb.sum().sort_values(ascending=False).head(top)
            st.plotly_chart(
                px.bar(x=tot.index, y=tot.values, labels={"x": "Término", "y": "Peso total"}, title=f"Top {len(tot)} términos"),
            )
            st.plotly_chart(
                px.imshow(dfb[tot.index], aspect="auto", color_continuous_scale="Blues", title="Mapa de calor (términos principales)"),
            )
        except ValueError as e:
            st.error(f"No se pudo construir la matriz: {e}")

# ------------------------------- Similitud -----------------------------------
with tab_sim:
    st.subheader("Métricas de similitud")
    textos, et = obtener_corpus()
    if len(textos) < 2:
        st.info("Necesitas al menos 2 textos en el corpus.")
    else:
        c1, c2 = st.columns(2)
        rep = c1.selectbox("Representación vectorial", ["BoW (conteo)", "TF-IDF", "Embeddings"])
        metrica = c2.selectbox("Métrica", METRICAS_VECTORIALES + METRICAS_TEXTO)
        if metrica in METRICAS_TEXTO:
            st.caption("Esta métrica se calcula sobre el texto directamente; la representación vectorial no aplica.")
        M = matriz_metrica(textos, metrica, rep)
        es_distancia = metrica in ("Euclidiana", "Manhattan")
        st.plotly_chart(
            px.imshow(
                M, x=et, y=et, text_auto=".2f",
                color_continuous_scale="Viridis_r" if es_distancia else "Viridis",
                title=f"{metrica} · {rep if metrica in METRICAS_VECTORIALES else 'texto'}",
            ),
        )
        st.caption("Euclidiana y Manhattan son distancias (menor = más parecido); el resto son similitudes.")

        st.markdown("**Comparar dos textos**")
        a, b = st.columns(2)
        ta = a.selectbox("Texto A", et, index=0)
        tb = b.selectbox("Texto B", et, index=1)
        i, j = et.index(ta), et.index(tb)
        reps = st.multiselect("Representaciones", ["BoW (conteo)", "TF-IDF", "Embeddings"], default=["BoW (conteo)", "TF-IDF"])
        filas = {}
        for r_ in reps:
            X = representar(textos, r_)
            filas[r_] = {
                "Coseno": float(cosine_similarity(X[[i]], X[[j]])[0, 0]),
                "Euclidiana": float(np.linalg.norm(X[i] - X[j])),
                "Manhattan": float(np.abs(X[i] - X[j]).sum()),
                "Producto punto": float(X[i] @ X[j]),
            }
        if filas:
            st.dataframe(pd.DataFrame(filas).round(4))
        st.dataframe(
            pd.DataFrame(
                {"Jaccard (palabras)": [jaccard(textos[i], textos[j])], "Levenshtein (normalizada)": [lev_sim(textos[i], textos[j])]}
            ).round(4),
            hide_index=True,
        )
        st.write(f"**{ta}:** {textos[i]}")
        st.write(f"**{tb}:** {textos[j]}")

# ------------------------------- Embeddings ----------------------------------
with tab_emb:
    st.subheader("Embeddings")
    textos, et = obtener_corpus()
    if len(textos) < 2:
        st.info("Necesitas al menos 2 textos en el corpus.")
    else:
        E = obtener_embeddings(textos)
        m = st.columns(3)
        m[0].metric("Textos", E.shape[0])
        m[1].metric("Dimensiones", E.shape[1])
        m[2].metric("Método", st.session_state.get("emb_metodo", EMB_OPCIONES[0]).split(":")[0])

        n_comp = min(2, E.shape[0], E.shape[1])
        P = PCA(n_components=n_comp).fit_transform(E)
        if P.shape[1] < 2:
            P = np.hstack([P, np.zeros((P.shape[0], 2 - P.shape[1]))])
        dfp = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "Etiqueta": et, "Texto": textos})
        fig = px.scatter(dfp, x="PC1", y="PC2", text="Etiqueta", hover_data=["Texto"], title="Proyección PCA 2D")
        fig.update_traces(textposition="top center", marker_size=12)
        st.plotly_chart(fig)

        with st.expander("Ver vector de un texto"):
            sel = st.selectbox("Texto", et, key="emb_vec_sel")
            st.bar_chart(pd.Series(E[et.index(sel)][:128], name="valor"))
            st.caption("Se muestran hasta las primeras 128 dimensiones.")

        st.markdown("**Búsqueda semántica**")
        consulta = st.text_input("Consulta", placeholder="p. ej. animal doméstico descansando")
        if consulta.strip():
            Eq = obtener_embeddings(textos + [consulta])
            sims = cosine_similarity(Eq[[-1]], Eq[:-1])[0]
            orden = np.argsort(-sims)
            st.dataframe(
                pd.DataFrame(
                    {"Etiqueta": [et[k] for k in orden], "Similitud coseno": sims[orden].round(4), "Texto": [textos[k] for k in orden]}
                ),
                hide_index=True,
            )
