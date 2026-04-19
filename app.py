"""
Recomendador de Evaluadores de Tesinas - FCEFyN UNC
Autor: Agustín Luna
Fuente de datos: tesinas.xlsx (hoja "Tesinas")
"""

import re
import unicodedata
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURACIÓN DE LA APP
# ============================================================
st.set_page_config(
    page_title="Recomendador de Evaluadores - FCEFyN",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXCEL_PATH = "tesinas.xlsx"
FUSIONES_PATH = "fusiones.txt"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Convención temporal
ULTIMO_AÑO = datetime.now().year
AÑO_3_ATRAS = ULTIMO_AÑO - 2

# Pesos del ranking
PESO_SIMILITUD = 0.70
PESO_PENAL_CARGA = 0.20
PESO_BONUS_ACTIVIDAD = 0.10

# ============================================================
# UTILIDADES DE NORMALIZACIÓN DE NOMBRES
# ============================================================
TITULO_PATTERNS = [
    r'\bdoctor[ae]?\b', r'\bdr[ae]?\b',
    r'\ben\s+ciencias?\s+(biol[oó]gicas|qu[ií]micas|de la salud|biolog[ií]a|m[eé]dicas|naturales|agr[ií]colas|qu[ií]mica)\b',
    r'\ben\s+cs\.?\s+biol[oó]gicas?\b', r'\ben\s+biolog[ií]a\b',
    r'\blicenciad[ao]\b', r'\besp(ecialista)?\b',
    r'\bprof(esor[a]?)?\b', r'\bing(eniero?|eniera)?\b',
    r'\bmg(ter)?\b', r'\bmagister\b', r'\bbi[oó]l(o?g[oa])?\b',
]
TITULO_REGEX = re.compile('|'.join(TITULO_PATTERNS), re.IGNORECASE)

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_display(d):
    """Limpia un nombre quitando títulos académicos y caracteres raros."""
    if not d or (isinstance(d, float) and pd.isna(d)):
        return ''
    s = str(d).strip()
    s = re.sub(r'\(.*?\)', '', s)
    s = TITULO_REGEX.sub(' ', s)
    s = s.replace(',', ' ')
    s = re.sub(r'\s+', ' ', s).strip(' .,;:/')
    s = re.sub(r'^y\s+', '', s, flags=re.IGNORECASE)
    return s

def build_key(display):
    """Convierte un nombre en una clave normalizada para comparar."""
    if not display:
        return ''
    s = strip_accents(display).lower()
    s = re.sub(r'[^a-z ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ============================================================
# CARGA DE DATOS (cacheada)
# ============================================================
@st.cache_data(show_spinner="Cargando datos...")
def cargar_fusiones(ruta=FUSIONES_PATH):
    """Lee el archivo de fusiones y devuelve un dict variante_key -> canonical_display."""
    fusiones = {}
    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=>' not in line:
                    continue
                variante, canonico = line.split('=>', 1)
                variante = variante.strip()
                canonico = canonico.strip()
                if variante and canonico:
                    fusiones[build_key(variante)] = canonico
    except FileNotFoundError:
        st.warning(f"No se encontró {ruta}. Los nombres no se desambiguarán.")
    return fusiones

def aplicar_fusion(nombre_raw, fusiones):
    """Dada una variante, devuelve el nombre canónico."""
    limpio = clean_display(nombre_raw)
    if not limpio:
        return ''
    key = build_key(limpio)
    return fusiones.get(key, limpio)

def parse_lista_nombres(texto, fusiones):
    """Parsea una celda con múltiples nombres separados por ';'."""
    if not texto or pd.isna(texto):
        return []
    partes = str(texto).split(';')
    out = []
    for p in partes:
        n = aplicar_fusion(p.strip(), fusiones)
        if n:
            out.append(n)
    return out

@st.cache_data(show_spinner="Leyendo Excel...")
def cargar_tesinas(ruta=EXCEL_PATH, fusiones_dict=None):
    """Carga y normaliza la hoja Tesinas."""
    if fusiones_dict is None:
        fusiones_dict = {}
    df = pd.read_excel(ruta, sheet_name='Tesinas')
    # Normalizar nombres de columnas esperadas
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl.startswith('año'): col_map[c] = 'año'
        elif 'tesinista' in cl: col_map[c] = 'tesinista'
        elif 'título' in cl or 'titulo' in cl: col_map[c] = 'titulo'
        elif cl.startswith('director'): col_map[c] = 'director'
        elif cl.startswith('codirector'): col_map[c] = 'codirector'
        elif 'propuestos' in cl: col_map[c] = 'propuestos'
        elif 'recusados' in cl: col_map[c] = 'recusados'
        elif 'tribunal 1' in cl: col_map[c] = 'trib1'
        elif 'tribunal 2' in cl: col_map[c] = 'trib2'
        elif 'tribunal 3' in cl: col_map[c] = 'trib3'
    df = df.rename(columns=col_map)

    # Normalizar valores
    df['año'] = pd.to_numeric(df['año'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['año', 'titulo']).reset_index(drop=True)
    df['titulo'] = df['titulo'].astype(str).str.strip()

    # Aplicar fusiones a nombres
    for col in ['director', 'codirector', 'trib1', 'trib2', 'trib3']:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: aplicar_fusion(v, fusiones_dict) if pd.notna(v) else '')
        else:
            df[col] = ''

    # Parsear listas multi-valor
    df['propuestos_list'] = df.get('propuestos', pd.Series([None]*len(df))).apply(lambda v: parse_lista_nombres(v, fusiones_dict))
    df['recusados_list'] = df.get('recusados', pd.Series([None]*len(df))).apply(lambda v: parse_lista_nombres(v, fusiones_dict))

    return df

@st.cache_data(show_spinner="Calculando métricas de evaluadores...")
def construir_evaluadores(df_tesinas, ultimo_año, año_3):
    """Reconstruye la tabla de evaluadores desde las tesinas."""
    eventos = []
    for _, t in df_tesinas.iterrows():
        año = int(t['año'])
        titulo = t['titulo']
        tesinista = t['tesinista']
        if t['director']:
            eventos.append(dict(año=año, persona=t['director'], rol='director', pos=0,
                                tesinista=tesinista, titulo=titulo))
        if t['codirector']:
            eventos.append(dict(año=año, persona=t['codirector'], rol='codirector', pos=0,
                                tesinista=tesinista, titulo=titulo))
        for pos, col in enumerate(['trib1','trib2','trib3'], start=1):
            v = t[col]
            if v:
                eventos.append(dict(año=año, persona=v, rol='tribunal', pos=pos,
                                    tesinista=tesinista, titulo=titulo))
    df_ev = pd.DataFrame(eventos)
    if df_ev.empty:
        return df_ev, pd.DataFrame()

    # Resumen por persona
    def resumen(g):
        tribs = g[g['rol']=='tribunal']
        dires = g[g['rol'].isin(['director','codirector'])]
        trib_3 = tribs[tribs['año']>=año_3]
        trib_u = tribs[tribs['año']==ultimo_año]
        dir_3 = dires[dires['año']>=año_3]
        dir_u = dires[dires['año']==ultimo_año]
        return pd.Series({
            'T_total': len(tribs),
            'T_titular': int((tribs['pos']==1).sum()),
            'T_vocal2': int((tribs['pos']==2).sum()),
            'T_vocal3': int((tribs['pos']==3).sum()),
            'T_ultimo': len(trib_u),
            'T_3años': len(trib_3),
            'Dir_total': len(dires),
            'Dir_director': int((dires['rol']=='director').sum()),
            'Dir_codirector': int((dires['rol']=='codirector').sum()),
            'Dir_ultimo': len(dir_u),
            'Dir_3años': len(dir_3),
            'carga_reciente': len(trib_3)+len(dir_3),
            'primer_año': int(g['año'].min()),
            'ultimo_año': int(g['año'].max()),
            'años_activos': sorted(g['año'].unique().tolist()),
            'titulos_tribunal': tribs['titulo'].tolist(),
            'titulos_direccion': dires['titulo'].tolist(),
            'todas_tesinas': g[['año','rol','pos','tesinista','titulo']].to_dict('records'),
        })
    df_pers = df_ev.groupby('persona').apply(resumen).reset_index()
    return df_ev, df_pers

# ============================================================
# MODELO DE EMBEDDINGS
# ============================================================
@st.cache_resource(show_spinner="Cargando modelo de embeddings (primera vez ~1 min)...")
def cargar_modelo():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data(show_spinner="Calculando embeddings del corpus...")
def calcular_embeddings_corpus(titulos_tuple, _modelo):
    """Calcula embeddings para una tupla de títulos (hasheable para cache)."""
    titulos = list(titulos_tuple)
    if not titulos:
        return np.zeros((0, 384))
    embs = _modelo.encode(titulos, convert_to_numpy=True, show_progress_bar=False,
                          normalize_embeddings=True)
    return embs

def similitud_coseno(a, b):
    """a: (d,) normalizado. b: (n,d) normalizado. Devuelve (n,)."""
    if a.ndim == 1:
        return b @ a
    return b @ a.T

# ============================================================
# MOTOR DE RECOMENDACIÓN
# ============================================================
def calcular_score_persona(sims_titulos, carga_reciente, ultimo_año, año_actual):
    """
    Combina similitud semántica, penalización por carga y bonus por actividad reciente.
    Devuelve: score_final (0-1), sim_max, carga, bonus
    """
    if len(sims_titulos) == 0:
        return 0.0, 0.0, 0.0, 0.0
    # Similitud: máx + promedio (capturar tanto match fuerte como diversidad temática)
    sim_max = float(np.max(sims_titulos))
    sim_mean_top3 = float(np.mean(np.sort(sims_titulos)[-3:]))  # promedio top 3
    sim_score = 0.6 * sim_max + 0.4 * sim_mean_top3
    # Penalización por carga (normalizar a [0,1] donde 0 carga = 1, 10+ carga = 0)
    penal = max(0.0, 1.0 - carga_reciente/10.0)
    # Bonus por actividad reciente
    años_inactividad = año_actual - ultimo_año
    if años_inactividad <= 1:
        bonus = 1.0
    elif años_inactividad <= 3:
        bonus = 0.7
    elif años_inactividad <= 5:
        bonus = 0.4
    else:
        bonus = 0.1
    score_final = (PESO_SIMILITUD * sim_score +
                   PESO_PENAL_CARGA * penal +
                   PESO_BONUS_ACTIVIDAD * bonus)
    return score_final, sim_score, penal, bonus

def buscar_candidatos(titulo_nuevo, director, codirector, excluir_adicionales,
                      df_pers, titulos_a_persona, modelo, embs_corpus, titulos_corpus):
    """Devuelve dos DataFrames: candidatos titulares y candidatos vocales."""
    # Embedding del título nuevo
    emb_query = modelo.encode([titulo_nuevo], convert_to_numpy=True,
                              normalize_embeddings=True)[0]
    # Similitud con todo el corpus
    sims_corpus = similitud_coseno(emb_query, embs_corpus)

    # Para cada persona, recolectar las similitudes de sus títulos
    excluidos = set()
    for n in [director, codirector] + excluir_adicionales:
        if n:
            excluidos.add(build_key(n))

    resultados = []
    for _, p in df_pers.iterrows():
        nombre = p['persona']
        if build_key(nombre) in excluidos:
            continue
        # índices de los títulos en que participó esta persona
        indices = titulos_a_persona.get(nombre, [])
        if not indices:
            continue
        sims_p = sims_corpus[indices]
        score, sim_score, penal, bonus = calcular_score_persona(
            sims_p, p['carga_reciente'], p['ultimo_año'], ULTIMO_AÑO
        )
        # Índice del título más similar (para justificación)
        idx_best = indices[int(np.argmax(sims_p))]
        titulo_match = titulos_corpus[idx_best]
        sim_best = float(np.max(sims_p))
        resultados.append({
            'persona': nombre,
            'score': score,
            'similitud_max': sim_best,
            'penal_carga': penal,
            'bonus_actividad': bonus,
            'titulo_match': titulo_match,
            'T_total': p['T_total'],
            'T_titular': p['T_titular'],
            'T_ultimo': p['T_ultimo'],
            'T_3años': p['T_3años'],
            'Dir_total': p['Dir_total'],
            'Dir_director': p['Dir_director'],
            'Dir_codirector': p['Dir_codirector'],
            'Dir_ultimo': p['Dir_ultimo'],
            'Dir_3años': p['Dir_3años'],
            'carga_reciente': p['carga_reciente'],
            'ultimo_año': p['ultimo_año'],
        })
    df_r = pd.DataFrame(resultados).sort_values('score', ascending=False)

    # Separar candidatos
    df_titulares = df_r[df_r['T_titular'] >= 1].head(15).reset_index(drop=True)
    df_vocales = df_r.head(20).reset_index(drop=True)
    return df_titulares, df_vocales

# ============================================================
# UI: ESTILOS
# ============================================================
st.markdown("""
<style>
.big-font { font-size: 20px; font-weight: bold; }
.evaluador-card {
    background-color: #F8F9FA;
    border-left: 4px solid #1F4E78;
    padding: 12px;
    margin: 8px 0;
    border-radius: 4px;
}
.titular-card {
    border-left-color: #FFA500 !important;
    background-color: #FFF8E1;
}
.score-badge {
    background-color: #1F4E78;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-weight: bold;
}
.carga-alta { color: #C62828; font-weight: bold; }
.carga-baja { color: #2E7D32; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CARGA INICIAL
# ============================================================
fusiones = cargar_fusiones()
try:
    df_tesinas = cargar_tesinas(fusiones_dict=fusiones)
except FileNotFoundError:
    st.error(f"No se encontró {EXCEL_PATH}. Asegurate de que el archivo esté en el repositorio.")
    st.stop()

df_eventos, df_pers = construir_evaluadores(df_tesinas, ULTIMO_AÑO, AÑO_3_ATRAS)

# Construir corpus de títulos y mapeo persona -> índices
titulos_corpus = df_tesinas['titulo'].tolist()
titulo_a_indices = {t: i for i, t in enumerate(titulos_corpus)}
titulos_a_persona = defaultdict(list)
for _, t in df_tesinas.iterrows():
    idx = titulo_a_indices[t['titulo']]
    for col in ['director','codirector','trib1','trib2','trib3']:
        v = t[col]
        if v:
            titulos_a_persona[v].append(idx)
# dedupe
titulos_a_persona = {k: list(set(v)) for k, v in titulos_a_persona.items()}

modelo = cargar_modelo()
embs_corpus = calcular_embeddings_corpus(tuple(titulos_corpus), modelo)

# ============================================================
# ESTADO DE SESIÓN (para navegación)
# ============================================================
if 'persona_detalle' not in st.session_state:
    st.session_state.persona_detalle = None
if 'modo' not in st.session_state:
    st.session_state.modo = 'buscar'

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🎓 Evaluadores de Tesinas")
st.sidebar.markdown("**FCEFyN - UNC**")
st.sidebar.markdown("---")

if st.sidebar.button("🔍 Buscar evaluadores", use_container_width=True):
    st.session_state.modo = 'buscar'
    st.session_state.persona_detalle = None
if st.sidebar.button("📊 Estadísticas", use_container_width=True):
    st.session_state.modo = 'stats'
    st.session_state.persona_detalle = None

st.sidebar.markdown("---")
st.sidebar.markdown("**Herramientas adicionales**")
if st.sidebar.button("👥 Ver todos los evaluadores", use_container_width=True):
    st.session_state.modo = 'explorador'
    st.session_state.persona_detalle = None
if st.sidebar.button("📚 Ver historial de tesinas", use_container_width=True):
    st.session_state.modo = 'tesinas'
    st.session_state.persona_detalle = None

st.sidebar.markdown("---")
st.sidebar.caption(f"**Corpus**: {len(df_tesinas)} tesinas | {len(df_pers)} evaluadores")
st.sidebar.caption(f"**Período**: {int(df_tesinas['año'].min())}–{int(df_tesinas['año'].max())}")
st.sidebar.caption(f"**Último año**: {ULTIMO_AÑO} | **Últimos 3 años**: {AÑO_3_ATRAS}–{ULTIMO_AÑO}")

# ============================================================
# HELPER: render tarjeta de evaluador
# ============================================================
def render_evaluador(r, es_titular=False, idx=0):
    carga_class = 'carga-alta' if r['carga_reciente'] >= 4 else ('carga-baja' if r['carga_reciente'] <= 1 else '')
    card_class = 'evaluador-card titular-card' if es_titular else 'evaluador-card'
    titular_badge = "⭐ TITULAR" if es_titular and r['T_titular'] >= 1 else ""
    col1, col2, col3 = st.columns([5, 2, 1])
    with col1:
        if st.button(f"**{r['persona']}**", key=f"btn_{idx}_{r['persona']}",
                     help="Ver perfil completo"):
            st.session_state.persona_detalle = r['persona']
            st.session_state.modo = 'perfil'
            st.rerun()
        st.caption(f"📋 Match: _{r['titulo_match'][:120]}{'...' if len(r['titulo_match'])>120 else ''}_")
    with col2:
        st.markdown(f"<span class='score-badge'>Score: {r['score']:.2f}</span>",
                    unsafe_allow_html=True)
        st.caption(f"Similitud: {r['similitud_max']:.2f}")
    with col3:
        st.caption(f"Últ. año: **{r['ultimo_año']}**")
        if titular_badge:
            st.caption(titular_badge)

    # Métricas detalladas
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.caption(f"**Tribunal**: {r['T_total']} total | {r['T_titular']} titular")
    with mc2:
        clase = 'carga-alta' if r['T_ultimo'] >= 2 else ''
        st.markdown(f"<span class='{clase}'>T en {ULTIMO_AÑO}: {r['T_ultimo']} | últ.3años: {r['T_3años']}</span>",
                    unsafe_allow_html=True)
    with mc3:
        st.caption(f"**Dirige**: {r['Dir_total']} tot ({r['Dir_director']}D+{r['Dir_codirector']}C)")
    with mc4:
        clase2 = 'carga-alta' if r['Dir_ultimo'] >= 2 else ''
        st.markdown(f"<span class='{clase2}'>Dir en {ULTIMO_AÑO}: {r['Dir_ultimo']} | últ.3años: {r['Dir_3años']}</span>",
                    unsafe_allow_html=True)
    mc5, _ = st.columns([1, 3])
    with mc5:
        clase3 = 'carga-alta' if r['carga_reciente'] >= 4 else ('carga-baja' if r['carga_reciente'] <= 1 else '')
        st.markdown(f"⚖️ **<span class='{clase3}'>Carga reciente total: {r['carga_reciente']}</span>** (T+Dir últ. 3 años)",
                    unsafe_allow_html=True)
    st.markdown("---")

# ============================================================
# MODO: BUSCAR
# ============================================================
if st.session_state.modo == 'buscar' and st.session_state.persona_detalle is None:
    st.title("🔍 Buscar evaluadores para una tesina nueva")
    st.markdown("Completá los datos de la tesina y obtené un ranking de evaluadores sugeridos basado en el historial.")

    with st.form("form_buscar"):
        titulo_nuevo = st.text_area("Título de la tesina nueva *", height=100,
                                     placeholder="Ej: Evaluación de la bioconversión de residuos orgánicos mediante larvas de Hermetia illucens...")
        c1, c2 = st.columns(2)
        with c1:
            director = st.text_input("Director *", placeholder="Ej: Juan Pérez")
        with c2:
            codirector = st.text_input("Codirector (opcional)", placeholder="Ej: Ana López")
        excluir = st.text_input(
            "Evaluadores a excluir (opcional, separados con punto y coma)",
            placeholder="Ej: García, María; Rodríguez, Pedro",
            help="Útil para excluir recusados o personas con conflicto de interés para esta tesina"
        )
        submit = st.form_submit_button("🔎 Buscar evaluadores", type="primary", use_container_width=True)

    if submit:
        if not titulo_nuevo.strip():
            st.error("Por favor, ingresá un título.")
        elif not director.strip():
            st.error("Por favor, ingresá un director.")
        else:
            excluir_lista = [s.strip() for s in excluir.split(';') if s.strip()] if excluir else []
            # Aplicar fusiones a los nombres ingresados para coincidir con canónicos
            director_canon = aplicar_fusion(director, fusiones)
            codirector_canon = aplicar_fusion(codirector, fusiones) if codirector else ''
            excluir_canon = [aplicar_fusion(n, fusiones) for n in excluir_lista]

            with st.spinner("Analizando título y calculando ranking..."):
                df_tit, df_voc = buscar_candidatos(
                    titulo_nuevo.strip(), director_canon, codirector_canon, excluir_canon,
                    df_pers, titulos_a_persona, modelo, embs_corpus, titulos_corpus
                )

            st.success(f"Búsqueda completada. Analicé el historial de **{len(df_pers)}** evaluadores.")

            # Info de exclusiones
            exc_info = [director_canon]
            if codirector_canon: exc_info.append(codirector_canon)
            exc_info.extend(excluir_canon)
            exc_info = [e for e in exc_info if e]
            if exc_info:
                st.info(f"🚫 Excluidos: {', '.join(exc_info)}")

            tab1, tab2 = st.tabs([f"⭐ Candidatos a TITULAR (pos.1) — {len(df_tit)}",
                                   f"👥 Candidatos a VOCAL (pos.2 y 3) — {len(df_voc)}"])
            with tab1:
                st.markdown("*Evaluadores con al menos 1 experiencia previa como titular (retira actas).*")
                if df_tit.empty:
                    st.warning("No se encontraron candidatos disponibles con experiencia como titular.")
                else:
                    for i, r in df_tit.iterrows():
                        render_evaluador(r, es_titular=True, idx=f"tit_{i}")
            with tab2:
                st.markdown("*Pool general para vocales. Incluye a todos los evaluadores disponibles.*")
                if df_voc.empty:
                    st.warning("No se encontraron candidatos.")
                else:
                    for i, r in df_voc.iterrows():
                        render_evaluador(r, es_titular=False, idx=f"voc_{i}")

# ============================================================
# MODO: PERFIL DE EVALUADOR
# ============================================================
elif st.session_state.persona_detalle is not None:
    nombre = st.session_state.persona_detalle
    if st.button("← Volver"):
        st.session_state.persona_detalle = None
        st.rerun()

    st.title(f"👤 {nombre}")
    pdata = df_pers[df_pers['persona']==nombre]
    if pdata.empty:
        st.error("Persona no encontrada.")
    else:
        p = pdata.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total actividad", p['T_total']+p['Dir_total'],
                  f"desde {p['primer_año']}")
        c2.metric("Como Tribunal", p['T_total'],
                  f"{p['T_titular']} titular")
        c3.metric("Dirigidas", p['Dir_total'],
                  f"{p['Dir_director']}D + {p['Dir_codirector']}C")
        c4.metric("Último año activo", p['ultimo_año'])

        st.markdown("### Métricas temporales")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric(f"T en {ULTIMO_AÑO}", p['T_ultimo'])
        t2.metric(f"T últ. 3 años", p['T_3años'])
        t3.metric(f"Dir en {ULTIMO_AÑO}", p['Dir_ultimo'])
        t4.metric(f"Dir últ. 3 años", p['Dir_3años'])

        carga_color = "🔴" if p['carga_reciente'] >= 4 else ("🟢" if p['carga_reciente'] <= 1 else "🟡")
        st.markdown(f"### {carga_color} Carga reciente total: **{p['carga_reciente']}** (últimos 3 años)")

        st.caption(f"Años activos: {', '.join(map(str, p['años_activos']))}")

        st.markdown("### Historial completo")
        hist = pd.DataFrame(p['todas_tesinas']).sort_values('año', ascending=False)
        hist['Rol'] = hist.apply(lambda r: {
            ('tribunal',1):'Tribunal TITULAR',
            ('tribunal',2):'Tribunal vocal (pos.2)',
            ('tribunal',3):'Tribunal vocal (pos.3)',
            ('director',0):'Dirección (director)',
            ('codirector',0):'Dirección (codirector)',
        }.get((r['rol'], r['pos']), r['rol']), axis=1)
        st.dataframe(hist[['año','Rol','tesinista','titulo']].rename(columns={
            'año':'Año','tesinista':'Tesinista','titulo':'Título'
        }), use_container_width=True, hide_index=True)

# ============================================================
# MODO: ESTADÍSTICAS
# ============================================================
elif st.session_state.modo == 'stats':
    st.title("📊 Estadísticas del corpus")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tesinas totales", len(df_tesinas))
    c2.metric("Evaluadores únicos", len(df_pers))
    c3.metric("Período", f"{int(df_tesinas['año'].min())}–{int(df_tesinas['año'].max())}")

    st.markdown("### Tesinas por año")
    por_año = df_tesinas.groupby('año').size().reset_index(name='N° tesinas')
    st.bar_chart(por_año, x='año', y='N° tesinas')

    st.markdown("### Top 20 evaluadores por experiencia como Tribunal")
    top = df_pers.sort_values('T_total', ascending=False).head(20)[['persona','T_total','T_titular','carga_reciente','ultimo_año']]
    top.columns = ['Evaluador','Tribunal total','Titular','Carga reciente','Último año']
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown("### Distribución de carga reciente")
    hist_carga = df_pers['carga_reciente'].value_counts().sort_index().reset_index()
    hist_carga.columns = ['Carga reciente (T+Dir últ. 3 años)', 'N° evaluadores']
    st.bar_chart(hist_carga, x='Carga reciente (T+Dir últ. 3 años)', y='N° evaluadores')

# ============================================================
# MODO: EXPLORADOR DE EVALUADORES (bajo demanda)
# ============================================================
elif st.session_state.modo == 'explorador':
    st.title("👥 Explorador de evaluadores")
    st.caption("Tabla filtrable con todos los evaluadores del corpus. Clickeá un nombre para ver su perfil.")

    c1, c2, c3 = st.columns(3)
    with c1:
        min_t_titular = st.number_input("Mín. veces como titular", 0, 20, 0)
    with c2:
        max_carga = st.number_input("Máx. carga reciente", 0, 20, 20)
    with c3:
        desde_año = st.number_input("Activo desde año", int(df_tesinas['año'].min()),
                                     ULTIMO_AÑO, int(df_tesinas['año'].min()))

    df_f = df_pers[
        (df_pers['T_titular'] >= min_t_titular) &
        (df_pers['carga_reciente'] <= max_carga) &
        (df_pers['ultimo_año'] >= desde_año)
    ].sort_values('T_total', ascending=False)

    st.caption(f"{len(df_f)} evaluadores encontrados.")
    df_show = df_f[['persona','T_total','T_titular','Dir_total','T_3años','Dir_3años','carga_reciente','ultimo_año']].copy()
    df_show.columns = ['Evaluador','T total','Titular','Dir total','T últ.3','Dir últ.3','Carga reciente','Último año']
    # Selección por radio para perfil
    st.dataframe(df_show, use_container_width=True, hide_index=True)
    st.markdown("**Ver perfil de un evaluador:**")
    selected = st.selectbox("Elegí un evaluador", [''] + df_f['persona'].tolist())
    if selected:
        st.session_state.persona_detalle = selected
        st.rerun()

# ============================================================
# MODO: HISTORIAL DE TESINAS (bajo demanda)
# ============================================================
elif st.session_state.modo == 'tesinas':
    st.title("📚 Historial de tesinas")
    st.caption("Tabla filtrable de todas las tesinas del corpus.")

    busqueda = st.text_input("Buscar en título, tesinista, director, tribunales...")
    df_t = df_tesinas.copy()
    if busqueda:
        b = strip_accents(busqueda.lower())
        def matches(row):
            texto = ' '.join([str(row[c]) for c in ['tesinista','titulo','director','codirector','trib1','trib2','trib3']])
            return b in strip_accents(texto.lower())
        df_t = df_t[df_t.apply(matches, axis=1)]

    st.caption(f"{len(df_t)} tesinas.")
    df_show = df_t[['año','tesinista','titulo','director','codirector','trib1','trib2','trib3']].rename(columns={
        'año':'Año','tesinista':'Tesinista','titulo':'Título','director':'Director','codirector':'Codirector',
        'trib1':'Trib. 1 (titular)','trib2':'Trib. 2','trib3':'Trib. 3'
    })
    st.dataframe(df_show, use_container_width=True, hide_index=True)
