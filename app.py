"""Detector de spam · Streamlit

Archivos que deben estar en la misma carpeta que este app.py
(o dentro de una subcarpeta `despliegue/`):
    - modelo_spam_svm.pkl     pipeline completo (limpieza + TF-IDF + SVM calibrado)
    - preprocesamiento.py     lo importa el .pkl para limpiar el texto
    - metadatos.json          umbral, métricas, versiones…
    - ejemplos_mensajes.csv   (opcional) mensajes de ejemplo

Ejecutar en local:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Detector de Spam", page_icon="📨", layout="centered")

ARCHIVO_MODELO = "modelo_spam_svm.pkl"
ARCHIVO_META = "metadatos.json"
ARCHIVO_EJEMPLOS = "ejemplos_mensajes.csv"

EJEMPLOS_FIJOS = {
    "— Escribe tu propio mensaje —": "",
    "Premio (spam)": "Congratulations! You have been selected to receive a FREE iPhone. "
                     "Click www.free-prize.com to claim now!",
    "Cuenta suspendida (spam)": "URGENT! Your account has been suspended. Call 08001234567 "
                                "immediately to verify your details.",
    "Medicinas baratas (spam)": "Get cheap meds online, no prescription needed. Save 80% today $$$",
    "Almuerzo (no spam)": "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "Diapositivas (no spam)": "Can you send me the slides from today's meeting? Thanks",
}


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
def encontrar_carpeta_modelo() -> Path:
    """Busca el .pkl junto a app.py o dentro de despliegue/."""
    base = Path(__file__).resolve().parent
    for carpeta in (base, base / "despliegue"):
        if (carpeta / ARCHIVO_MODELO).exists():
            return carpeta
    raise FileNotFoundError(
        f"No se encontró '{ARCHIVO_MODELO}' junto a app.py ni en la carpeta 'despliegue/'."
    )


@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo():
    carpeta = encontrar_carpeta_modelo()
    # El pipeline guardado referencia preprocesamiento.limpiar_texto: su carpeta debe estar en el path
    if str(carpeta) not in sys.path:
        sys.path.insert(0, str(carpeta))
    if not (carpeta / "preprocesamiento.py").exists():
        raise FileNotFoundError("Falta 'preprocesamiento.py' junto al modelo.")

    modelo = joblib.load(carpeta / ARCHIVO_MODELO)

    meta = {}
    if (carpeta / ARCHIVO_META).exists():
        meta = json.loads((carpeta / ARCHIVO_META).read_text(encoding="utf-8"))

    ejemplos = None
    if (carpeta / ARCHIVO_EJEMPLOS).exists():
        ejemplos = pd.read_csv(carpeta / ARCHIVO_EJEMPLOS)

    return modelo, meta, ejemplos


def obtener_svm_interno(modelo):
    """Devuelve (vectorizador, svc) si el modelo es lineal, para explicar la predicción."""
    try:
        vectorizador = modelo.named_steps["tfidf"]
        calibrado = modelo.named_steps["svm"].calibrated_classifiers_[0]
        svc = getattr(calibrado, "estimator", None) or getattr(calibrado, "base_estimator")
        if getattr(svc, "kernel", None) != "linear":
            return None
        return vectorizador, svc
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def predecir(modelo, textos, umbral):
    prob = modelo.predict_proba(list(textos))[:, 1]
    return prob, prob >= umbral


def explicar(piezas, texto, n=8):
    """Términos del mensaje que más empujan hacia spam y hacia no spam (solo kernel lineal)."""
    vectorizador, svc = piezas
    vector = vectorizador.transform([texto])
    if vector.nnz == 0:
        return None
    coef = svc.coef_
    coef = coef.toarray().ravel() if hasattr(coef, "toarray") else np.asarray(coef).ravel()
    terminos = vectorizador.get_feature_names_out()
    idx = vector.indices
    aporte = vector.data * coef[idx]
    tabla = pd.DataFrame({"Término": terminos[idx], "Aporte": aporte})
    hacia_spam = tabla[tabla.Aporte > 0].nlargest(n, "Aporte")
    hacia_ham = tabla[tabla.Aporte < 0].nsmallest(n, "Aporte")
    return hacia_spam, hacia_ham


def cobertura_vocabulario(modelo, texto):
    """Fracción de palabras del mensaje que el modelo conoce (sirve para detectar otro idioma)."""
    try:
        vectorizador = modelo.named_steps["tfidf"]
        palabras = [t for t in vectorizador.build_analyzer()(texto) if " " not in t and not t.startswith("token")]
        if not palabras:
            return None
        return sum(t in vectorizador.vocabulary_ for t in palabras) / len(palabras)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
try:
    modelo, meta, ejemplos_csv = cargar_modelo()
except Exception as error:
    st.error(f"No se pudo cargar el modelo: {error}")
    st.info("Revisa que `modelo_spam_svm.pkl` y `preprocesamiento.py` estén junto a `app.py` "
            "y que la versión de scikit-learn sea la misma con la que se entrenó.")
    st.stop()

UMBRAL_MODELO = float(meta.get("umbral_spam", 0.5))
piezas_lineales = obtener_svm_interno(modelo)

# ----- Encabezado -----
st.title("📨 Detector de Spam")
st.subheader("Presentado por: Aaron Lopez - Camila Dubon")
st.write("Escribe o pega un mensaje (SMS o correo) y el modelo te dirá si parece **spam**. "
         "Usa un SVM con TF-IDF.")

pestana_uno, pestana_modelo = st.tabs(["✉️ Analizar mensaje", "⚙️ Modelo"])

# ----- Pestaña 1: un mensaje -----
with pestana_uno:
    umbral = st.session_state.get("umbral", UMBRAL_MODELO)   # se ajusta en la pestaña "Modelo"
    opciones = dict(EJEMPLOS_FIJOS)
    if ejemplos_csv is not None and {"mensaje", "etiqueta"} <= set(ejemplos_csv.columns):
        for i, fila in ejemplos_csv.reset_index(drop=True).iterrows():
            etiqueta = "spam" if fila["etiqueta"] == "spam" else "no spam"
            opciones[f"Del dataset #{i + 1} ({etiqueta})"] = str(fila["mensaje"])

    def cargar_ejemplo():
        st.session_state["texto"] = opciones[st.session_state["ejemplo"]]

    st.selectbox("Probar con un ejemplo", list(opciones), key="ejemplo", on_change=cargar_ejemplo)
    texto = st.text_area("Mensaje", key="texto", height=170,
                         placeholder="Ej.: WINNER!! You have won a £1000 prize. Call 09061790121 now!")

    if st.button("Analizar mensaje", type="primary", width="stretch"):
        if not texto or not texto.strip():
            st.warning("Escribe un mensaje antes de analizar.")
        else:
            prob, es_spam = predecir(modelo, [texto], umbral)
            p = float(prob[0])

            if es_spam[0]:
                st.error(f"### 🚫 SPAM\nProbabilidad de spam: **{p:.1%}**")
            else:
                st.success(f"### ✅ NO ES SPAM\nProbabilidad de spam: **{p:.1%}**")
            st.progress(p, text=f"P(spam) = {p:.3f} · umbral = {umbral:.2f}")

            if abs(p - umbral) < 0.15:
                st.info("El resultado está cerca del umbral: el modelo no está muy seguro.")
            if len(texto.split()) < 3:
                st.info("El mensaje es muy corto: con pocas palabras la predicción es menos confiable.")
            cobertura = cobertura_vocabulario(modelo, texto)
            if cobertura is not None and cobertura < 0.5:
                st.warning(f"Solo el {cobertura:.0%} de las palabras del mensaje son conocidas por el modelo "
                           "(¿está en otro idioma?). El modelo se entrenó en inglés: toma el resultado con cautela.")

            if piezas_lineales is not None:
                explicacion = explicar(piezas_lineales, texto)
                with st.expander("🔍 ¿Por qué? Términos que más influyeron"):
                    if explicacion is None:
                        st.write("Ninguna palabra del mensaje está en el vocabulario del modelo "
                                 "(¿está en otro idioma?).")
                    else:
                        hacia_spam, hacia_ham = explicacion
                        st.caption("Los términos como `tokenurl`, `tokentelefono` o `tokendinero` "
                                   "representan enlaces, teléfonos y símbolos de dinero.")
                        c1, c2 = st.columns(2)
                        c1.markdown("**Empujan hacia spam**")
                        c1.dataframe(hacia_spam.round(3), hide_index=True, width="stretch")
                        c2.markdown("**Empujan hacia no spam**")
                        c2.dataframe(hacia_ham.round(3), hide_index=True, width="stretch")


# ----- Pestaña 2: información del modelo -----
with pestana_modelo:
    st.subheader(meta.get("nombre_modelo", "Detector de spam"))
    hp = meta.get("hiperparametros", {})
    if hp:
        st.caption(f"Kernel: `{meta.get('kernel')}` · C = {hp.get('C')} · "
                   f"class_weight = {hp.get('class_weight')}")

    metricas = meta.get("metricas_prueba", {})
    if metricas:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metricas.get('accuracy', 0):.1%}")
        c2.metric("F1 spam", f"{metricas.get('f1', 0):.1%}")
        c3.metric("Precisión", f"{metricas.get('precision', 0):.1%}")
        c4.metric("Recall", f"{metricas.get('recall', 0):.1%}")
        st.caption("Métricas en el conjunto de prueba.")

    st.divider()
    st.slider("Umbral de spam", 0.05, 0.95, UMBRAL_MODELO, 0.05, key="umbral",
              help="Probabilidad mínima para marcar un mensaje como spam. "
                   "Súbelo para tener menos falsos positivos; bájalo para atrapar más spam.")

    version_entreno = meta.get("versiones", {}).get("scikit-learn")
    if version_entreno and version_entreno != sklearn.__version__:
        st.warning(f"El modelo se entrenó con scikit-learn {version_entreno} y aquí está "
                   f"instalado {sklearn.__version__}. Pueden salir errores o predicciones distintas.")
    st.info("El modelo se entrenó con mensajes en **inglés**.")

st.caption("Modelo: TF-IDF + SVM calibrado · Proyecto de clasificación de spam")
