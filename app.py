import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Concentración de Mercado - Monte Carlo",
    page_icon="📊",
    layout="wide",
)

st.title(
    "📊 Simulador Interactivo y Evaluador de Indicadores de Concentración de Mercado"
)
st.markdown(
    "**Organización Industrial** | Método de Monte Carlo y Retroalimentación Pedagógica"
)
st.write("---")

# ==========================================
# 1. FUNCIONES MATEMÁTICAS DE LOS 4 INDICADORES
# ==========================================


def calcular_indicadores(cuotas, k_val=4):
  """Calcula CRk, IHH, Dominancia (ID) y Entropía (IE) dado un vector de cuotas (0 a 1)."""
  s = np.array(cuotas)
  s_ordenadas = np.sort(s)[::-1]  # Orden descendente

  # 1. Ratio de Concentración (CRk)
  k = min(k_val, len(s))
  cr_k = np.sum(s_ordenadas[:k])

  # 2. Índice de Herfindahl-Hirschman (IHH)
  ihh = np.sum(s_ordenadas**2)

  # 3. Índice de Dominancia (ID) -> Suma de (s_i / IHH)^(...) o equivalente estándar (ej. s_1 / IHH o participación ponderada)
  # Usaremos la definición estándar de Dominancia basada en cuotas y Herfindahl
  id_indice = (
      np.sum((s_ordenadas / ihh) ** 2) if ihh > 0 else 0
  )  # o una aproximación robusta de dominancia de líder
  # Alternativa común de Dominancia: s_1 o relación ponderada
  id_dominancia = s_ordenadas[0] / ihh if ihh > 0 else 0

  # 4. Índice de Entropía (IE) -> - sum(s_i * ln(s_i)) para s_i > 0
  s_pos = s_ordenadas[s_ordenadas > 0]
  ie = -np.sum(s_pos * np.log(s_pos))

  return {
      f"CR_{k}": cr_k,
      "IHH": ihh,
      "Dominancia (ID)": id_dominancia,
      "Entropía (IE)": ie,
  }


# ==========================================
# 2. PANEL LATERAL (CONFIGURACIÓN)
# ==========================================
st.sidebar.header("⚙️ Configuración del Mercado")

# Selección del Indicador principal a evaluar
indicador_seleccionado = st.sidebar.selectbox(
    "Seleccione el Indicador de Concentración",
    ["IHH (Herfindahl-Hirschman)", "Ratio de Concentración (CRk)", "Entropía (IE)", "Dominancia (ID)"]
)

k_cr = 4
if "CRk" in indicador_seleccionado:
  k_cr = st.sidebar.slider("Valor de k para CRk", 2, 10, 4)

# Número de empresas N (2 a 100)
n_empresas = st.sidebar.slider(
    "Número de Empresas (N)", min_value=2, max_value=100, value=10
)

# Configuración de Iteraciones de Monte Carlo
st.sidebar.markdown("### 🎲 Motor Monte Carlo")
n_iteraciones = st.sidebar.number_input(
    "Número de Iteraciones", min_value=100, max_value=20000, value=1000, step=100
)

if n_iteraciones > 5000:
  st.sidebar.warning(
      "⚠️ **Advertencia de Carga Computacional:** Un volumen elevado de"
      " iteraciones puede incrementar la latencia de respuesta y el consumo de"
      " recursos de cómputo en la web[cite: 2]."
  )

# ==========================================
# 3. MOTOR DE SIMULACIÓN DE MONTE CARLO
# ==========================================
@st.cache_data
def ejecutar_monte_carlo(n, iteraciones, k_param):
  resultados = []
  # Generación de vectores aleatorios con Dirichlet (garantiza suma = 1.0 estrictamente)
  # Usamos alfa uniforme para simular cuotas de mercado distribuidas estocásticamente
  alphas = np.ones(n)
  muestras = np.random.dirichlet(alphas, size=iteraciones)

  for muestra in muestras:
    inds = calcular_indicadores(muestra, k_param)
    resultados.append(inds)

  return pd.DataFrame(resultados), muestras


df_simulacion, _ = ejecutar_monte_carlo(
    n_empresas, int(n_iteraciones), k_cr
)

# Mapear nombre del indicador seleccionado con el DataFrame
map_keys = {
    "IHH (Herfindahl-Hirschman)": "IHH",
    "Ratio de Concentración (CRk)": f"CR_{k_cr}",
    "Entropía (IE)": "Entropía (IE)",
    "Dominancia (ID)": "Dominancia (ID)",
}
col_ind_activo = map_keys[indicador_seleccionado]

# ==========================================
# 4. DEFINICIÓN DEL CASO PARTICULAR
# ==========================================
st.subheader("🏢 Definición del Caso Particular")
tipo_entrada = st.radio(
    "Seleccione el método de ingreso del caso particular:",
    ["Generación Aleatoria Puntual", "Ingreso Manual de Cuotas"],
)

if tipo_entrada == "Generación Aleatoria Puntual":
  if st.button("Generar Cuotas Aleatorias para el Caso"):
    vector_caso = np.random.dirichlet(np.ones(n_empresas))
    st.session_state["vector_caso"] = vector_caso
  elif "vector_caso" not in st.session_state:
    st.session_state["vector_caso"] = np.random.dirichlet(np.ones(n_empresas))
  caso_cuotas = st.session_state["vector_caso"]
else:
  st.markdown(
      "Ingrese las cuotas de mercado (separadas por comas, deben sumar 1.0 o"
      " 100%):"
  )
  default_str = ", ".join([f"{1.0/n_empresas:.3f}"] * n_empresas)
  input_manual = st.text_input("Vector de Cuotas ($s_i$):", value=default_str)
  try:
    caso_cuotas = [float(x.strip()) for x in input_manual.split(",")]
    if len(caso_cuotas) != n_empresas:
      st.error(f"Debe ingresar exactamente {n_empresas} valores.")
      caso_cuotas = np.ones(n_empresas) / n_empresas
    elif not np.isclose(sum(caso_cuotas), 1.0, atol=1e-2):
      # Si ingresan en escala 0-100, normalizar automáticamente a 0-1
      if sum(caso_cuotas) > 1.1:
        caso_cuotas = np.array(caso_cuotas) / sum(caso_cuotas)
        st.info(
            "Se normalizaron los valores ingresados para que sumen 1.0"
            " estrictamente."
        )
      else:
        st.warning("La suma de cuotas debería aproximarse a 1.0.")
  except:
    st.error("Error en el formato. Usando distribución equitativa por defecto.")
    caso_cuotas = np.ones(n_empresas) / n_empresas

# Mostrar tabla/gráfico corto del caso particular
df_caso_tabla = pd.DataFrame(
    {
        "Empresa": [f"Empresa {i+1}" for i in range(n_empresas)],
        "Cuota ($s_i$)": caso_cuotas,
    }
)
with st.expander("Ver detalle de cuotas del caso particular"):
  st.dataframe(df_caso_tabla.T)

# Calcular indicador para el caso particular
valores_caso = calcular_indicadores(caso_cuotas, k_cr)
valor_caso_activo = valores_caso[col_ind_activo]

# ==========================================
# 5. VISUALIZACIÓN COMPARATIVA (MONTE CARLO VS CASO)
# ==========================================
st.subheader("📈 Visualización Comparativa y Distribución Estocástica")

# Calcular percentil relativo del caso particular dentro de la simulación
percentil = (
    np.sum(df_simulacion[col_ind_activo] <= valor_caso_activo)
    / len(df_simulacion)
) * 100

fig = px.histogram(
    df_simulacion,
    x=col_ind_activo,
    nbins=30,
    title=f"Distribución de Monte Carlo para {col_ind_activo} (N={n_empresas}, Iteraciones={n_iteraciones})",
    labels={
        col_ind_activo: f"Valor del Indicador ({col_ind_activo})",
        "count": "Frecuencia (Iteraciones)",
    },
    opacity=0.75,
)

# Añadir línea vertical del caso particular
fig.add_vline(
    x=valor_caso_activo,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Caso Particular: {valor_caso_activo:.4f} (Percentil: {percentil:.1f}%)",
    annotation_position="top right",
)

fig.update_layout(bargap=0.1)
st.plotly_chart(fig, use_container_width=True)

st.info(
    f"💡 **Interpretación Posicional:** El caso particular evaluado presenta un"
    f" valor de **{valor_caso_activo:.4f}** en **{col_ind_activo}**, ubicándose"
    f" en el **percentil {percentil:.1f}%** de la distribución estocástica simulada"
    "[cite: 2, 4]."
)

# ==========================================
# 6. MÓDULO EVALUADOR Y RETROALIMENTACIÓN
# ==========================================
st.markdown("---")
st.subheader("🎯 Módulo Evaluador y Retroalimentación Pedagógica")

st.markdown(
    "De acuerdo con los umbrales estándar de concentración industrial (por"
    " ejemplo, criterios IHH: Competitivo < 0.15, Moderado 0.15-0.25, Alto >"
    " 0.25):"
)

pregunta_usuario = st.selectbox(
    "¿Cómo clasificaría el nivel de concentración de este mercado según el caso"
    " particular analizado?",
    [
        "Seleccione una opción...",
        "Mercado Altamente Competitivo / Disperso",
        "Mercado con Concentración Moderada",
        "Mercado Altamente Concentrado / Oligopólico",
    ],
)

# Evaluar umbral teórico real para el IHH o el indicador activo
if "IHH" in col_ind_activo:
  if valor_caso_activo < 0.15:
    respuesta_correcta = "Mercado Altamente Competitivo / Disperso"
    explicacion_teorica = (
        "El IHH es menor a 0.15, lo que denota una estructura de mercado"
        " altamente fragmentada y competitiva sin dominio crítico de pocas"
        " empresas."
    )
  elif valor_caso_activo <= 0.25:
    respuesta_correcta = "Mercado con Concentración Moderada"
    explicacion_teorica = (
        "El IHH se encuentra entre 0.15 y 0.25, indicando un nivel moderado de"
        " concentración industrial."
    )
  else:
    respuesta_correcta = "Mercado Altamente Concentrado / Oligopólico"
    explicacion_teorica = (
        "El IHH supera el umbral crítico de 0.25, señalando un mercado con"
        " fuerte concentración u oligopolio marcado."
    )
else:
  # Umbrales genéricos para CRk u otros
  if valor_caso_activo < 0.50:
    respuesta_correcta = "Mercado Altamente Competitivo / Disperso"
    explicacion_teorica = (
        "El indicador muestra baja concentración acumulada en las principales"
        " firmas."
    )
  elif valor_caso_activo <= 0.80:
    respuesta_correcta = "Mercado con Concentración Moderada"
    explicacion_teorica = (
        "Existe una concentración intermedia de cuotas en el mercado."
    )
  else:
    respuesta_correcta = "Mercado Altamente Concentrado / Oligopólico"
    explicacion_teorica = (
        "Las principales empresas agrupan la gran mayoría de la cuota total del"
        " mercado."
    )

if pregunta_usuario != "Seleccione una opción...":
  if pregunta_usuario == respuesta_correcta:
    st.success(
        "✅ **¡Respuesta Correcta!** Has evaluado acertadamente el nivel de"
        f" concentración. \n\n**Justificación Técnica:** {explicacion_teorica}"
        f" Adicionalmente, el caso se sitúa en el percentil {percentil:.1f}% de"
        " la simulación estocástica[cite: 2, 4]."
    )
  else:
    st.error(
        "❌ **Respuesta Incorrecta.** Revisa los fundamentos teóricos del"
        " indicador.\n\n**Justificación Técnica Correcta:**"
        f" {explicacion_teorica} El valor obtenido fue"
        f" **{valor_caso_activo:.4f}** (Percentil: {percentil:.1f}%)."
    )