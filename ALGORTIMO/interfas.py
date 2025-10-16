import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, expon, norm
from simulacion_core import GeneradorLCG, GeneradorVariables, PruebasAjuste, MonteCarlo
import time
from datetime import datetime

# =========================================================
# CONFIGURACIÓN AVANZADA - TEMA PROFESIONAL
# =========================================================
st.set_page_config(
    page_title="ESTADISTICA COMPUTACIONAL",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# =========================================================
# ESTILOS PERSONALIZADOS - TONOS PROFESIONALES
# =========================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 35px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        margin-bottom: 35px;
        border: 1px solid #34495e;
    }
    .card {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #3498db;
        margin: 12px 0;
        border: 1px solid #e3e8f0;
    }
    .success-box {
        background: #e8f5e8;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        color: #2d5016;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background: #fff8e6;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .metric-box {
        background: #f8fafc;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .professional-tab {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #465c7a;
    }
    .feature-highlight {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# FUNCIÓN: MANUAL Y GUÍA PROFESIONAL
# =========================================================
def mostrar_manual():
    """Muestra el manual completo de la aplicación con enfoque académico y profesional"""
    st.markdown("""
    # Manual de Usuario
    
    ## Propósito del Sistema
    
    Esta es una plataforma de simulación computacional orientada a la toma de decisiones empresariales y científicas.  
    Permite el análisis de datos mediante generación de variables aleatorias, pruebas de ajuste estadístico y simulaciones por el método de Monte Carlo.
    
    **Objetivos principales:**
    - Análisis predictivo y modelado de escenarios empresariales.
    - Validación estadística de procesos productivos y operativos.
    - Soporte a la toma de decisiones basada en evidencia cuantitativa.
    - Generación de reportes estadísticos confiables y reproducibles.
    
    ---
    
    ## Módulos Disponibles
    
    ### 1. Generación de Variables Aleatorias
    
    **Aplicación:** Modelado de procesos y simulación de eventos aleatorios.
    
    **Distribuciones Implementadas:**
    
    #### Distribución de Poisson
    - **Uso típico:** Conteo de llegadas o eventos por unidad de tiempo.
    - **Parámetro (λ):** Tasa promedio de ocurrencia.
    - **Ejemplo:** Número de clientes que llegan a una sucursal por hora.
    
    #### Distribución Exponencial
    - **Uso típico:** Tiempos de espera o intervalos entre eventos.
    - **Parámetro (λ):** Frecuencia de ocurrencia.
    - **Ejemplo:** Tiempo entre transacciones en un sistema financiero.
    
    #### Distribución Normal
    - **Uso típico:** Medición de variables continuas con tendencia central.
    - **Parámetros:** μ (media) y σ (desviación estándar).
    - **Ejemplo:** Variación de tiempos de producción en una planta industrial.
    
    **Flujo de Trabajo:**
    1. Seleccionar la distribución apropiada.
    2. Configurar los parámetros del modelo.
    3. Definir el tamaño de la muestra (10 a 10,000 observaciones).
    4. Visualizar el histograma y las estadísticas descriptivas.
    5. Exportar los resultados a un archivo de texto.
    
    ---
    
    ### 2. Pruebas de Ajuste Estadístico
    
    **Aplicación:** Validación de supuestos de distribución para procesos y datos empresariales.
    
    **Métodos Implementados:**
    
    #### Prueba Chi-Cuadrado (χ²)
    - **Aplicación:** Variables discretas.
    - **Hipótesis nula (H₀):** Los datos siguen la distribución teórica esperada.
    - **Criterio de aceptación:** p > 0.05 → Se acepta la hipótesis nula.
    
    #### Prueba Kolmogorov-Smirnov (KS)
    - **Aplicación:** Variables continuas.
    - **Ventaja:** Mayor sensibilidad en muestras pequeñas.
    - **Criterio de aceptación:** p > 0.05 → Ajuste aceptable al modelo teórico.
    
    **Procedimiento:**
    1. Cargar los datos desde archivo (TXT o CSV).
    2. Seleccionar la distribución teórica a evaluar.
    3. Ejecutar la prueba estadística.
    4. Analizar los resultados y el valor p obtenido.
    5. Generar un reporte de conformidad.
    
    ---
    
    ### 3. Método de Monte Carlo
    
    **Aplicación:** Estimación numérica de parámetros o métricas complejas mediante simulación repetitiva.
    
    **Fundamento:**
    - Generación de escenarios aleatorios controlados.
    - Cálculo estadístico de proporciones y estimaciones.
    - Implementación demostrativa: estimación del valor de π mediante la relación entre áreas.
    
    **Precisión de la simulación:**
    - 1,000 puntos → margen de error aproximado del 5–10%.
    - 10,000 puntos → margen de error aproximado del 1–2%.
    - 100,000 o más puntos → margen de error inferior al 0.1%.
    
    ---
    
    ## Configuración del Sistema
    
    ### Gestión de Semillas Aleatorias
    - **Automática:** Semilla generada internamente para exploración general.
    - **Manual:** Semilla definida por el usuario para reproducibilidad de resultados.
    
    **Recomendación:** Utilizar semilla manual en procesos de validación o auditoría.
    
    ---
    
    ## Casos de Uso
    
    | Sector         | Aplicación              | Distribución | Funcionalidad |
    |----------------|-------------------------|--------------|----------------|
    | Banca          | Llegada de clientes     | Poisson      | Generación     |
    | Manufactura    | Tiempos de producción   | Normal       | Prueba de ajuste |
    | Logística      | Intervalos de entrega   | Exponencial  | Generación     |
    | Salud          | Tiempos de atención     | Normal       | Prueba de ajuste |
    | Consultoría    | Análisis de riesgo      | Uniforme     | Monte Carlo    |
    
    ---
    
    ## Base Teórica
    
    **Distribución de Poisson:**  
    P(X = k) = (e^(-λ) × λ^k) / k!
    
    **Distribución Exponencial:**  
    f(x) = λe^(-λx)
    
    **Distribución Normal:**  
    f(x) = (1 / (σ√(2π))) × e^(-(x - μ)² / (2σ²))
    
    **Método de Monte Carlo:**  
    π ≈ 4 × (Área del círculo / Área del cuadrado)
    
    ---
    
    ## Estándares de Calidad
    
    - Nivel de confianza: 95%
    - Valor crítico de significancia: 0.05
    - Tamaño mínimo de muestra recomendado: 30 observaciones
    - Precisión estándar en reportes: 4 decimales
    
    ---
    """)


# =========================================================
# INICIALIZACIÓN DE SESIÓN
# =========================================================
if 'historial_resultados' not in st.session_state:
    st.session_state.historial_resultados = []

if 'sesion_activa' not in st.session_state:
    st.session_state.sesion_activa = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# =========================================================
# CABECERA PRINCIPAL
# =========================================================
st.markdown(f"""
<div class="main-header">
    <h1>Estadística Computacional</h1>
    <h3>Conceptos Preliminares, Simulación de Números Pseudoaleatorios. Teoría del Bootstrap.</h3>
    <p style="font-size: 1.1em; opacity: 0.95;">PROYECTO DE PRIMERA UNIDAD - Sesion Activa: {st.session_state.sesion_activa}</p>
    <p style="font-size: 0.9em; opacity: 0.8;">Universidad Nacional del Altiplano</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR - PANEL DE CONTROL
# =========================================================
with st.sidebar:
    st.markdown("### PANEL DE CONTROL")
    st.markdown("---")
    
    # Menú profesional
    menu = st.selectbox(
        "Seleccione modulo operativo:",
        ["Dashboard Principal", 
         "Generacion de Variables", 
         "Pruebas de Ajuste", 
         "Simulacion Monte Carlo",
         "Manual de Usuario",
         "Acerca del Sistema"]
    )
    
    st.markdown("---")
    st.markdown("### CONFIGURACION")
    
    # Tipo de semilla
    semilla_op = st.radio(
        "Gestion de Semillas:",
        ["Automatica (Exploracion)", "Manual (Auditoria)"]
    )
    
    if "Manual" in semilla_op:
        semilla_input = st.number_input(
            "Ingrese semilla de auditoria:", 
            min_value=0, 
            value=12345, 
            step=1,
            help="Valor unico para reproducibilidad en auditorias"
        )
        semilla = int(semilla_input)
    else:
        semilla = int(time.time() * 1000) % 2**32
    
    # Display semilla
    with st.container():
        st.markdown("<div class='professional-tab'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Semilla Actual", f"{semilla}")
        with col2:
            st.metric("Timestamp", f"{datetime.now().strftime('%H:%M:%S')}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### METRICAS DE SESION")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Resultados", len(st.session_state.historial_resultados))
    with col2:
        st.metric("Duracion", "Activa")
    
    st.markdown("---")
    
    # Información de contacto
    with st.expander("Informacion de Contacto"):
        st.markdown("""
        **Desarrollador:** Jonathan Yimmy Mamani Pari  
        **Email Corporativo:** vannmamani@gmail.com  
        **Telefono Oficial:** +51 917831235  
        **Supervisor Academico:** Ing. Quispe Yapo Edgardo  
        **Institucion:** Universidad Nacional del Altiplano - Puno  
        **Version:** 2.0 Profesional  
        """)

# Crear generador con semilla
try:
    gen = GeneradorLCG(semilla)
except TypeError:
    gen = GeneradorLCG()
    gen.semilla = semilla

var_gen = GeneradorVariables(gen)

# =========================================================
# DASHBOARD PRINCIPAL
# =========================================================
if "Dashboard" in menu:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Bienvenido al Sistema de Simulacion</h3>
            <p>Plataforma integral diseñada para profesionales que requieren:</p>
            <ul>
                <li>Analisis predictivo avanzado</li>
                <li>Validacion estadistica de procesos</li>
                <li>Simulacion de escenarios corporativos</li>
                <li>Generacion de reportes ejecutivos</li>
                <li>Integracion con sistemas empresariales</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Protocolo de Uso</h3>
            <p><strong>1.</strong> Revise el manual de usuario</p>
            <p><strong>2.</strong> Seleccione modulo operativo</p>
            <p><strong>3.</strong> Configure parametros del proceso</p>
            <p><strong>4.</strong> Ejecute y analice resultados</p>
            <p><strong>5.</strong> Exporte para integracion</p>
            <p><strong>6.</strong> Genere reporte ejecutivo</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tarjetas de características
    st.markdown("<h3>Modulos Disponibles</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>Generacion de Variables</h4>
            <p>Modelado de procesos operativos con distribuciones probabilisticas</p>
            <div class="feature-highlight">
                <strong>Aplicaciones:</strong>
                <br>• Llegada de clientes
                <br>• Tiempos de servicio
                <br>• Metricas de calidad
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>Pruebas de Ajuste</h4>
            <p>Validacion estadistica de supuestos de proceso</p>
            <div class="feature-highlight">
                <strong>Metodologias:</strong>
                <br>• Chi-Cuadrado (χ²)
                <br>• Kolmogorov-Smirnov
                <br>• Reportes de cumplimiento
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h4>Monte Carlo</h4>
            <p>Simulacion avanzada para estimacion de metricas</p>
            <div class="feature-highlight">
                <strong>Caracteristicas:</strong>
                <br>• Escenarios probabilisticos
                <br>• Calculo de areas complejas
                <br>• Analisis de convergencia
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Métricas rápidas de sesión
    st.markdown("---")
    st.markdown("<h3>Resumen de Sesion Activa</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Inicio de Sesion", st.session_state.sesion_activa.split()[1])
    with col2:
        st.metric("Resultados Generados", len(st.session_state.historial_resultados))
    with col3:
        st.metric("Semilla Configurada", semilla)
    with col4:
        st.metric("Modulos Disponibles", "6")

# =========================================================
# GENERACIÓN DE VARIABLES
# =========================================================
elif "Generacion" in menu:
    st.markdown("<h2>Generador de Variables Aleatorias</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tipo = st.selectbox(
            "Seleccione distribucion:",
            ["Poisson", "Exponencial", "Normal (Box-Muller)"],
            help="Seleccione segun el proceso operativo a modelar"
        )
    
    with col2:
        n = st.number_input(
            "Volumen de datos:",
            min_value=10,
            max_value=10000,
            value=1000,
            step=100,
            help="Muestras representativas para analisis empresarial"
        )
    
    st.markdown("---")
    
    # Parámetros según distribución
    col1, col2, col3 = st.columns(3)
    
    if "Poisson" in tipo:
        with col1:
            lam = st.slider("λ (tasa promedio esperada):", 0.1, 10.0, 3.0, 0.1,
                           help="Ejemplo: 3 clientes por hora en sucursal")
        datos = var_gen.poisson(lam, int(n))
        dist, params = poisson, (lam,)
        info = f"**λ = {lam}** | Tasa promedio de eventos"
        
    elif "Exponencial" in tipo:
        with col1:
            lam = st.slider("λ (tasa de ocurrencia):", 0.1, 5.0, 1.5, 0.1,
                           help="Ejemplo: 1.5 transacciones por minuto")
        datos = var_gen.exponencial(lam, int(n))
        dist, params = expon, (0, 1/lam)
        info = f"**λ = {lam}** | Tiempo promedio = {1/lam:.3f} unidades"
        
    else:  # Normal
        with col1:
            mu = st.number_input("μ (Media del proceso):", value=0.0, step=0.5,
                                help="Valor central esperado del proceso")
        with col2:
            sigma = st.number_input("σ (Variabilidad del proceso):", min_value=0.1, value=1.0, step=0.1,
                                   help="Medida de dispersion del proceso")
        datos = var_gen.normal(mu, sigma, int(n))
        dist, params = norm, (mu, sigma)
        info = f"**μ = {mu}** | **σ = {sigma}** | Proceso centralizado"
    
    # Estadísticas
    st.markdown("---")
    st.markdown("<h3>Metricas Estadisticas del Proceso</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Media Muestral", f"{np.mean(datos):.4f}")
    with col2:
        st.metric("Desviacion Estandar", f"{np.std(datos):.4f}")
    with col3:
        st.metric("Minimo Registrado", f"{np.min(datos):.4f}")
    with col4:
        st.metric("Maximo Registrado", f"{np.max(datos):.4f}")
    
    st.markdown("---")
    
    # Tabla de primeros valores
    with st.expander("Vista Ejecutiva - Primeros 20 Registros"):
        df_preview = pd.DataFrame(datos[:20], columns=["Valor Simulado"])
        df_preview['Registro'] = range(1, len(df_preview) + 1)
        df_preview['Proceso'] = tipo.split()[0]
        st.dataframe(df_preview[['Registro', 'Proceso', 'Valor Simulado']], use_container_width=True)
    
    # Visualización
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Distribucion Muestral vs Teorica</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(datos, bins=40, color='#3498db', edgecolor='#2c3e50', density=True, alpha=0.7)
        x = np.linspace(min(datos), max(datos), 200)
        try:
            y = dist.pdf(x, *params)
        except AttributeError:
            y = dist.pmf(np.round(x), *params)
        ax.plot(x, y, 'r--', lw=2.5, label="Distribucion Teorica", color='#e74c3c')
        ax.set_title(f"Analisis de Distribucion: {tipo.split('(')[0].strip()}", 
                    fontsize=14, fontweight='bold', color='#2c3e50')
        ax.set_xlabel("Valores del Proceso", fontsize=12)
        ax.set_ylabel("Densidad de Probabilidad", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h4>Analisis de Variabilidad</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        box_plot = ax.boxplot(datos, vert=True, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('#3498db')
        box_plot['boxes'][0].set_alpha(0.7)
        ax.set_ylabel("Rango de Valores", fontsize=12)
        ax.set_title("Diagrama de Caja - Analisis de Dispersion", 
                    fontsize=14, fontweight='bold', color='#2c3e50')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Descargas
    st.markdown("<h3>Exportacion de Datos</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = pd.DataFrame(datos, columns=["Valor_Simulado"]).to_csv(index=False)
        st.download_button(
            "Descargar CSV",
            data=csv_data,
            file_name=f"datos_{tipo.split()[0].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        txt_data = "\n".join(map(str, datos))
        st.download_button(
            "Descargar TXT para Analisis",
            data=txt_data,
            file_name=f"datos_{tipo.split()[0].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        st.markdown("""
        <div class="success-box">
        <strong>Datos Listos</strong><br>
        Formatos disponibles para integracion con sistemas empresariales
        </div>
        """, unsafe_allow_html=True)
    
    # Guardar en historial
    st.session_state.historial_resultados.append({
        'tipo': tipo,
        'n': n,
        'media': np.mean(datos),
        'desviacion': np.std(datos),
        'timestamp': datetime.now(),
        'modulo': 'Generacion Variables'
    })

# =========================================================
# PRUEBAS DE AJUSTE
# =========================================================
elif "Pruebas" in menu:
    st.markdown("<h2>Pruebas de Ajuste Estadistico</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <strong>Protocolo de Validacion:</strong>
    <ol>
        <li>Cargue archivo con datos del proceso (TXT/CSV)</li>
        <li>Seleccione distribucion de referencia</li>
        <li>Ejecute analisis de conformidad estadistica</li>
        <li>Interprete resultados: p > 0.05 → Cumplimiento</li>
        <li>Genere reporte de validacion</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    archivo = st.file_uploader(
        "Cargar Datos (TXT o CSV):",
        type=["txt", "csv"],
        help="Archivo con datos numericos del proceso a validar"
    )
    
    if archivo:
        try:
            datos = np.loadtxt(archivo)
        except:
            df = pd.read_csv(archivo)
            datos = df.iloc[:, 0].values
        
        st.success(f"Archivo cargado: {len(datos)} registros de proceso")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dist_sel = st.selectbox(
                "Distribucion de Referencia:",
                ["Poisson", "Exponencial", "Normal"]
            )
        
        with col2:
            if "Poisson" in dist_sel:
                prueba_tipo = "Chi-Cuadrado (χ²)"
            else:
                prueba_tipo = st.radio("Metodologia de Validacion:", 
                                     ["Kolmogorov-Smirnov", "Chi-Cuadrado"])
        
        st.markdown("---")
        
        # Realizar prueba
        if "Poisson" in dist_sel:
            lam = np.mean(datos)
            dist, params = poisson, (lam,)
            chi2_stat, gl, p_val = PruebasAjuste.chi_cuadrado(datos, dist, params)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("λ Estimado", f"{lam:.4f}")
            with col2:
                st.metric("Estadistico χ²", f"{chi2_stat:.4f}")
            with col3:
                st.metric("Grados Libertad", gl)
            with col4:
                color = "green" if p_val > 0.05 else "red"
                st.metric("Valor p", f"{p_val:.4f}", 
                         delta="Conforme" if p_val > 0.05 else "No Conforme")
        
        elif "Exponencial" in dist_sel:
            lam = 1 / np.mean(datos)
            dist, params = expon, (0, 1/lam)
            d_stat, p_val = PruebasAjuste.kolmogorov_smirnov(datos, dist, params)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("λ Estimado", f"{lam:.4f}")
            with col2:
                st.metric("Estadistico D", f"{d_stat:.4f}")
            with col3:
                st.metric("Valor p", f"{p_val:.4f}", 
                         delta="Conforme" if p_val > 0.05 else "No Conforme")
        
        else:
            mu, sigma = np.mean(datos), np.std(datos)
            dist, params = norm, (mu, sigma)
            d_stat, p_val = PruebasAjuste.kolmogorov_smirnov(datos, dist, params)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Media (μ)", f"{mu:.4f}")
            with col2:
                st.metric("Desv.Est. (σ)", f"{sigma:.4f}")
            with col3:
                st.metric("Estadistico D", f"{d_stat:.4f}")
            with col4:
                st.metric("Valor p", f"{p_val:.4f}", 
                         delta="Conforme" if p_val > 0.05 else "No Conforme")
        
        st.markdown("---")
        
        # Interpretación
        st.markdown("<h3>Dictamen de Validacion</h3>", unsafe_allow_html=True)
        
        if p_val > 0.05:
            st.markdown("""
            <div class="success-box">
            <strong>DICTAMEN FAVORABLE</strong><br>
            Los datos del proceso CUMPLEN con la distribucion de referencia seleccionada.<br>
            <em>Recomendacion: Proceder con la implementacion del modelo.</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            <strong>DICTAMEN NO FAVORABLE</strong><br>
            Los datos del proceso NO CUMPLEN con la distribucion de referencia.<br>
            <em>Recomendacion: Revisar supuestos del proceso o considerar distribucion alternativa.</em>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico de validación
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(datos, bins=30, density=True, color='#3498db', edgecolor='#2c3e50', alpha=0.7)
        x = np.linspace(min(datos), max(datos), 200)
        try:
            y = dist.pdf(x, *params)
        except AttributeError:
            y = dist.pmf(np.round(x), *params)
        ax.plot(x, y, 'r--', lw=2.5, label="Distribucion Teorica de Referencia", color='#e74c3c')
        ax.set_title(f"Validacion: {dist_sel} | Valor p = {p_val:.4f}", 
                    fontweight='bold', fontsize=14, color='#2c3e50')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig, use_container_width=True)

# =========================================================
# MONTE CARLO
# =========================================================
elif "Monte Carlo" in menu:
    st.markdown("<h2>Simulacion Monte Carlo</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <strong>Aplicacion:</strong> Metodo de simulacion utilizado para la estimacion de metricas complejas mediante generacion de escenarios probabilisticos controlados.
    <br><strong>Caso Demostrativo:</strong> Estimacion del numero π mediante la relacion de areas.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n = st.number_input(
            "Escala de Simulacion (numero de puntos):",
            min_value=100,
            max_value=500000,
            value=10000,
            step=1000,
            help="Cantidad de escenarios simulados para estimar el valor de π"
        )
    
    with col2:
        precision = st.slider(
            "Nivel de Detalle Grafico:",
            min_value=1,
            max_value=10,
            value=5,
            help="Control de la resolucion visual del grafico"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        boton_ejecutar = st.button("Ejecutar Simulacion", use_container_width=True)
    with col2:
        boton_info = st.button("Fundamentos Matematicos", use_container_width=True)
    with col3:
        boton_comparar = st.button("Analisis de Precision", use_container_width=True)
    
    if boton_info:
        st.latex(r"\pi \approx 4 \times \frac{\text{Puntos dentro del circulo}}{\text{Puntos totales en el cuadrado}}")
        st.markdown("""
        <div class="feature-highlight">
        <strong>Base Matematica:</strong> La relacion entre el area del circulo (πr²) y el area del cuadrado ((2r)²) 
        equivale a π/4. Al generar puntos aleatorios uniformemente distribuidos, la proporcion de puntos que caen dentro 
        del circulo converge al valor de π/4. Multiplicando por 4 se obtiene una estimacion de π.
        </div>
        """, unsafe_allow_html=True)

    if boton_ejecutar:
        pi_est = MonteCarlo.estimar_pi(int(n), gen)
        error = abs(np.pi - pi_est)
        error_relativo = (error / np.pi) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("π Estimado", f"{pi_est:.6f}")
        with col2:
            st.metric("π Real", f"{np.pi:.6f}")
        with col3:
            st.metric("Error Absoluto", f"{error:.6f}")
        with col4:
            st.metric("Error Relativo", f"{error_relativo:.4f}%")
        
        st.markdown("---")
        
        # Generar puntos para visualización
        puntos_x = np.array([gen.siguiente() for _ in range(int(n))])
        puntos_y = np.array([gen.siguiente() for _ in range(int(n))])
        dentro = puntos_x**2 + puntos_y**2 <= 1
        
        # Visualización
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(puntos_x[dentro], puntos_y[dentro], color="#27ae60", s=1, alpha=0.6, 
                      label="Dentro del circulo")
            ax.scatter(puntos_x[~dentro], puntos_y[~dentro], color="#e74c3c", s=1, alpha=0.6, 
                      label="Fuera del circulo")
            circle = plt.Circle((0, 0), 1, color="#2c3e50", fill=False, linewidth=2.5)
            ax.add_patch(circle)
            ax.set_aspect('equal')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title("Simulacion Monte Carlo - Distribucion de Puntos", 
                        fontweight='bold', fontsize=13, color='#2c3e50')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            # Gráfico de convergencia - CORREGIDO
            fig, ax = plt.subplots(figsize=(7, 7))
            
            # Calcular convergencia para análisis
            paso = max(1, n // 100)
            puntos_parciales = np.arange(paso, n + 1, paso)
            pi_convergencia = []
            
            for np_actual in puntos_parciales:
                dentro_parcial = np.sum((puntos_x[:np_actual]**2 + puntos_y[:np_actual]**2) <= 1)
                pi_est_parcial = 4 * dentro_parcial / np_actual
                pi_convergencia.append(pi_est_parcial)
            
            # Línea corregida - eliminado el color duplicado
            ax.plot(puntos_parciales, pi_convergencia, linewidth=2.5, 
                   label="π Estimado", color='#3498db')
            ax.axhline(y=np.pi, linestyle='--', linewidth=2.5, 
                      label="π Real", color='#e74c3c')
            ax.fill_between(puntos_parciales, np.pi - 0.1, np.pi + 0.1, 
                          alpha=0.2, color='#27ae60', label="Banda de ±0.1")
            ax.set_xlabel("Escala de Simulacion (puntos)", fontsize=11)
            ax.set_ylabel("Valor Estimado de π", fontsize=11)
            ax.set_title("Convergencia del Metodo - Analisis de Estabilidad", 
                        fontweight='bold', fontsize=13, color='#2c3e50')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            st.pyplot(fig, use_container_width=True)
    
    if boton_comparar:
        st.markdown("### Analisis de Precision vs Escala")
        
        valores_n = [100, 500, 1000, 5000, 10000, 50000, 100000]
        resultados = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, n_test in enumerate(valores_n):
            status_text.text(f"Ejecutando simulacion con {n_test:,} puntos...")
            pi_test = MonteCarlo.estimar_pi(n_test, gen)
            error = abs(np.pi - pi_test)
            error_rel = (error / np.pi) * 100
            resultados.append({
                'Escala': f"{n_test:,}",
                'π Estimado': f"{pi_test:.6f}",
                'Error Absoluto': f"{error:.6f}",
                'Error Relativo': f"{error_rel:.4f}%",
                'Precision': 'Alta' if error_rel < 1 else 'Media' if error_rel < 5 else 'Baja'
            })
            progress_bar.progress((idx + 1) / len(valores_n))
        
        status_text.text("Analisis completado")
        
        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados, use_container_width=True)
        
        # Gráfico ejecutivo de error vs escala
        fig, ax = plt.subplots(figsize=(12, 6))
        errores = [abs(np.pi - MonteCarlo.estimar_pi(n_val, gen)) for n_val in valores_n]
        ax.plot(valores_n, errores, 'o-', linewidth=2.5, markersize=10, color='#3498db', 
               markerfacecolor='#2980b9')
        ax.set_xlabel("Escala de Simulacion (puntos - escala logaritmica)", fontsize=12)
        ax.set_ylabel("Error Absoluto (escala logaritmica)", fontsize=12)
        ax.set_title("Relacion: Precision vs Escala de Simulacion", 
                    fontweight='bold', fontsize=14, color='#2c3e50')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("""
        <div class="success-box">
        <strong>Conclusiones:</strong>
        <ul>
            <li>La precision mejora significativamente con escala logaritmica</li>
            <li>Para π con 4 decimales exactos: 50,000-100,000 puntos requeridos</li>
            <li>Metodo aplicable a problemas complejos de estimacion de areas</li>
            <li>Escalabilidad demostrada para proyectos corporativos</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# MANUAL DE USUARIO
# =========================================================
elif "Manual" in menu:
    mostrar_manual()

# =========================================================
# ACERCA DEL SISTEMA
# =========================================================
elif "Sistema" in menu:
    st.markdown("""
    <div class="main-header">
        <h2>UNIVERSIDAD NACIONAL DEL ALTIPLANO</h2>
        <p>Facultad de Ingenieria Estadistica e Informatica</p>
        <p>Escuela Profesional de Ingenieria Estadistica e Informatica</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Perfil del Desarrollador</h3>
            <p><strong>Jonathan Yimmy Mamani Pari</strong></p>
            <p><strong>Email Corporativo:</strong> vannmamani@gmail.com</p>
            <p><strong>Contacto Oficial:</strong> +51 917831235</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Marco Institucional</h3>
            <p><strong>Universidad Nacional del Altiplano</strong></p>
            <p>Puno, Peru - Sede Central</p>
            <p>Ano Academico 2025</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Supervision Academica</h3>
            <p><strong>Ing. Quispe Yapo Edgardo</strong></p>
            <p><strong>Curso:</strong> Estadistica Computacional</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Arquitectura Tecnologica</h3>
            <div class="feature-highlight">
            <strong>Stack Tecnologico:</strong>
            <br>• Streamlit - Framework web
            <br>• Python 3.9+ - Lenguaje base
            <br>• NumPy / Pandas - Procesamiento de datos
            <br>• Matplotlib - Visualizacion
            <br>• SciPy - Analisis estadistico
            <br>• CSS Personalizado - Interfaz
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="card">
        <h3>Proposito del Sistema</h3>
        <p>Esta plataforma representa una solucion integral desarrollada como proyecto 
        academico del curso <strong>Simulacion Computacional</strong>. Esta diseñada para profesionales que requieren:</p>
        
        <div class="feature-highlight">
        <strong>Objetivos Estrategicos:</strong>
        <br>• Comprender generadores de numeros aleatorios (LCG) en entornos corporativos
        <br>• Generar variables de distribuciones probabilisticas para modelado de procesos
        <br>• Aplicar pruebas estadisticas de bondad de ajuste para validacion de supuestos
        <br>• Experimentar con simulacion Monte Carlo para estimacion de metricas complejas
        <br>• Visualizar y analizar resultados en tiempo real para toma de decisiones
        <br>• Exportar datos para integracion con sistemas empresariales existentes
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="warning-box">
        <h3>Marco Legal y Terminos de Uso</h3>
        <p>Este software se proporciona con propositos educativos y de investigacion.</p>
        <p><strong>Responsabilidades del Usuario:</strong> Los usuarios son responsables del uso adecuado de esta herramienta 
        en sus contextos operativos especificos.</p>
        <p><strong>Limitacion de Garantia:</strong> No hay garantia absoluta de precision en todos los calculos. 
        Se recomienda validacion cruzada para aplicaciones criticas.</p>
        <p><strong>Soporte Tecnico:</strong> Para informacion adicional, reporte de incidencias o requerimientos 
        de personalizacion, contacte al desarrollador.</p>
    </div>
    """, unsafe_allow_html=True)

    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9em; padding: 20px;">
        <p>hola </p>
        <p>© 2025 Jonathan Yimmy Mamani Pari - Universidad Nacional del Altiplano</p>
        <p style="font-size: 0.8em; margin-top: 10px;">Todos los derechos reservados</p>
    </div>
    """, unsafe_allow_html=True)