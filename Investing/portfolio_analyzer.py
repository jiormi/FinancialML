import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import seaborn as sns
import yfinance as yf
import warnings

warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    module="yfinance",
)


def generar_dashboard_riesgo(
    tickers,
    pesos_actuales,
    pesos_optimos,
    retornos_diarios,
    matriz_corr,
    trayectorias_mc,
    valor_total,
    var_95_usd,
):
    """Genera un dashboard visual de 4 cuadrantes con los resultados del análisis de riesgo."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#0f172a")
    fig.suptitle(
        f"DASHBOARD CUANTITATIVO DE RIESGO Y CARTERA (Capital: ${valor_total:,.2f})",
        fontsize=16,
        fontweight="bold",
        color="#f8fafc",
        y=0.98,
    )

    # 1. CUADRANTE 1: Comparativa de Pesos
    ax1 = axes[0, 0]
    ax1.set_facecolor("#1e293b")
    x = np.arange(len(tickers))
    width = 0.35

    ax1.bar(
        x - width / 2,
        pesos_actuales * 100,
        width,
        label="Actual",
        color="#38bdf8",
    )
    ax1.bar(
        x + width / 2,
        pesos_optimos * 100,
        width,
        label="Óptimo (Sharpe)",
        color="#4ade80",
    )
    ax1.set_title(
        "1. Rebalanceo de Pesos (%)",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers, color="#cbd5e1")
    ax1.set_ylabel("Porcentaje (%)", color="#cbd5e1")
    ax1.legend(facecolor="#0f172a", edgecolor="none", labelcolor="#f8fafc")
    ax1.grid(axis="y", alpha=0.15)

    # 2. CUADRANTE 2: Matriz de Correlación
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1e293b")
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax2,
        cbar=False,
        linewidths=0.5,
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax2.set_title(
        "2. Matriz de Correlación de Activos",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
    )

    # 3. CUADRANTE 3: Distribución de Retornos y VaR (95%)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1e293b")
    retornos_cartera_usd = retornos_diarios.dot(pesos_actuales) * valor_total

    ax3.hist(
        retornos_cartera_usd,
        bins=50,
        color="#3b82f6",
        alpha=0.6,
        density=True,
        edgecolor="none",
    )
    ax3.axvline(
        -var_95_usd,
        color="#ef4444",
        linestyle="--",
        linewidth=2,
        label=f"VaR 95% 1D: -${var_95_usd:,.0f}",
    )
    ax3.set_title(
        "3. Distribución de P&L Diario y VaR Limite",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xlabel("Ganancia / Pérdida Diaria ($)", color="#cbd5e1")
    ax3.legend(facecolor="#0f172a", edgecolor="none", labelcolor="#f8fafc")
    ax3.grid(alpha=0.15)

    # 4. CUADRANTE 4: Proyecciones Monte Carlo (1 año)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#1e293b")
    horizonte = trayectorias_mc.shape[1] - 1
    t = np.arange(horizonte + 1)

    # Dibujar 100 trayectorias aleatorias como fondo
    indices_muestra = np.random.choice(
        trayectorias_mc.shape[0], 100, replace=False
    )
    for i in indices_muestra:
        ax4.plot(t, trayectorias_mc[i, :], color="#38bdf8", alpha=0.08)

    # Percentiles P10, P50, P90
    p10 = np.percentile(trayectorias_mc, 10, axis=0)
    p50 = np.percentile(trayectorias_mc, 50, axis=0)
    p90 = np.percentile(trayectorias_mc, 90, axis=0)

    ax4.plot(
        t,
        p50,
        color="#f59e0b",
        linewidth=2,
        label=f"Neutral (P50): ${p50[-1]:,.0f}",
    )
    ax4.plot(
        t,
        p90,
        color="#10b981",
        linewidth=2,
        linestyle="--",
        label=f"Optimista (P90): ${p90[-1]:,.0f}",
    )
    ax4.plot(
        t,
        p10,
        color="#ef4444",
        linewidth=2,
        linestyle="--",
        label=f"Pesimista (P10): ${p10[-1]:,.0f}",
    )
    ax4.set_title(
        "4. Proyección Monte Carlo a 1 Año (Simulación)",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_xlabel("Días Bursátiles", color="#cbd5e1")
    ax4.set_ylabel("Capital ($)", color="#cbd5e1")
    ax4.legend(facecolor="#0f172a", edgecolor="none", labelcolor="#f8fafc")
    ax4.grid(alpha=0.15)

    plt.tight_layout(pad=3.0)
    plt.show()


def analizar_cartera_avanzada(
    posiciones: dict,
    periodo: str = "2y",
    tasa_libre_riesgo: float = 0.03,
    horizonte_dias: int = 252,
    num_simulaciones: int = 10000,
):
    tickers = list(posiciones.keys())
    cantidades = np.array(list(posiciones.values()))

    print("Obteniendo datos de mercado desde Yahoo Finance...\n")

    datos = yf.download(tickers, period=periodo)["Close"]
    if isinstance(datos, pd.Series):
        datos = datos.to_frame()
    datos = datos[tickers].dropna()

    # 1. Posición Actual
    ultimos_precios = datos.iloc[-1].values
    valores_posicion = ultimos_precios * cantidades
    valor_total = np.sum(valores_posicion)
    pesos_actuales = valores_posicion / valor_total

    # 2. Rendimientos, Covarianza y Correlación
    retornos_diarios = datos.pct_change().dropna()
    mu_diario = retornos_diarios.mean()
    mu_anual = (1 + mu_diario) ** 252 - 1
    cov_diaria = retornos_diarios.cov()
    cov_anual = cov_diaria * 252
    matriz_corr = retornos_diarios.corr()

    # Métricas Actuales
    ret_cartera_actual = np.dot(pesos_actuales, mu_anual)
    vol_cartera_actual = np.sqrt(
        np.dot(pesos_actuales.T, np.dot(cov_anual, pesos_actuales))
    )

    sharpe_actual = (
        ret_cartera_actual - tasa_libre_riesgo
    ) / vol_cartera_actual

    # 3. Value at Risk (VaR 95% 1D)
    z_95 = norm.ppf(0.95)
    vol_diaria_cartera = vol_cartera_actual / np.sqrt(252)

    var_95_1d_pct = z_95 * vol_diaria_cartera - (ret_cartera_actual / 252)
    var_95_1d_usd = valor_total * var_95_1d_pct
    var_95_10d_usd = var_95_1d_usd * np.sqrt(10)  # Escalado raíz del tiempo

    # B) VaR Histórico (Percentil 5% de retornos reales)
    retornos_cartera_historicos = retornos_diarios.dot(pesos_actuales)
    var_95_hist_pct = -np.percentile(retornos_cartera_historicos, 5)
    var_95_hist_usd = valor_total * var_95_hist_pct

    # 4. Optimización de Pesos (Sharpe)
    def minus_sharpe(w):
        r = np.dot(w, mu_anual)
        v = np.sqrt(np.dot(w.T, np.dot(cov_anual, w)))
        return -(r - tasa_libre_riesgo) / v

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = tuple((0.0, 1.0) for _ in range(len(tickers)))
    w0 = np.ones(len(tickers)) / len(tickers)

    res_opt = minimize(
        minus_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    pesos_optimos = res_opt.x
    ret_opt = np.dot(pesos_optimos, mu_anual)
    vol_opt = np.sqrt(np.dot(pesos_optimos.T, np.dot(cov_anual, pesos_optimos)))
    sharpe_opt = (ret_opt - tasa_libre_riesgo) / vol_opt

    # 5. Proyecciones Monte Carlo
    dt = 1 / 252
    mu_sim = ret_cartera_actual
    vol_sim = vol_cartera_actual

    Z = np.random.normal(0, 1, (num_simulaciones, horizonte_dias))
    drift = (mu_sim - 0.5 * vol_sim**2) * dt
    shock = vol_sim * np.sqrt(dt) * Z
    retornos_simulados = np.exp(drift + shock)

    trayectorias = np.zeros((num_simulaciones, horizonte_dias + 1))
    trayectorias[:, 0] = valor_total
    for t in range(1, horizonte_dias + 1):
        trayectorias[:, t] = trayectorias[:, t - 1] * retornos_simulados[:, t - 1]

    # Generar Panel de Gráficos
    generar_dashboard_riesgo(
        tickers=tickers,
        pesos_actuales=pesos_actuales,
        pesos_optimos=pesos_optimos,
        retornos_diarios=retornos_diarios,
        matriz_corr=matriz_corr,
        trayectorias_mc=trayectorias,
        valor_total=valor_total,
        var_95_usd=var_95_1d_usd,
    )

    valores_finales = trayectorias[:, -1]
    pesimista_p10 = np.percentile(valores_finales, 10)
    neutral_p50 = np.percentile(valores_finales, 50)
    optimista_p90 = np.percentile(valores_finales, 90)

    # ------------------------------------------------------------------
    # MOSTRAR RESULTADOS EN CONSOLA
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"1. RESUMEN DE POSICIONES Y PESOS (Valor Total: ${valor_total:,.2f})")
    print("=" * 70)
    df_pos = pd.DataFrame(
        {
            "Acciones": cantidades,
            "Precio Actual": ultimos_precios,
            "Valor Total ($)": valores_posicion,
            "Peso Actual (%)": pesos_actuales * 100,
            "Peso Óptimo (%)": pesos_optimos * 100,
        },
        index=tickers,
    )
    print(df_pos.round(2).to_string())

    print("\n" + "=" * 70)
    print("2. MÉTRICAS DE RIESGO: VALUE AT RISK (VaR Al 95% Confianza)")
    print("=" * 70)
    print(
        "VaR Paramétrico (1 día):   "
        f" ${var_95_1d_usd:,.2f} ({var_95_1d_pct*100:.2f}% de la cartera)"
    )
    print(
        "VaR Paramétrico (10 días): "
        f" ${var_95_10d_usd:,.2f} ({var_95_10d_usd/valor_total*100:.2f}% de la"
        " cartera)"
    )
    print(
        "VaR Histórico (1 día):     "
        f" ${var_95_hist_usd:,.2f} ({var_95_hist_pct*100:.2f}% de la cartera)"
    )

    print("\n" + "=" * 70)
    print("3. REBALANCEO: CARTERA ACTUAL VS CARTERA OPTIMIZADA")
    print("=" * 70)
    print(
        f"Retorno Anual Esperado:  Actual: {ret_cartera_actual*100:.2f}%  |"
        f"  Óptimo: {ret_opt*100:.2f}%"
    )
    print(
        f"Volatilidad (Riesgo):    Actual: {vol_cartera_actual*100:.2f}%  |"
        f"  Óptimo: {vol_opt*100:.2f}%"
    )
    print(
        f"Ratio de Sharpe (Rf=3%): Actual: {sharpe_actual:.2f}   |  Óptimo:"
        f" {sharpe_opt:.2f}"
    )

    print("\n" + "=" * 70)
    print(
        f"4. PROYECCIÓN A 1 AÑO (MONTE CARLO - {num_simulaciones:,} RUTAS)"
    )
    print("=" * 70)
    print(
        "Escenario Pesimista (Percentil 10): "
        f" ${pesimista_p10:,.2f} ({(pesimista_p10/valor_total - 1)*100:+.2f}%)"
    )
    print(
        "Escenario Neutral   (Percentil 50): "
        f" ${neutral_p50:,.2f} ({(neutral_p50/valor_total - 1)*100:+.2f}%)"
    )
    print(
        "Escenario Optimista (Percentil 90): "
        f" ${optimista_p90:,.2f} ({(optimista_p90/valor_total - 1)*100:+.2f}%)"
    )


# ==========================================
# EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    mi_cartera = {
        "ANET": 1, 
        "GOOG": 2, 
        "NVDA": 1, 
        }
    analizar_cartera_avanzada(posiciones=mi_cartera, periodo="2y")
