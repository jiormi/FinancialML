import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats
import warnings

warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    module="yfinance",
)


def calcular_rsi(series, period=14):
  delta = series.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))

def obtener_parametros_mercado():
    """Obtiene la Tasa Libre de Riesgo (Bono 10Y) y la Prima de Mercado."""
    tnx = yf.Ticker("^TNX")
    df_tnx = tnx.history(period="5d")

    if not df_tnx.empty:
        rf_annual = df_tnx["Close"].iloc[-1] / 1000.0
    else:
        rf_annual = 0.042  # Fallback

    market_premium = 0.050  # Prima de Riesgo Consenso (Damodaran)
    return rf_annual, market_premium

def evaluar_necesidad_jump_diffusion(log_returns, umbral_std=2.5, umbral_curtosis=3.5):
    """Evalúa la presencia de saltos extremos (Drops/Rallies) en la serie.

    Retorna True si detecta colas pesadas o saltos frecuentes.
    """
    media = log_returns.mean()
    std = log_returns.std()

    # 1. Conteo de retornos atípicos (> N desviaciones estándar)
    outliers = np.abs(log_returns - media) > (umbral_std * std)
    proporcion_saltos = np.sum(outliers) / len(log_returns)

    # 2. Exceso de Curtosis (Normal = 0 en scipy stats, 3 en definición clásica)
    curtosis_exceso = stats.kurtosis(log_returns)

    # Criterio de activación: si la curtosis es alta o más del 3% de días son saltos
    requiere_jumps = (curtosis_exceso > umbral_curtosis) or (
        proporcion_saltos > 0.03
    )

    return requiere_jumps, proporcion_saltos, curtosis_exceso


def simular_merton_jump_diffusion(
    S0, drift_anual, vol_anual, dias, n_sim, lambda_jumps, mu_j, sigma_j
):
    """Simulador Monte Carlo para Merton Jump-Diffusion (MJD)."""
    dt = 1 / 252

    # Compensación del drift para neutralizar la media esperada de los saltos (Itô)
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_mjd = (drift_anual - 0.5 * vol_anual**2 - lambda_jumps * k) * dt

    # 1. Componente Difusivo Continuo
    Z = np.random.normal(0, 1, size=(dias, n_sim))
    incrementos_difusivos = np.exp(drift_mjd + vol_anual * np.sqrt(dt) * Z)

    # 2. Componente Discontinuo de Saltos (Poisson)
    # Número de saltos por día por trayectoria
    n_saltos = np.random.poisson(lambda_jumps * dt, size=(dias, n_sim))

    # Magnitud acumulada de los saltos
    magnitud_saltos = np.ones((dias, n_sim))
    indices_saltos = n_saltos > 0

    if np.any(indices_saltos):
        # Para simplificar la vectorización, aproximamos la magnitud según el n° de saltos
        for i in range(1, np.max(n_saltos) + 1):
            mask = n_saltos == i
            if np.any(mask):
                # Suma de variables normales N(mu_j * i, sigma_j^2 * i)
                magnitud_saltos[mask] = np.exp(
                    np.random.normal(mu_j * i, sigma_j * np.sqrt(i), size=np.sum(mask))
                )

    # Trayectoria Total combinada
    incrementos_totales = incrementos_difusivos * magnitud_saltos
    precios_trayectoria = S0 * np.cumprod(incrementos_totales, axis=0)

    return precios_trayectoria


def simular_gbm_capm(S0, drift_anual, vol_anual, dias, n_sim):
    """Simulador Monte Carlo para GBM Estándar."""
    dt = 1 / 252
    drift_diario = (drift_anual - 0.5 * vol_anual**2) * dt
    vol_diaria = vol_anual * np.sqrt(dt)

    Z = np.random.normal(0, 1, size=(dias, n_sim))
    incrementos = np.exp(drift_diario + vol_diaria * Z)
    precios_trayectoria = S0 * np.cumprod(incrementos, axis=0)

    return precios_trayectoria

def simular_gbm_capm_momentum(
    S0, 
    rf_annual, 
    beta, 
    market_premium, 
    vol_anual, 
    momentum_reciente, 
    dias, 
    n_sim, 
    half_life_dias=63
):
    """
    Simulación GBM combinando CAPM (Largo Plazo) y Momentum (Corto Plazo).
    
    - momentum_reciente: Retorno anualizado reciente de la acción (ej. 0.35 para +35% en el último año).
    - half_life_dias: Días para que el exceso de momentum pierda la mitad de su fuerza 
                      (63 días = ~3 meses de persistencia del momentum).
    """
    dt = 1 / 252
    
    # 1. Drift de equilibrio a largo plazo vía CAPM
    drift_capm = rf_annual + beta * market_premium
    
    # 2. Exceso de rendimiento actual respecto al CAPM
    exceso_momentum = momentum_reciente - drift_capm
    
    # 3. Constante de decaimiento (lambda) basada en el tiempo de vida media (half-life)
    lambda_decay = np.log(2) / (half_life_dias / 252)
    
    # 4. Vector temporal para decaimiento del momentum
    t = np.arange(1, dias + 1)[:, np.newaxis]  # Shape: (dias, 1)
    
    # 5. Drift variable: empieza reflejando el momentum y converge hacia el CAPM
    drift_temporal = drift_capm + exceso_momentum * np.exp(-lambda_decay * (t * dt))
    
    # 6. Cálculo de incrementos GBM
    drift_diario = (drift_temporal - 0.5 * vol_anual**2) * dt
    vol_diaria = vol_anual * np.sqrt(dt)

    Z = np.random.normal(0, 1, size=(dias, n_sim))
    incrementos = np.exp(drift_diario + vol_diaria * Z)
    
    precios_trayectoria = S0 * np.cumprod(incrementos, axis=0)

    return precios_trayectoria

def simular_proyeccion_adaptativa(
    ticker_symbol, n_simulaciones=10000, dias_proyeccion=126, periodo_historial="2y"
):
    """Analiza la serie temporal y aplica GBM o Merton Jump-Diffusion según la volatilidad."""
    #print(f"Descargando datos históricos para {ticker_symbol.upper()}...")
    tk = yf.Ticker(ticker_symbol)
    df = tk.history(period=periodo_historial)

    if df.empty:
        print(f"No hay datos para {ticker_symbol} continuo con otras empresas...")
        return 0,0

    precio_actual = df["Close"].iloc[-1]
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

    # 1. Obtener Parámetros de Mercado y Beta
    rf_annual, market_premium = obtener_parametros_mercado()
    beta = tk.info.get("beta", 1.0)
    beta = 1.0 if beta is None else beta

    # Drift mediante CAPM para evitar sesgo histórico de tendencia
    drift_anual_capm = rf_annual + beta * market_premium
    vol_anual = log_returns.std() * np.sqrt(252)

    # 2. Diagnóstico Estadístico de la Serie
    usar_jumps, prop_saltos, curtosis = evaluar_necesidad_jump_diffusion(log_returns)

    print("\n" + "=" * 55)
    print(f" DIAGNÓSTICO DE VOLATILIDAD Y SELECCIÓN DE MODELO: {ticker_symbol.upper()}")
    print("=" * 55)
    print(f"Exceso de Curtosis: {curtosis:.2f} (Ref Normal: 0.00)")
    print(f"Proporción de días con saltos extremos (>2.5σ): {prop_saltos * 100:.2f}%")

    # 3. Ejecución del Modelo Seleccionado
    if usar_jumps:
        print("\n[ESTADO]: Detectada alta frecuencia de colas pesadas / saltos bruscos.")
        print(" -> Modelo Seleccionado: MERTON JUMP-DIFFUSION (MJD)")

        # Estimación heurística de parámetros del salto
        std_diaria = log_returns.std()
        media_diaria = log_returns.mean()
        mask_saltos = np.abs(log_returns - media_diaria) > (2.0 * std_diaria)

        retornos_saltos = log_returns[mask_saltos]
        lambda_jumps = len(retornos_saltos) / (len(log_returns) / 252)  # Frecuencia anual
        mu_j = retornos_saltos.mean() if len(retornos_saltos) > 0 else -0.02
        sigma_j = retornos_saltos.std() if len(retornos_saltos) > 0 else 0.05

        trayectorias = simular_merton_jump_diffusion(
            S0=precio_actual,
            drift_anual=drift_anual_capm,
            vol_anual=vol_anual,
            dias=dias_proyeccion,
            n_sim=n_simulaciones,
            lambda_jumps=lambda_jumps,
            mu_j=mu_j,
            sigma_j=sigma_j,
        )
    else:
        print("\n[ESTADO]: Comportamiento de precios relativamente continuo.")
        print(" -> Modelo Seleccionado: GEOMETRIC BROWNIAN MOTION (GBM + CAPM)")

        #trayectorias = simular_gbm_capm(
         #   S0=precio_actual,
         #   drift_anual=drift_anual_capm,
         #   vol_anual=vol_anual,
         #   dias=dias_proyeccion,
         #   n_sim=n_simulaciones,
        #)

        retorno_6m = (df["Close"].iloc[-1] / df["Close"].iloc[-126]) - 1
        momentum_reciente = (1 + retorno_6m)**(252 / 126) - 1 # 6 meses ultimos

        trayectorias = simular_gbm_capm_momentum(
          S0=precio_actual, 
          rf_annual=rf_annual, 
          beta=beta, 
          market_premium=market_premium, 
          vol_anual=vol_anual, 
          momentum_reciente=momentum_reciente, 
          dias=dias_proyeccion, 
          n_sim=n_simulaciones,           
        )

    # 4. Cálculo de Resultados
    precios_finales = trayectorias[-1]
    p_medio = np.mean(precios_finales)
    mediana = np.median(precios_finales)
    p5 = np.percentile(precios_finales, 5)
    p95 = np.percentile(precios_finales, 95)
    p15 = 0.95*precio_actual
    pprecio = (p_medio-precio_actual) / p_medio

    print("-" * 55)
    print(f"Precio Actual (S0): ${precio_actual:.2f}")
    print(f"Precio Objetivo Medio (E[S_T]): ${p_medio:.2f}")
    print(f"Expectativa de subida/bajada: {pprecio*100:.2f} %")
    print(f"Mediana Esperada: ${mediana:.2f}")
    print(f"Intervalo de Confianza 90% (P5 - P95): ${p5:.2f} - ${p95:.2f}")
    print(f"Precio de salida (5% perdida): ${p15:.2f}") 

    return precios_finales, trayectorias

def evaluar_estrategia(ticker_symbol):
  print(f"\n{'='*40}\nAnalizando ticker: {ticker_symbol}\n{'='*40}")
  try:
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
  except Exception as e:
    print(f"Error al obtener información de {ticker_symbol}: {e}")
    return

def evaluar_estrategia(ticker_symbol):
  print(f"\n{'='*40}\nAnalizando ticker: {ticker_symbol}\n{'='*40}")
  try:
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
  except Exception as e:
    print(f"Error al obtener información de {ticker_symbol}: {e}")
    return

  puntuacion = 0
  total_criterios = (
      14  # Criterios automatizables cuantitativamente vía yfinance
  )

  # 1. Market Cap > 2B
  market_cap = info.get("marketCap", 0)
  cond_market_cap = market_cap and market_cap > 2e9
  if cond_market_cap:
    puntuacion += 1
  print(
      f"• Market Cap > 2B: {cond_market_cap} (Actual:"
      f" {market_cap/1e9:.2f}B USD)"
  )

  # 2. PER entre 30 y 50
  per = info.get("trailingPE", None)
  cond_per = per is not None and 30 <= per <= 50
  if cond_per:
    puntuacion += 1
  print(f"• PER (30-50): {cond_per} (Actual: {per})")

  # Descarga de datos históricos para análisis técnico
  df = ticker.history(period="2y")
  if df.empty:
    print("❌ No hay suficientes datos históricos para análisis técnico.")
    return

  # 3. SMA50 por encima de SMA200 (Tendencia alcista / Golden Cross)
  df["SMA50"] = df["Close"].rolling(window=50).mean()
  df["SMA200"] = df["Close"].rolling(window=200).mean()
  ultimo = df.iloc[-1]
  cond_sma = ultimo["SMA50"] > ultimo["SMA200"]
  if cond_sma:
    puntuacion += 1
  print(
      f"• SMA50 > SMA200: {cond_sma} (SMA50: {ultimo['SMA50']:.2f} | SMA200:"
      f" {ultimo['SMA200']:.2f})"
  )

  # 4. RSI entre 40 y 75
  df["RSI"] = calcular_rsi(df["Close"])
  rsi_actual = df["RSI"].iloc[-1]
  cond_rsi = 40 < rsi_actual < 75
  if cond_rsi:
    puntuacion += 1
  print(f"• RSI (40-75): {cond_rsi} (Actual: {rsi_actual:.2f})")

  # 5. Rendimiento a 3 meses fuerte (Momentum)
  if len(df) >= 63:
    rend_3m = (df["Close"].iloc[-1] - df["Close"].iloc[-63]) / df[
        "Close"
    ].iloc[-63]
    cond_mom = rend_3m > 0
    if cond_mom:
      puntuacion += 1
    print(f"• Rendimiento 3M positivo: {cond_mom} ({rend_3m*100:.2f}%)")
  else:
    print("• Rendimiento 3M: Datos insuficientes")

  # 5.B Rendimiento a 1 mes fuerte (Momentum)
  if len(df) >= 31:
    rend_1m = (df["Close"].iloc[-1] - df["Close"].iloc[-31]) / df[
        "Close"
    ].iloc[-31]
    cond_mom1 = rend_1m > 0
    if cond_mom1:
      puntuacion += 1
    print(f"• Rendimiento 1M positivo: {cond_mom1} ({rend_1m*100:.2f}%)")
  else:
    print("• Rendimiento 3M: Datos insuficientes")

# 5.C Fuerza Relativa vs Índice (S&P 500 / STOXX 600) > 80% de su rango anual
  try:
    divisa = info.get("currency", "USD")
    es_eeuu = (divisa == "USD") and ("." not in ticker_symbol)
    bench_symbol = "^GSPC" if es_eeuu else "^STOXX"

    # Descarga el historial del índice para el mismo periodo
    bench_df = yf.Ticker(bench_symbol).history(period="1y")

    if not df.empty and not bench_df.empty:
      # Alineamos ambas series por fecha para evitar descalces de días festivos
      df_ratio = pd.DataFrame({
          "Stock": df["Close"],
          "Bench": bench_df["Close"],
      }).dropna()

      if len(df_ratio) >= 126:  # Al menos 6 meses de datos
        # Cálculo del Ratio RS (Precio Acción / Precio Índice)
        ratio_rs = df_ratio["Stock"] / df_ratio["Bench"]

        rs_actual = ratio_rs.iloc[-1]
        rs_min = ratio_rs.min()
        rs_max = ratio_rs.max()

        # Posición porcentual del ratio en su rango histórico de 1 año
        if rs_max > rs_min:
          percentil_rs = ((rs_actual - rs_min) / (rs_max - rs_min)) * 100
        else:
          percentil_rs = 0.0

        cond_rs = percentil_rs >= 80.0
        if cond_rs:
          puntuacion += 1
        print(
            f"• Fuerza Relativa vs {bench_symbol} (>80% Rango 1Y): {cond_rs}"
            f" (Nivel Actual: {percentil_rs:.1f}%)"
        )
      else:
        print("• Fuerza Relativa: Datos insuficientes para alinear con índice")
    else:
      print(f"• Fuerza Relativa: No se pudieron descargar datos de {bench_symbol}")

  except Exception as e:
    print(f"• Fuerza Relativa: No disponible ({e})")

  # 6. Deuda / EBITDA <= 3
  total_debt = info.get("totalDebt", 0)
  ebitda = info.get("ebitda", 1)
  deuda_ebitda = (
      total_debt / ebitda if ebitda and ebitda > 0 else float("inf")
  )
  cond_deuda = deuda_ebitda <= 3
  if cond_deuda:
    puntuacion += 1
  print(f"• Deuda / EBITDA <= 3: {cond_deuda} (Ratio: {deuda_ebitda:.2f})")

  # 7. ROE y ROA altos (ROE > 10%, ROA > 5%)
  roe = info.get("returnOnEquity", 0) or 0
  roa = info.get("returnOnAssets", 0) or 0
  cond_rentabilidad = roe > 0.10 and roa > 0.05
  if cond_rentabilidad:
    puntuacion += 1
  print(
      f"• ROE (>10%) y ROA (>5%): {cond_rentabilidad} (ROE: {roe*100:.2f}% |"
      f" ROA: {roa*100:.2f}%)"
  )

  # 8. Presencia de Cash (Free Cash Flow positivo)
  fcf = info.get("freeCashflow", 0)
  cond_cash = fcf is not None and fcf > 0
  if cond_cash:
    puntuacion += 1
  print(f"• Cash / FCF positivo: {cond_cash} (FCF: {fcf})")

  # 9. Analistas recomiendan compra
  rec_key = info.get("recommendationKey", "")
  cond_analistas = rec_key in ["buy", "strong_buy"]
  if cond_analistas:
    puntuacion += 1
  print(
      f"• Recomendación de analistas (Compra): {cond_analistas} (Consenso:"
      f" {rec_key})"
  )

# 10. Sin resultados inminentes (Próximos 14 días)
  try:
    calendar = ticker.calendar
    inminente = False
    fechas_earnings = []

    if calendar is not None:
      # 1. Si yfinance devuelve un diccionario
      if isinstance(calendar, dict):
        fechas_earnings = calendar.get("Earnings Date", [])

      # 2. Si yfinance devuelve un DataFrame
      elif isinstance(calendar, pd.DataFrame) and not calendar.empty:
        if "Earnings Date" in calendar.index:
          fechas_earnings = calendar.loc["Earnings Date"].tolist()
        elif "Earnings Date" in calendar.columns:
          fechas_earnings = calendar["Earnings Date"].tolist()

      # Aseguramos que sea una lista itenable
      if not isinstance(fechas_earnings, (list, pd.Series, np.ndarray)):
        fechas_earnings = [fechas_earnings]

      # Comprobación de fechas
      hoy = pd.Timestamp.today().normalize()
      for f in fechas_earnings:
        if pd.notna(f):
          # Convertimos a timestamp, eliminamos zona horaria y quitamos la hora
          d_ts = pd.to_datetime(f).tz_localize(None).normalize()
          dias_restantes = (d_ts - hoy).days
          if 0 <= dias_restantes <= 14:
            inminente = True
            break

    cond_earnings = not inminente
  except Exception:
    # Si no hay datos (muy común en empresas europeas o chinas), asumimos True
    cond_earnings = True

  if cond_earnings:
    puntuacion += 1
  print(f"• Sin resultados inminentes (<14 días): {cond_earnings}")

  # 11. Crecimiento de Ingresos constante (últimos 2 años vía info o financiales)
  rev_growth = info.get("revenueGrowth", 0) or 0
  cond_rev = rev_growth > 0.10
  if cond_rev:
    puntuacion += 1
  print(f"• Crecimiento de ingresos YoY > 10%: {cond_rev} ({rev_growth*100}%)")

# 12. Margen Operativo Fuerte (>15%)
  op_margin = info.get("operatingMargins", 0) or 0
  cond_margin = op_margin > 0.15
  if cond_margin:
    puntuacion += 1
  print(f"• Margen Operativo > 15%: {cond_margin} (Actual: {op_margin*100:.2f}%)")

  # Evaluación del umbral del 75%
  porcentaje = (puntuacion / total_criterios) * 100
  print(f"\n--- Resultado Cuantitativo ---")
  print(f"Puntuación: {puntuacion}/{total_criterios} ({porcentaje:.1f}%)")

  if porcentaje >= 80:
    print(
        "✅ Supera el umbral del 80%. Apto para profundizar en la revisión"
        " cualitativa (liderazgo de sector, cambios gerenciales, compras de"
        " insiders)."
    )
  elif porcentaje >= 70:
    print(
        "🟠 Alcanza el 70% : posible candidato, hacer un estudio"
        " cuantitativas."
    )    
  else:
    print(
        "❌ No alcanza el 80% requerido por la estrategia en métricas"
        " cuantitativas."
    )


if __name__ == "__main__":
  # Ejemplos de prueba (ASML en Euronext Amsterdam, LVMH en París, Microsoft en EEUU)
#  lista_tickers = ["ASML.AS", "MC.PA", "MSFT", "BC.MI", "MONC.MI"]
#  lista_tickers = ["TTE.PA", "NESTE.HE","REP.MC"]
#  lista_tickers = ["REP.MC"]
  lista_tickers = ["ANET", "NVDA", "GOOGL"]
  
#  lista_tickers = ["NET", "FCX", "SNOW", "MRVL", "DELL", "CRWD", "TER", "SMTC", "RBRK"]

  
  for t in lista_tickers:
    evaluar_estrategia(t)
    print("\n" + "=" * 55)
    print(f"PREVISIÓN A 6 MESES:") 
    print("=" * 55)
    precios_finales, trayectorias = simular_proyeccion_adaptativa(t)
    print("\n" + "=" * 55)
    print(f"PREVISIÓN A 1 AÑO:") 
    print("=" * 55)
    precios_finales, trayectorias = simular_proyeccion_adaptativa(t,dias_proyeccion=252)
