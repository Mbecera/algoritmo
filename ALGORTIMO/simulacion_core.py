import math
import numpy as np
from scipy.stats import poisson, expon, norm, chi2, kstest

# =========================================================
# 1️⃣ GENERADOR LINEAL CONGRUENCIAL
# =========================================================
class GeneradorLCG:
    def __init__(self, semilla=12345, a=1664525, c=1013904223, m=2**32):
        self.semilla = semilla
        self.a = a
        self.c = c
        self.m = m

    def siguiente(self):
        """Devuelve el siguiente número aleatorio U(0,1)."""
        self.semilla = (self.a * self.semilla + self.c) % self.m
        return self.semilla / self.m

# =========================================================
# 2️⃣ GENERADOR DE VARIABLES ALEATORIAS
# =========================================================
class GeneradorVariables:
    def __init__(self, generador: GeneradorLCG):
        self.generador = generador

    def uniforme(self, n):
        return [self.generador.siguiente() for _ in range(n)]

    def exponencial(self, lam, n):
        """Distribución Exponencial con parámetro lambda."""
        return [-math.log(1 - self.generador.siguiente()) / lam for _ in range(n)]

    def normal(self, mu, sigma, n):
        """Distribución Normal usando Box-Muller."""
        datos = []
        for _ in range(n // 2):
            u1, u2 = self.generador.siguiente(), self.generador.siguiente()
            z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
            datos.append(mu + sigma * z1)
            datos.append(mu + sigma * z2)
        if n % 2 != 0:
            u1, u2 = self.generador.siguiente(), self.generador.siguiente()
            z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            datos.append(mu + sigma * z1)
        return datos

    def poisson(self, lam, n):
        """Distribución de Poisson con media λ."""
        datos = []
        for _ in range(n):
            L = math.exp(-lam)
            k, p = 0, 1
            while p > L:
                k += 1
                p *= self.generador.siguiente()
            datos.append(k - 1)
        return datos

# =========================================================
# 3️⃣ PRUEBAS DE AJUSTE
# =========================================================
class PruebasAjuste:
    @staticmethod
    def chi_cuadrado(datos, distribucion, params):
        """Prueba Chi-cuadrado (para datos discretos, p. ej. Poisson)."""
        valores, frec_obs = np.unique(datos, return_counts=True)
        frec_esp = len(datos) * distribucion.pmf(valores, *params)
        chi2_stat = np.sum((frec_obs - frec_esp) ** 2 / frec_esp)
        gl = len(frec_obs) - len(params) - 1
        p_val = 1 - chi2.cdf(chi2_stat, gl)
        return chi2_stat, gl, p_val

    @staticmethod
    def kolmogorov_smirnov(datos, distribucion, params):
        """Prueba de Kolmogorov-Smirnov (para datos continuos)."""
        d_stat, p_val = kstest(datos, distribucion.cdf, args=params)
        return d_stat, p_val

# =========================================================
# 4️⃣ MÉTODO DE MONTE CARLO
# =========================================================
class MonteCarlo:
    @staticmethod
    def estimar_pi(n, generador: GeneradorLCG):
        """Estimación de π usando Monte Carlo."""
        dentro = 0
        for _ in range(n):
            x, y = generador.siguiente(), generador.siguiente()
            if x**2 + y**2 <= 1:
                dentro += 1
        return 4 * dentro / n
