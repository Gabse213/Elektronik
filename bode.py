import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# === CSV-Datei einlesen ===
df = pd.read_csv("Data/reduced_scope_38_RLC_100kHz.csv")  # <- Pfad anpassen
#print(df.columns)
#exit()
t = df["second"].values
u1 = df["Volt1"].values  # Eingang
u2 = df["Volt2"].values  # Ausgang

# === Zeitintervall und Samplingfrequenz berechnen ===
dt = t[1] - t[0]        # Zeit zwischen zwei Messpunkten
fs = 1.0 / dt           # Samplingrate in Hz
n = len(t)              # Anzahl der Samples

# === FFT berechnen ===
U1_fft = fft(u1)
U2_fft = fft(u2)
freq = fftfreq(n, d=dt)

# === Nur positive Frequenzen verwenden ===
mask = freq > 0
freq = freq[mask]
U1_fft = U1_fft[mask]
U2_fft = U2_fft[mask]

# === Übertragungsfunktion ===
H = U2_fft / U1_fft
gain_dB = 20 * np.log10(np.abs(H))
phase_deg = np.angle(H, deg=True)

# === Bode-Diagramm plotten ===
plt.figure()
plt.semilogx(freq, gain_dB)
plt.title("Bode-Diagramm – Amplitude")
plt.xlabel("Frequenz [Hz]")
plt.ylabel("Amplitude [dB]")
plt.grid(True, which='both')

plt.figure()
plt.semilogx(freq, phase_deg)
plt.title("Bode-Diagramm – Phase")
plt.xlabel("Frequenz [Hz]")
plt.ylabel("Phase [°]")
plt.grid(True, which='both')

plt.show()