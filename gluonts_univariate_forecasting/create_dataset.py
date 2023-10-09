import numpy as np
import pandas as pd


def sin_wave(t: np.ndarray, amp: float, freq: float) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * t)


signal_length: int = 128
t: np.ndarray = np.arange(signal_length)

amps: list[float] = [0.5, 1.0, 2.0, 2.5, 1.5]
freqs: list[float] = [0.01, 0.02, 0.04, 0.05, 0.03]

for i, (amp, freq) in enumerate(zip(amps, freqs)):
    signal: np.ndarray = sin_wave(t, amp, freq)
    pd.DataFrame(signal).to_csv(f"data/sin_wave_{i}.csv",
                                index=False, header=False)
