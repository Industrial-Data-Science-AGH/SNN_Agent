import serial
import wave
import numpy as np
import time

# --- KONFIGURACJA ---
PORT = 'COM3' 
BAUD = 115200 # Przy 100Hz to aż nadto
SOUND_FILE = "sounds/dog.wav"
WIN_MS = 10  # Okno 10ms

def play_wav_fast():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        time.sleep(2)
        ser.write(b'SYN\n')
        print("Synchronizacja...")
    except Exception as e:
        print(f"Blad: {e}"); return

    # 1. Wczytanie i obliczenie ENERGII w oknach 10ms
    with wave.open(SOUND_FILE, 'rb') as wav:
        fs = wav.getframerate()
        samples = np.frombuffer(wav.readframes(-1), dtype=np.int16).astype(float)
        # Normalizacja do 0.0 - 1.0
        samples /= 32768.0
        
        chunk_size = int(fs * (WIN_MS / 1000))
        # Obliczamy RMS dla każdego okna 10ms
        energy_frames = []
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            if len(chunk) == 0: break
            rms = np.sqrt(np.mean(chunk**2))
            # Mapujemy na 0-1023 dla Arduino
            energy_frames.append(int(min(rms * 5000, 1023)))

    print(f"Plik przetworzony. Wysyłam {len(energy_frames)} klatek energii (10ms każda)...")
    
    start_time = time.time()
    for val in energy_frames:
        # Wysyłamy energię jako tekst (szybkie, bo tylko 100 razy na sek)
        ser.write(f"{val}\n".encode())
        
        # Odbieramy 3 parametry [Peak, Mean, Std]
        response = ser.readline().decode(errors='ignore').strip()
        if response:
            print(f"Energy: {val:4d} | SNN Stats: {response.replace('\t', ' | ')}")

    print(f"\nGotowe! Czas rzeczywisty pliku: {len(samples)/fs:.2f}s")
    print(f"Czas analizy: {time.time() - start_time:.2f}s")
    ser.close()

if __name__ == "__main__":
    play_wav_fast()