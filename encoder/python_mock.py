import serial
import wave
import numpy as np
import time
import pandas as pd

# --- KONFIGURACJA ---
PORT = '/dev/ttyACM0'  # port arduino na linuksie
BAUD = 115200
DOG = "sounds/dog.wav"
DOOR = "sounds/door.wav"
GLASS1 = "sounds/glass1.wav"
SOUND_FILE = GLASS1
WINDOW_MS = 10         # Musi być zgodne z RC_WINDOW_MS w Arduino
OUTPUT_FILE = "output/data.txt"

def play_wav_to_arduino():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2) # Czekaj na restart Arduino
        
        with wave.open(SOUND_FILE, 'rb') as wav:
            fps = wav.getframerate()
            n_channels = wav.getnchannels()
            # Ile próbek audio przypada na jedno okno czasowe (np. 10ms)
            samples_per_window = int(fps * (WINDOW_MS / 1000.0))

            with open(OUTPUT_FILE, "w") as file:
            
                print(f"Start wysyłania: {SOUND_FILE}")
                print(f"Parametry: {fps}Hz, Okno: {WINDOW_MS}ms")

                while True:
                    data = wav.readframes(samples_per_window)
                    if not data: break
                    
                    # Konwersja bajtów na liczby
                    samples = np.frombuffer(data, dtype=np.int16)
                    if n_channels > 1: samples = samples[::n_channels] # Jeśli stereo, bierz jeden kanał
                    
                    if len(samples) > 0:
                        # Obliczamy amplitudę peak-to-peak (max - min)
                        amp = int(np.max(samples) - np.min(samples))
                        
                        # Skalujemy do zakresu 0-1023 (Arduino ADC)
                        # Zakładając że 16-bit audio ma zakres 65535, skalujemy:
                        scaled_amp = int((amp / 65535.0) * 1023)
                        
                        # Wysyłamy do Arduino
                        ser.write(f"{scaled_amp}\n".encode())
                        
                        # Czekamy tyle, ile trwa okno, żeby zachować czas rzeczywisty
                        time.sleep(WINDOW_MS / 1000.0)
                        
                        # Czytamy co Arduino odpowiedziało (opcjonalnie do debugu)
                        if ser.in_waiting:
                            response = ser.readline().decode().strip()
                            info = f"Wysłano: {scaled_amp} | Arduino: {response}"
                            print(info)
                            file.write(f"{info}\n")

                print("Koniec pliku.")

    except Exception as e:
        print(f"Błąd: {e}")
    finally:
        if 'ser' in locals(): ser.close()

if __name__ == "__main__":
    play_wav_to_arduino()