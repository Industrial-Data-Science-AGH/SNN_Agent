import serial
import wave
import numpy as np
import time

# --- KONFIGURACJA ---
# PORT = '/dev/ttyACM0'  # linux
PORT = 'COM3' # windows
BAUD = 115200
WINDOW_MS = 10         # Musi być zgodne z RC_WINDOW_MS w Arduino

# musisz być w /encoder
DOG = "sounds/dog.wav"
DOOR = "sounds/door.wav"
SILENCE = "sounds/silence.wav"
GLASS1 = "sounds/glass1.wav"
SOUND_FILE = SILENCE # ten plik jest grany

def output_file():
    if SOUND_FILE == DOG:
        return "spike_output/dog.txt"
    if SOUND_FILE == DOOR:
        return "spike_output/door.txt"
    if SOUND_FILE == GLASS1:
        return "spike_output/glass.txt"
    if SOUND_FILE == SILENCE:
        return "spike_output/silence.txt"
    return "spike_output/data.txt"

OUTPUT_FILE = output_file()

def get_spike_cnt(s):
    return int(s.split("\t")[-1].strip())

def play_wav_to_arduino():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(1) # Czekaj na restart Arduino
        
        with wave.open(SOUND_FILE, 'rb') as wav:
            fps = wav.getframerate()
            n_channels = wav.getnchannels()
            # Ile próbek audio przypada na jedno okno czasowe (np. 10ms)
            samples_per_window = int(fps * (WINDOW_MS / 1000.0))

            with open(OUTPUT_FILE, "w") as file:
            
                print(f"Start wysyłania: {SOUND_FILE}")
                print(f"Parametry: {fps}Hz, Okno: {WINDOW_MS}ms")

                is_ready = False
                spike_cnt = 0

                while True:
                    line = ser.readline().decode(errors="ignore").strip()
                    if line == "READY": 
                        is_ready = True
                    if not is_ready: 
                        continue
                    
                    data = wav.readframes(samples_per_window)
                    if not data: 
                        break
                    
                    # Konwersja bajtów na liczby
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.int32)
                    if n_channels > 1: 
                        samples = samples[::n_channels] # Jeśli stereo, bierz jeden kanał
                    
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
                        
                        # Czytamy co Arduino odpowiedziało
                        if ser.in_waiting:
                            response = ser.readline().decode("utf-8", errors="replace").strip()
                            info = f"Wyslano: {scaled_amp} | Arduino: {response}"
                            new_spike_cnt = get_spike_cnt(response)

                            if(spike_cnt != new_spike_cnt):
                                print(info.replace("\t", " | "))
                                file.write(f"{info.replace("\t", " | ")}\n")
                            spike_cnt = new_spike_cnt
                            
                spike_cnt = response.split("\t")[-1].strip()
                end_msg = f"\nDzwiek {SOUND_FILE} wywolal {spike_cnt} spike'ow"
                file.write(end_msg)
                print(end_msg)

    except Exception as e:
        print(f"Błąd: {e}")
    finally:
        if 'ser' in locals(): ser.close()

if __name__ == "__main__":
    play_wav_to_arduino()