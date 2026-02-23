
import spidev
import time
import threading
import queue
import numpy as np
import struct
import hashlib
import subprocess
import sys
from collections import deque
from datetime import datetime
import logging

# logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/tmp/decoder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# config
SPI_BUS = 0
SPI_DEVICE = 0
SPI_SPEED = 4_000_000  # 4 MHz (z Arduino)

# Bufory
AUDIO_BUFFER_SIZE = 16000  # 2 sekund audio @ 8kHz
circular_buffer = deque(maxlen=AUDIO_BUFFER_SIZE)

# Event detection
SPIKE_QUEUE = queue.Queue(maxsize=100)
ANOMALY_THRESHOLD = 150
SPIKE_COOLDOWN_MS = 100  # Minimum ms między spkami przed alertem

# LLM Agent config
LLM_MODEL = "tinyllama"  
OLLAMA_HOST = "http://localhost:11434"

class SpiPacket:
    """Parser pakietu AER spike'a z Arduino"""
    HEADER = 0xAE
    SIZE = 6  # bajtów
    
    def __init__(self, raw_bytes):
        if len(raw_bytes) < self.SIZE:
            raise ValueError(f"Packet too short: {len(raw_bytes)} < {self.SIZE}")
        
        self.header = raw_bytes[0]
        self.timestamp = (raw_bytes[1] << 8) | raw_bytes[2]
        self.energy_level = raw_bytes[3]
        self.spike_id = raw_bytes[4]
        self.checksum = raw_bytes[5]
        self.received_time = time.time()
    
    def validate(self):
        if self.header != self.HEADER:
            return False
        
        expected_checksum = self.header ^ self.timestamp ^ self.energy_level ^ self.spike_id
        if self.checksum != expected_checksum:
            logger.warning(f"Checksum mismatch: got {hex(self.checksum)}, expected {hex(expected_checksum)}")
            return False
        
        return True
    
    def __repr__(self):
        return f"Spike(T={self.timestamp:05d}, E={self.energy_level:3d}, ID={self.spike_id})"

class SpiReader:
    def __init__(self, bus=0, device=0, speed=4_000_000):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed
        self.spi.mode = 0  # CPOL=0, CPHA=0
        self.spi.lsb_first = False
        
        self.running = False
        self.thread = None
        self.packet_count = 0
        self.error_count = 0
        
        logger.info(f"SPI Reader initialized: Bus={bus}, Device={device}, Speed={speed/1e6:.1f}MHz")
    
    def start(self):
        """Start SPI reading thread"""
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info("SPI Reader thread started")
    
    def stop(self):
        """Stop SPI reading"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.spi.close()
        logger.info(f"SPI Reader stopped - {self.packet_count} packets received, {self.error_count} errors")
    
    def _read_loop(self):
        """Main SPI reading loop (runs in thread)"""
        rx_buffer = bytearray(SpiPacket.SIZE)
        last_spike_time = 0
        
        while self.running:
            try:
                rx_data = self.spi.readbytes(SpiPacket.SIZE)
                
                try:
                    pkt = SpiPacket(rx_data)
                    
                    if not pkt.validate():
                        self.error_count += 1
                        logger.warning(f"Invalid packet: {pkt}")
                        continue
                    
                    self.packet_count += 1
                    current_time = time.time() * 1000  # ms
                    if current_time - last_spike_time < SPIKE_COOLDOWN_MS:
                        continue
                    
                    last_spike_time = current_time
                    
                    try:
                        SPIKE_QUEUE.put_nowait((pkt, datetime.now()))
                        logger.info(f"[SPIKE] {pkt} | Energy rise detected!")
                        subprocess.run(
                            ["bash", "-c", "echo 1 > /sys/class/gpio/gpio17/value"],
                            capture_output=True
                        )
                        time.sleep(0.05)
                        subprocess.run(
                            ["bash", "-c", "echo 0 > /sys/class/gpio/gpio17/value"],
                            capture_output=True
                        )
                    except queue.Full:
                        logger.warning("Spike queue full - dropping packet")
                
                except Exception as e:
                    logger.error(f"Error parsing packet: {e}")
                    self.error_count += 1
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"SPI read error: {e}")
                self.error_count += 1
                time.sleep(0.01)

class AnomalyDetector:
    """Filtruje fałszywe alarmy, agreguje spiki w zdarzenia"""
    
    def __init__(self, spike_queue, buffer):
        self.spike_queue = spike_queue
        self.buffer = buffer
        self.spike_history = deque(maxlen=10)  # Last 10 spikes
        self.event_threshold = 3  # 3 spikes w 500ms = event
        self.last_event_time = 0
    
    def process(self):
        """Process incoming spikes, return True if significant event detected"""
        try:
            pkt, timestamp = self.spike_queue.get(timeout=0.1)
            
            self.spike_history.append({
                'energy': pkt.energy_level,
                'time': timestamp,
                'id': pkt.spike_id
            })
            
            self.buffer.append(pkt.energy_level)
            
            # multiple spikes close together = anomaly
            if len(self.spike_history) >= self.event_threshold:
                time_window = (self.spike_history[-1]['time'] - self.spike_history[0]['time']).total_seconds()
                
                if time_window < 0.5:  # Within 500ms
                    total_energy = sum(s['energy'] for s in self.spike_history)
                    avg_energy = total_energy / len(self.spike_history)
                    
                    logger.info(
                        f"[EVENT] Spike cluster detected: "
                        f"{len(self.spike_history)} spikes in {time_window:.2f}s, "
                        f"avg energy: {avg_energy:.0f}"
                    )
                    
                    self.spike_history.clear()
                    return True
            
            return False
        
        except queue.Empty:
            return False
        except Exception as e:
            logger.error(f"Error in AnomalyDetector: {e}")
            return False


class LLMAgent:
    """Interfejs do lokalnego LLM (Ollama)"""
    
    def __init__(self, model=LLM_MODEL, host=OLLAMA_HOST):
        self.model = model
        self.host = host
        self.is_running = False
        
        try:
            import requests
            self.requests = requests
            logger.info(f"LLM Agent initialized: model={model}")
        except ImportError:
            logger.warning("requests library not found - installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
            import requests
            self.requests = requests
    
    def wake_up(self, spike_data, context_buffer):
        """Wybudź agenta z kontekstem"""
        if self.is_running:
            logger.warning("LLM already running - skipping")
            return
        
        self.is_running = True
        logger.info("[LLM] WAKING UP AGENT")
        
        # Przygotuj prompt z kontekstem
        context_str = f"Audio energy spike: {spike_data} | Last 100 samples: {list(context_buffer)[-100:]}"
        
        prompt = f"""
You are an anomaly detector. The home sensor just detected an acoustic event:

{context_str}

Analyze this and respond in ONE LINE with action:
- ALARM: If danger detected (glass breaking, siren, etc)
- WARNING: If suspicious
- NORMAL: If false alarm

Action: """
        
        try:
            response = self.requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                action = result.get('response', '').strip().split('\n')[0]
                logger.info(f"[LLM RESPONSE] {action}")
                
                if 'ALARM' in action:
                    logger.critical("🚨 ALARM TRIGGERED!")
                    self._trigger_alarm()
                elif 'WARNING' in action:
                    logger.warning("⚠️ WARNING")
                else:
                    logger.info("✓ False alarm dismissed")
            else:
                logger.error(f"Ollama error: {response.status_code}")
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
        
        finally:
            self.is_running = False
    
    def _trigger_alarm(self):
        """Trigger physical alarm (siren, email, notification, etc)"""
        logger.critical("ALARM ACTION: Email / Push notification / Siren triggered")

class NeuromorphicDecoder:
    """Main decoder system"""
    
    def __init__(self):
        self.spi_reader = SpiReader(SPI_BUS, SPI_DEVICE, SPI_SPEED)
        self.anomaly_detector = AnomalyDetector(SPIKE_QUEUE, circular_buffer)
        self.llm_agent = LLMAgent()
        self.running = False
    
    def start(self):
        """Start the full system"""
        logger.info("=" * 60)
        logger.info("NEUROMORPHIC DECODER - STARTING")
        logger.info("=" * 60)
        
        self.spi_reader.start()
        self.running = True
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.stop()
    
    def _main_loop(self):
        """Main processing loop"""
        while self.running:
            # Process incoming spikes
            event_detected = self.anomaly_detector.process()
            
            if event_detected:
                # Wybudź LLM agenta z kontekstem
                recent_energy = list(circular_buffer)[-256:] if len(circular_buffer) > 0 else [0]
                avg_energy = np.mean(recent_energy) if recent_energy else 0
                
                self.llm_agent.wake_up(
                    spike_data=avg_energy,
                    context_buffer=recent_energy
                )
            
            time.sleep(0.01)  # 10ms loop
    
    def stop(self):
        """Stop the system"""
        self.running = False
        self.spi_reader.stop()
        logger.info("System stopped")

# entry
if __name__ == "__main__":
    try:
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(17, GPIO.OUT)  # LED na GPIO 17
            logger.info("GPIO initialized")
        except:
            logger.warning("GPIO library not available - visual LED disabled")
        
        decoder = NeuromorphicDecoder()
        decoder.start()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)