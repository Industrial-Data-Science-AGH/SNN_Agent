#!/usr/bin/env python3
import serial
import time
import sys
from rpi_agents.agent.serial_protocol import *


def main(port, baudrate=115200):
    ser = serial.Serial(port, baudrate, timeout=1.0)
    assembler = LineAssembler()

    def get_new_boot_id():
        return f"boot_{int(time.time())}"

    tracker = StreamTracker(get_new_boot_id)

    start_time = time.time()
    last_boot_request = 0.0
    frames_received = 0
    anomalies_detected = {}
    gaps_detected = 0
    restarts = 0

    print("Rozpoczęcie testu stabilności. Zbieranie danych...")

    try:
        while True:
            chunk = ser.read(1024)
            if not chunk:
                continue

            for line_res in assembler.feed(chunk):
                if isinstance(line_res, bytes):
                    events = tracker.feed_line(line_res)
                    for event in events:
                        if isinstance(event, FrameEvent):
                            frames_received += 1
                            for anomaly in event.anomalies:
                                anomalies_detected[anomaly] = anomalies_detected.get(anomaly, 0) + 1
                        elif isinstance(event, GapEvent):
                            gaps_detected += event.missing_hops
                        elif isinstance(event, BootEvent):
                            restarts += 1
                        elif isinstance(event, RejectedEvent):
                            if event.code == "NO_BOOT" and (time.time() - last_boot_request) > 1.0:
                                ser.write(b"I")
                                last_boot_request = time.time()
                            elif event.code != "NO_BOOT":
                                print(f"Odrzucono ramkę: {event.code} - {event.detail}")

            elapsed = time.time() - start_time
            if elapsed >= 3600:  # Zakończ po 60 minutach
                break

    except KeyboardInterrupt:
        print("\nTest przerwany przez użytkownika.")
    finally:
        print("\n=== Wyniki testu stabilności ===")
        print(f"Czas trwania: {int(elapsed / 60)} min {int(elapsed % 60)} s")
        print(f"Odebrane ramki: {frames_received}")
        print(f"Utracone pakiety (GapEvent hops): {gaps_detected}")
        print(f"Liczba restartów: {restarts}")
        print(f"Wykryte wahania opóźnień i błędy (Jitter/Anomalie): {anomalies_detected}")


if __name__ == "__main__":
    port_name = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    main(port_name)