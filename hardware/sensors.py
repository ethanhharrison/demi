#!/usr/bin/env python3
"""
Poll TVOC (Total Volatile Organic Compounds) from the Sensirion SGP30
(GY-SGP30 breakout board) over Raspberry Pi I2C (GPIO2 SDA / GPIO3 SCL).

Reference:
- Sensirion SGP30 Datasheet (Document Number: 001168)
  https://sensirion.com/media/documents/3E3B3B27/6163E39E/Sensirion_Gas_Sensors_SGP30_Datasheet.pdf

This script:
  • Initializes the IAQ algorithm (required after every power-up)
  • Periodically issues MEASURE_AIR_QUALITY (0x2008)
  • Validates each 16-bit word with Sensirion’s CRC-8 checksum
  • Extracts TVOC (ppb) and prints to console (JSON or plain text)
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from typing import List, Tuple

from smbus2 import SMBus, i2c_msg

# === I2C parameters ===
SGP30_ADDR: int = 0x58  # Default 7-bit I2C address per datasheet

# === SGP30 command definitions (MSB, LSB) ===
# Datasheet §6.3 “Command Set”
CMD_INIT_AIR_QUALITY: Tuple[int, int] = (0x20, 0x03)   # IAQ initialization
CMD_MEASURE_AIR_QUALITY: Tuple[int, int] = (0x20, 0x08)  # Measure CO2eq & TVOC

# Global stop flag (for clean exit)
STOP: bool = False


# ========================
# Signal Handling
# ========================

def handle_sigint(signum: int, frame) -> None:
    """Signal handler to safely stop polling loop on Ctrl+C or SIGTERM."""
    global STOP
    STOP = True


signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)


# ========================
# Core Utility Functions
# ========================

def sensirion_crc(data_bytes: List[int]) -> int:
    """
    Compute Sensirion's CRC-8 checksum for a 2-byte word.
    (Datasheet §5.2.2 "Checksum")

    Polynomial: 0x31 (x^8 + x^5 + x^4 + 1)
    Initialization: 0xFF
    Args:
        data_bytes: Two data bytes [MSB, LSB].
    Returns:
        Computed 8-bit CRC value.
    """
    crc = 0xFF
    for b in data_bytes:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ 0x31
            else:
                crc = (crc << 1) & 0xFF
    return crc & 0xFF


def write_cmd(bus: SMBus, addr: int, cmd: Tuple[int, int]) -> None:
    """
    Send a 2-byte command to the SGP30 over I2C.

    Args:
        bus: Active SMBus instance.
        addr: I2C address of the SGP30 (default 0x58).
        cmd: Tuple containing two command bytes (MSB, LSB).
    """
    # The write_i2c_block_data method sends the two command bytes to the device.
    # No data payload follows for SGP30 commands.
    bus.write_i2c_block_data(addr, cmd[0], [cmd[1]])


def read_words_with_crc(bus: SMBus, addr: int, num_words: int) -> List[int]:
    """
    Read N words (each = 16 bits + CRC byte) from the sensor and validate each CRC.

    Per datasheet §5.2.1, each data word is transmitted as:
      [MSB] [LSB] [CRC]
    Args:
        bus: Active SMBus instance.
        addr: I2C address.
        num_words: Number of 16-bit words to read.
    Returns:
        List of validated 16-bit unsigned integers.
    Raises:
        ValueError: If CRC check fails for any word.
    """
    # Each word is 2 data bytes + 1 CRC = 3 bytes total.
    raw = bus.read_i2c_block_data(addr, 0x00, 3 * num_words)
    words: List[int] = []

    for i in range(num_words):
        msb = raw[3 * i]
        lsb = raw[3 * i + 1]
        crc = raw[3 * i + 2]

        # Verify CRC to ensure data integrity
        if sensirion_crc([msb, lsb]) != crc:
            raise ValueError(f"CRC mismatch on word {i}: got 0x{crc:02X}")

        words.append((msb << 8) | lsb)

    return words


# ========================
# SGP30 Command Wrappers
# ========================

def sgp30_init(bus: SMBus, addr: int) -> None:
    """
    Initialize the SGP30’s internal air quality algorithm.

    Must be called once after every power-up (datasheet §6.3.1).
    The sensor will start a “baseline learning” phase after this.

    Args:
        bus: Active SMBus instance.
        addr: I2C address.
    """
    write_cmd(bus, addr, CMD_INIT_AIR_QUALITY)
    # Datasheet specifies >10 ms delay for this command.
    time.sleep(0.01)


def sgp30_measure_air_quality(bus: SMBus, addr: int) -> Tuple[int, int]:
    """
    Request one air-quality measurement (datasheet §6.3.2).

    Returns:
      (CO2eq_ppm, TVOC_ppb)
      - CO2eq: Equivalent CO₂ concentration [ppm]
      - TVOC: Total Volatile Organic Compounds [ppb]

    Each MEASURE_AIR_QUALITY command triggers a new measurement internally.
    The sensor requires ~12 ms typical to respond with data.

    Args:
        bus: Active SMBus instance.
        addr: I2C address.
    Returns:
        Tuple of two integers (CO₂eq, TVOC).
    """
    write_cmd(bus, addr, CMD_MEASURE_AIR_QUALITY)
    time.sleep(0.015)  # Wait ~15 ms before reading (datasheet recommends 12 ms typical)
    eco2, tvoc = read_words_with_crc(bus, addr, 2)
    return eco2, tvoc


# ========================
# Main Execution Loop
# ========================

def main() -> None:
    """
    Main function for continuous TVOC polling from SGP30.

    Logic:
    1. Open I2C bus.
    2. Verify SGP30 presence (ACK check).
    3. Initialize IAQ algorithm (required once after boot).
    4. Enter loop: send MEASURE_AIR_QUALITY → read → print TVOC value.

    User can choose plain-text or JSON output format.
    """
    parser = argparse.ArgumentParser(
        description="Poll TVOC from SGP30 (GY-SGP30) on Raspberry Pi I2C."
    )
    parser.add_argument("--bus", type=int, default=1,
                        help="I2C bus number (default: 1)")
    parser.add_argument("--addr", type=lambda x: int(x, 0),
                        default=SGP30_ADDR,
                        help="I2C address in decimal or hex (default: 0x58)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Polling interval seconds (default: 1.0)")
    parser.add_argument("--plain", action="store_true",
                        help="Output plain text instead of JSON")
    args = parser.parse_args()

    use_json = not args.plain

    # Open the I2C bus. On Raspberry Pi, bus 1 uses GPIO2 (SDA) and GPIO3 (SCL).
    try:
        bus = SMBus(args.bus)
    except FileNotFoundError:
        print(f"Cannot open I2C bus {args.bus}. Enable I2C via raspi-config.",
              file=sys.stderr)
        sys.exit(2)

    try:
        # Quick probe to ensure device acknowledges (0-byte write)
        try:
            bus.i2c_rdwr(i2c_msg.write(args.addr, []))
        except OSError:
            print(f"No device found at 0x{args.addr:02X}. Check wiring and power.",
                  file=sys.stderr)
            sys.exit(3)

        # Initialize the IAQ algorithm
        sgp30_init(bus, args.addr)
        print("Polling TVOC (ppb). Press Ctrl+C to stop.")

        # Continuous measurement loop
        while not STOP:
            try:
                # Command measurement and parse results
                eco2_ppm, tvoc_ppb = sgp30_measure_air_quality(bus, args.addr)
                ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

                # Print as JSON or plain text
                if use_json:
                    print(json.dumps({
                        "time_utc": ts,
                        "sensor": "SGP30",
                        "tvoc_ppb": tvoc_ppb
                    }, separators=(",", ":"), ensure_ascii=False))
                else:
                    print(f"{ts} TVOC={tvoc_ppb} ppb")

            except (ValueError, OSError) as e:
                # CRC or I²C communication errors
                print(f"[SGP30] Read error: {e}", file=sys.stderr)

            # Maintain user-specified polling interval (default 1 Hz)
            time.sleep(max(0.01, args.interval))

    finally:
        try:
            bus.close()
        except Exception:
            pass


# ========================
# Entry Point
# ========================

if __name__ == "__main__":
    main()