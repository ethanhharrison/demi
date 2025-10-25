#!/usr/bin/env python3
"""
Poll TVOC (Total Volatile Organic Compounds) from the Sensirion SGP30
(GY-SGP30 breakout board) over Raspberry Pi I2C (GPIO2 SDA / GPIO3 SCL).

Reference:
- Sensirion SGP30 Datasheet (Document Number: 001168)
  https://sensirion.com/media/documents/3E3B3B27/6163E39E/Sensirion_Gas_Sensors_SGP30_Datasheet.pdf

This module provides functions to:
  • Initialize the IAQ algorithm (required after every power-up)
  • Measure air quality (CO2eq & TVOC) via MEASURE_AIR_QUALITY (0x2008)
  • Validate data with Sensirion's CRC-8 checksum
"""

import json
import sys
import time
import random
from datetime import datetime
from typing import List, Optional, Tuple

from smbus2 import SMBus, i2c_msg

# === I2C parameters ===
SGP30_ADDR: int = 0x58  # Default 7-bit I2C address per datasheet

# === SGP30 command definitions (MSB, LSB) ===
# Datasheet §6.3 "Command Set"
CMD_INIT_AIR_QUALITY: Tuple[int, int] = (0x20, 0x03)   # IAQ initialization
CMD_MEASURE_AIR_QUALITY: Tuple[int, int] = (0x20, 0x08)  # Measure CO2eq & TVOC


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

def sgp30_measure_air_quality_pseudo(bus: SMBus, addr: int) -> Tuple[int, int]:
    """
    Generate a pseudorandom, artificial air-quality measurement
    that mimics real SGP30 output for testing or simulation.

    Args:
        bus: SMBus instance (ignored, kept for API compatibility).
        addr: I2C address (ignored, kept for API compatibility).

    Returns:
        (CO2eq_ppm, TVOC_ppb)
    """
    # Generate realistic ranges based on SGP30 behavior:
    # CO2eq: ~400–2000 ppm typical indoor range
    # TVOC: 0–600 ppb normal, 600–1200 elevated, >1200 high
    eco2_pseudo = random.randint(400, 2000)
    tvoc_pseudo = int(
        max(0, random.gauss(150, 100))
    )

    return eco2_pseudo, tvoc_pseudo

def open_i2c_bus(bus_number: int = 1) -> Optional[SMBus]:
    """
    Open an I2C bus connection.
    
    Args:
        bus_number: I2C bus number (default: 1 for Raspberry Pi).
    
    Returns:
        SMBus instance if successful, None if the bus cannot be opened.
    """
    try:
        return SMBus(bus_number)
    except FileNotFoundError:
        return None


def verify_device_presence(bus: SMBus, addr: int) -> bool:
    """
    Verify that a device is present at the specified I2C address.
    
    Args:
        bus: Active SMBus instance.
        addr: I2C address to check.
    
    Returns:
        True if device acknowledges, False otherwise.
    """
    try:
        bus.i2c_rdwr(i2c_msg.write(addr, []))
        return True
    except OSError:
        return False


def format_measurement_json(tvoc_ppb: int, sensor_name: str = "SGP30") -> str:
    """
    Format a TVOC measurement as a JSON string.
    
    Args:
        tvoc_ppb: TVOC value in parts per billion.
        sensor_name: Name of the sensor (default: "SGP30").
    
    Returns:
        JSON string with timestamp, sensor name, and TVOC value.
    """
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return json.dumps({
        "time_utc": ts,
        "sensor": sensor_name,
        "tvoc_ppb": tvoc_ppb
    }, separators=(",", ":"), ensure_ascii=False)


def format_measurement_plain(tvoc_ppb: int) -> str:
    """
    Format a TVOC measurement as plain text.
    
    Args:
        tvoc_ppb: TVOC value in parts per billion.
    
    Returns:
        Plain text string with timestamp and TVOC value.
    """
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return f"{ts} TVOC={tvoc_ppb} ppb"


def close_bus(bus: Optional[SMBus]) -> None:
    """
    Safely close an I2C bus connection.
    
    Args:
        bus: SMBus instance to close (can be None).
    """
    if bus is not None:
        try:
            bus.close()
        except Exception:
            pass