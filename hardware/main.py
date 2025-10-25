#!/usr/bin/env python3
"""
Simple sampling from camera and SGP30 sensor with moving average filter.
"""

import threading
import time
import signal
from typing import Optional, Any, List

import camera
import sensors
import filter as filter_module

latest_frame: Optional[Any] = None
tvoc_filtered: List[float] = []
tvoc_raw: List[int] = []
stop: bool = False


def handle_exit(sig: int, frame: Any) -> None:
    """Handle shutdown signal."""
    global stop
    stop = True


signal.signal(signal.SIGINT, handle_exit)


def camera_thread(camera_id: int = 0, sampling_hz: float = 10.0) -> None:
    """
    Sample camera at specified rate and store latest frame.
    
    Args:
        camera_id: Camera device ID.
        sampling_hz: Sampling frequency in Hz.
    """
    global latest_frame, stop
    
    cap = camera.initialize_camera(camera_id)
    if not cap:
        return
    
    interval = 1.0 / sampling_hz
    
    try:
        while not stop:
            result = camera.capture_frame(cap)
            if result:
                ret, frame = result
                if ret:
                    latest_frame = frame
            time.sleep(interval)
    finally:
        camera.release_camera(cap)


def sensor_thread(bus_num: int = 1, addr: int = sensors.SGP30_ADDR, 
                  sampling_hz: float = 10.0, filter_points: int = 5) -> None:
    """
    Sample TVOC at specified rate with N-point moving average.
    
    Args:
        bus_num: I2C bus number.
        addr: I2C address of SGP30 sensor.
        sampling_hz: Sampling frequency in Hz.
        filter_points: Number of points for moving average filter.
    """
    global tvoc_filtered, tvoc_raw, stop
    
    bus = sensors.open_i2c_bus(bus_num)
    if not bus or not sensors.verify_device_presence(bus, addr):
        sensors.close_bus(bus)
        return
    
    sensors.sgp30_init(bus, addr)
    moving_avg = filter_module.MovingAverage(N=filter_points)
    
    interval = 1.0 / sampling_hz
    
    try:
        while not stop:
            try:
                eco2, tvoc = sensors.sgp30_measure_air_quality(bus, addr)
                filtered = moving_avg.update(float(tvoc))
                
                tvoc_raw.append(tvoc)
                if filtered is not None:
                    tvoc_filtered.append(filtered)
                
                if len(tvoc_raw) > 1000:
                    tvoc_raw.pop(0)
                if len(tvoc_filtered) > 1000:
                    tvoc_filtered.pop(0)
                    
            except Exception:
                pass
            
            time.sleep(interval)
    finally:
        sensors.close_bus(bus)


if __name__ == "__main__":
    SAMPLING_HZ = 10.0
    FILTER_POINTS = 5
    
    cam = threading.Thread(target=camera_thread, args=(0, SAMPLING_HZ), daemon=True)
    sen = threading.Thread(target=sensor_thread, args=(1, sensors.SGP30_ADDR, SAMPLING_HZ, FILTER_POINTS), daemon=True)
    
    cam.start()
    sen.start()
    
    try:
        while not stop:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop = True
    
    cam.join(timeout=1)
    sen.join(timeout=1)