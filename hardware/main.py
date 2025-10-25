#!/usr/bin/env python3
"""
Simple sampling from camera and SGP30 sensor with moving average filter.
"""

import threading
import time
import signal
from typing import Optional, Any, List, Tuple

import camera
import sensors
import filter as filter_module


class HardwareSampler:
    """
    Manages concurrent sampling from camera and SGP30 TVOC sensor.
    """
    
    def __init__(self, camera_id: int = 0, bus_num: int = 1, 
                 sensor_addr: int = sensors.SGP30_ADDR, sampling_hz: float = 10.0,
                 filter_points: int = 5, tvoc_threshold: float = 100.0) -> None:
        """
        Initialize hardware sampler.
        
        Args:
            camera_id: Camera device ID.
            bus_num: I2C bus number.
            sensor_addr: I2C address of SGP30 sensor.
            sampling_hz: Sampling frequency in Hz.
            filter_points: Number of points for moving average filter.
            tvoc_threshold: TVOC threshold in ppb for smelly detection.
        """
        self.camera_id = camera_id
        self.bus_num = bus_num
        self.sensor_addr = sensor_addr
        self.sampling_hz = sampling_hz
        self.filter_points = filter_points
        self.tvoc_threshold = tvoc_threshold
        
        self.latest_frame: Optional[Any] = None
        self.tvoc_filtered: List[float] = []
        self.tvoc_raw: List[int] = []
        self.smelly: bool = False
        self.stop: bool = False
        
        self._camera_thread: Optional[threading.Thread] = None
        self._sensor_thread: Optional[threading.Thread] = None
        
        signal.signal(signal.SIGINT, self._handle_exit)
    
    def _handle_exit(self, sig: int, frame: Any) -> None:
        """Handle shutdown signal."""
        self.stop = True
    
    def _camera_loop(self) -> None:
        """Sample camera at specified rate and store latest frame."""
        cap = camera.initialize_camera(self.camera_id)
        if not cap:
            return
        
        interval = 1.0 / self.sampling_hz
        
        try:
            while not self.stop:
                result = camera.capture_frame(cap)
                if result:
                    ret, frame = result
                    if ret:
                        self.latest_frame = frame
                time.sleep(interval)
        finally:
            camera.release_camera(cap)
    
    def _sensor_loop(self) -> None:
        """Sample TVOC at specified rate with N-point moving average."""
        bus = sensors.open_i2c_bus(self.bus_num)
        if not bus or not sensors.verify_device_presence(bus, self.sensor_addr):
            sensors.close_bus(bus)
            return
        
        sensors.sgp30_init(bus, self.sensor_addr)
        moving_avg = filter_module.MovingAverage(N=self.filter_points)
        
        interval = 1.0 / self.sampling_hz
        
        try:
            while not self.stop:
                try:
                    eco2, tvoc = sensors.sgp30_measure_air_quality(bus, self.sensor_addr)
                    filtered = moving_avg.update(float(tvoc))
                    
                    self.tvoc_raw.append(tvoc)
                    if filtered is not None:
                        self.tvoc_filtered.append(filtered)
                        
                        if filtered >= self.tvoc_threshold:
                            self.smelly = True
                        else:
                            self.smelly = False
                    
                    if len(self.tvoc_raw) > self.filter_points:
                        self.tvoc_raw.pop(0)
                    if len(self.tvoc_filtered) > self.filter_points:
                        self.tvoc_filtered.pop(0)
                        
                except Exception:
                    pass
                
                time.sleep(interval)
        finally:
            sensors.close_bus(bus)
    
    def start(self) -> None:
        """Start camera and sensor sampling threads."""
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        
        self._camera_thread.start()
        self._sensor_thread.start()
    
    def stop_sampling(self) -> None:
        """Stop all sampling threads."""
        self.stop = True
        
        if self._camera_thread:
            self._camera_thread.join(timeout=1)
        if self._sensor_thread:
            self._sensor_thread.join(timeout=1)
    
    def get_latest_frame(self) -> Optional[Any]:
        """
        Get the latest camera frame.
        
        Returns:
            Latest frame or None if no frame available.
        """
        return self.latest_frame
    
    def get_smelly(self) -> bool:
        """
        Get current smelly status.
        
        Returns:
            True if TVOC is above threshold, False otherwise.
        """
        return self.smelly


if __name__ == "__main__":
    SAMPLING_HZ = 10.0
    FILTER_POINTS = 5
    TVOC_THRESHOLD = 100.0
    
    sampler = HardwareSampler(
        camera_id=0,
        bus_num=1,
        sensor_addr=sensors.SGP30_ADDR,
        sampling_hz=SAMPLING_HZ,
        filter_points=FILTER_POINTS,
        tvoc_threshold=TVOC_THRESHOLD
    )
    
    sampler.start()
    
    try:
        while not sampler.stop:
            time.sleep(0.1)
    except KeyboardInterrupt:
        sampler.stop = True
    
    sampler.stop_sampling()