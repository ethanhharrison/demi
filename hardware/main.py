#!/usr/bin/env python3
"""
Sampling from camera and SGP30 sensor with moving average filter.
Periodically appends JSON data (smelly + latest frame) to one file on the Mac over SSH.
"""

import threading
import time
import signal
import json
import base64
import cv2
import paramiko
from datetime import datetime
from typing import Optional, Any, List

import camera
import sensors
import filter as filter_module


class HardwareSampler:
    def __init__(self, camera_id: int = 0, bus_num: int = 1,
                 sensor_addr: int = sensors.SGP30_ADDR, sampling_hz: float = 10.0,
                 filter_points: int = 5, tvoc_threshold: float = 100.0,
                 send_interval: float = 5.0,
                 mac_ip: str = "",
                 mac_user: str = "",
                 remote_path: str = "") -> None:

        self.camera_id = camera_id
        self.bus_num = bus_num
        self.sensor_addr = sensor_addr
        self.sampling_hz = sampling_hz
        self.filter_points = filter_points
        self.tvoc_threshold = tvoc_threshold
        self.send_interval = send_interval
        self.mac_ip = mac_ip
        self.mac_user = mac_user
        self.remote_path = remote_path

        self.latest_frame: Optional[Any] = None
        self.tvoc_filtered: List[float] = []
        self.tvoc_raw: List[int] = []
        self.smelly: bool = False
        self.stop: bool = False

        self._camera_thread: Optional[threading.Thread] = None
        self._sensor_thread: Optional[threading.Thread] = None
        self._sender_thread: Optional[threading.Thread] = None

        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, sig: int, frame: Any) -> None:
        self.stop = True

    def _camera_loop(self) -> None:
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
                        self.smelly = filtered >= self.tvoc_threshold
                    if len(self.tvoc_raw) > self.filter_points:
                        self.tvoc_raw.pop(0)
                    if len(self.tvoc_filtered) > self.filter_points:
                        self.tvoc_filtered.pop(0)
                except Exception:
                    pass
                time.sleep(interval)
        finally:
            sensors.close_bus(bus)

    def _sender_loop(self) -> None:
        """Append JSON lines to one remote file over SSH."""
        while not self.stop:
            try:
                frame = self.latest_frame
                smelly = self.smelly
                timestamp = time.time()

                # Optionally encode latest frame
                if frame is not None:
                    _, jpeg = cv2.imencode(".jpg", frame)
                    frame_b64 = base64.b64encode(jpeg).decode("utf-8")
                else:
                    frame_b64 = None

                payload = {
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                    "smelly": smelly,
                    "frame_b64": frame_b64,
                }

                json_line = json.dumps(payload) + "\n"

                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.mac_ip, username=self.mac_user)
                sftp = ssh.open_sftp()

                # Append mode via remote shell command
                with sftp.file(self.remote_path, "a") as f:
                    f.write(json_line)

                sftp.close()
                ssh.close()

                print(f"Appended data at {payload['datetime']}")

            except Exception as e:
                print(f"Error sending data: {e}")

            time.sleep(self.send_interval)

    def start(self) -> None:
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._camera_thread.start()
        self._sensor_thread.start()
        self._sender_thread.start()

    def stop_sampling(self) -> None:
        self.stop = True
        if self._camera_thread:
            self._camera_thread.join(timeout=1)
        if self._sensor_thread:
            self._sensor_thread.join(timeout=1)
        if self._sender_thread:
            self._sender_thread.join(timeout=1)


if __name__ == "__main__":
    sampler = HardwareSampler(
        sampling_hz=10.0,
        filter_points=5,
        tvoc_threshold=100.0,
        send_interval=5.0,  # seconds
        mac_ip="10.0.27.83",
        mac_user="eharrison",
        remote_path="/Users/eharrison/data/sensor_log.json",
    )

    sampler.start()

    try:
        while not sampler.stop:
            time.sleep(0.1)
    except KeyboardInterrupt:
        sampler.stop = True

    sampler.stop_sampling()