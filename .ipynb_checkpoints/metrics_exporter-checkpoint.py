from prometheus_client import start_http_server, Gauge
import psutil
import random
import time

# Métricas
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage in percent')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage in percent')
model_accuracy = Gauge('model_accuracy', 'Model accuracy in percent')

def collect_metrics():
    while True:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        acc = random.uniform(80, 95)  # Simula precisión de un modelo

        cpu_usage.set(cpu)
        memory_usage.set(mem)
        model_accuracy.set(acc)

        time.sleep(5)

if __name__ == "__main__":
    start_http_server(8000)  # Exposición en localhost:8000/metrics
    collect_metrics()
