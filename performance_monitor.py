try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

import time
import logging
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to track performance
_start_time = None
if PSUTIL_AVAILABLE:
    _process = psutil.Process() if psutil else None
else:
    _process = None

def start_monitoring():
    """
    Start performance monitoring.
    """
    global _start_time
    if not PSUTIL_AVAILABLE:
        return
    _start_time = time.time()

def end_monitoring(operation_name="Operation"):
    """
    End performance monitoring and log results.
    """
    global _start_time
    if not PSUTIL_AVAILABLE or _start_time is None:
        return None, None, None
        
    if _start_time is not None:
        elapsed_time = time.time() - _start_time
        memory_info = _process.memory_info() if _process else None
        cpu_percent = _process.cpu_percent() if _process else 0
        
        if memory_info:
            logger.info(f"{operation_name} took {elapsed_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            logger.info(f"CPU usage: {cpu_percent}%")
        
        _start_time = None
        if memory_info:
            return elapsed_time, memory_info.rss / 1024 / 1024, cpu_percent
    return None, None, None

def get_system_resources():
    """
    Get current system resource usage.
    """
    if not PSUTIL_AVAILABLE:
        return {}
        
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        return {}

def monitor_resources():
    """
    Display system resource usage in the Streamlit app.
    """
    if not PSUTIL_AVAILABLE:
        return
        
    try:
        resources = get_system_resources()
        if resources:
            # Create metrics in the sidebar
            with st.sidebar:
                st.markdown("---")
                st.subheader("🖥️ System Resources")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CPU Usage", f"{resources['cpu_percent']:.1f}%")
                    st.metric("Memory Used", f"{resources['memory_percent']:.1f}%")
                
                with col2:
                    st.metric("Available Memory", f"{resources['memory_available_gb']:.1f} GB")
                    st.metric("Disk Usage", f"{resources['disk_percent']:.1f}%")
    except Exception as e:
        logger.error(f"Error monitoring resources: {e}")

def format_bytes(bytes_value):
    """
    Format bytes to human readable format.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"