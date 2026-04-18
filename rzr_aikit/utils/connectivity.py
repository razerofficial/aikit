"""Lightweight utility for checking network connectivity to external services."""

import socket


def check_huggingface_connectivity(timeout: float = 2.0) -> bool:
    """
    Check if Hugging Face Hub is reachable using a lightweight DNS lookup.
    
    This function performs a fast DNS resolution check to verify that Hugging Face
    is accessible. It's much faster than making HTTP requests and adds minimal
    overhead to execution time.
    
    Args:
        timeout: Maximum time in seconds to wait for DNS resolution. Default is 2.0.
    
    Returns:
        True if Hugging Face Hub DNS can be resolved, False otherwise.
    
    Examples:
        >>> if check_huggingface_connectivity():
        ...     # Fetch model info from HuggingFace
        ...     model_info = model_info(model_name)
        ... else:
        ...     # Use offline mode
        ...     print("Working in offline mode")
    """
    try:
        # Fast DNS lookup - much lighter than HTTP request
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname("huggingface.co")
        return True
    except (socket.gaierror, socket.timeout, OSError):
        return False
