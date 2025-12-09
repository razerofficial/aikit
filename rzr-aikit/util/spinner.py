def spinner(message: str = "Processing...", console=None):
    """
    A utility decorator to show a spinner during execution.

    Args:
        message (str): The message to display next to the spinner.
        speed (float): Speed of the spinner animation.
    """
    from rich.console import Console
    from typing import Callable, Any
    import functools

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            new_console = console or Console()
            with new_console.status(message):
                result = func(*args, **kwargs)
                return result

        return wrapper

    return decorator
