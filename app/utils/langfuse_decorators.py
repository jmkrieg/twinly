"""
Langfuse observability decorators with graceful fallback.
"""
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, Optional, cast

from app.utils.langfuse_config import get_langfuse_availability

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def safe_observe(
    *,
    name: Optional[str] = None,
    as_type: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    **kwargs
) -> Callable[[F], F]:
    """
    Safe wrapper for Langfuse @observe decorator with graceful fallback.
    
    If Langfuse is not available, this decorator acts as a pass-through.
    
    Args:
        name: Optional name for the observation
        as_type: Optional type (e.g., "generation", "span")
        capture_input: Whether to capture function input
        capture_output: Whether to capture function output
        **kwargs: Additional keyword arguments for the observe decorator
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        if get_langfuse_availability():
            try:
                from langfuse.decorators import observe
                
                # Apply the actual Langfuse observe decorator
                observed_func = observe(
                    name=name,
                    as_type=as_type,
                    capture_input=capture_input,
                    capture_output=capture_output,
                    **kwargs
                )(func)
                
                # Mark function as observed for testing
                setattr(observed_func, '_langfuse_observed', True)
                return cast(F, observed_func)
                
            except ImportError:
                logger.warning("Langfuse decorators not available")
            except Exception as e:
                logger.warning(f"Failed to apply Langfuse observe decorator: {str(e)}")
        
        # Fallback: return original function with marker
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        setattr(wrapper, '_langfuse_observed', False)
        return cast(F, wrapper)
    
    return decorator


def safe_update_current_trace(**kwargs) -> None:
    """
    Safely update current trace with graceful fallback.
    
    Args:
        **kwargs: Trace update parameters (name, session_id, user_id, tags, etc.)
    """
    if get_langfuse_availability():
        try:
            from langfuse.decorators import langfuse_context
            langfuse_context.update_current_trace(**kwargs)
        except ImportError:
            logger.debug("Langfuse context not available for trace update")
        except Exception as e:
            logger.warning(f"Failed to update current trace: {str(e)}")


def safe_update_current_observation(**kwargs) -> None:
    """
    Safely update current observation with graceful fallback.
    
    Args:
        **kwargs: Observation update parameters (name, input, output, etc.)
    """
    if get_langfuse_availability():
        try:
            from langfuse.decorators import langfuse_context
            langfuse_context.update_current_observation(**kwargs)
        except ImportError:
            logger.debug("Langfuse context not available for observation update")
        except Exception as e:
            logger.warning(f"Failed to update current observation: {str(e)}")


def safe_score_current_trace(name: str, value: float, comment: Optional[str] = None) -> None:
    """
    Safely score current trace with graceful fallback.
    
    Args:
        name: Score name
        value: Score value
        comment: Optional comment
    """
    if get_langfuse_availability():
        try:
            from langfuse.decorators import langfuse_context
            langfuse_context.score_current_trace(
                name=name,
                value=value,
                comment=comment
            )
        except ImportError:
            logger.debug("Langfuse context not available for trace scoring")
        except Exception as e:
            logger.warning(f"Failed to score current trace: {str(e)}")


def safe_get_current_trace_id() -> Optional[str]:
    """
    Safely get current trace ID with graceful fallback.
    
    Returns:
        Trace ID if available, None otherwise
    """
    if get_langfuse_availability():
        try:
            from langfuse.decorators import langfuse_context
            return langfuse_context.get_current_trace_id()
        except ImportError:
            logger.debug("Langfuse context not available for trace ID")
        except Exception as e:
            logger.warning(f"Failed to get current trace ID: {str(e)}")
    
    return None


def safe_get_current_observation_id() -> Optional[str]:
    """
    Safely get current observation ID with graceful fallback.
    
    Returns:
        Observation ID if available, None otherwise
    """
    if get_langfuse_availability():
        try:
            from langfuse.decorators import langfuse_context
            return langfuse_context.get_current_observation_id()
        except ImportError:
            logger.debug("Langfuse context not available for observation ID")
        except Exception as e:
            logger.warning(f"Failed to get current observation ID: {str(e)}")
    
    return None
