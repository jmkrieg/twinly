"""
Langfuse configuration utility.
"""
import os
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def get_langfuse_config() -> Optional[Dict[str, Any]]:
    """
    Get Langfuse configuration from environment variables.
    
    Returns:
        Dictionary with Langfuse configuration or None if not configured
    """
    public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
    secret_key = os.getenv('LANGFUSE_SECRET_KEY')
    host = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
    
    if not public_key or not secret_key:
        logger.info("Langfuse not configured - missing public or secret key")
        return None
    
    config = {
        'enabled': True,
        'public_key': public_key,
        'secret_key': secret_key,
        'host': host
    }
    
    logger.info(f"Langfuse configured with host: {host}")
    return config


def is_langfuse_enabled() -> bool:
    """
    Check if Langfuse is enabled and properly configured.
    
    Returns:
        True if Langfuse is enabled, False otherwise
    """
    config = get_langfuse_config()
    return config is not None and config.get('enabled', False)


def configure_langfuse() -> bool:
    """
    Configure Langfuse if available.
    
    Returns:
        True if configuration was successful, False otherwise
    """
    try:
        if not is_langfuse_enabled():
            logger.info("Langfuse observability is disabled")
            return False
            
        # Try to import and configure Langfuse
        from langfuse import Langfuse
        
        config = get_langfuse_config()
        
        # Initialize Langfuse client
        langfuse = Langfuse(
            public_key=config['public_key'],
            secret_key=config['secret_key'],
            host=config['host']
        )
        
        # Test connection
        langfuse.auth_check()
        logger.info("Langfuse observability initialized successfully")
        return True
        
    except ImportError:
        logger.warning("Langfuse package not installed")
        return False
    except Exception as e:
        logger.warning(f"Failed to configure Langfuse: {str(e)}")
        return False


# Initialize configuration on import
_langfuse_available = configure_langfuse()


def get_langfuse_availability() -> bool:
    """
    Get the current Langfuse availability status.
    
    Returns:
        True if Langfuse is available and configured, False otherwise
    """
    return _langfuse_available
