"""
----------------------------------------------------------------
# Twinly API - FastAPI Application Entry Point
----------------------------------------------------------------
# 
# A simple FastAPI application that serves as a container for 
# the development of LLM Powered Chatbots and agents.
#
# The application will be consumed by a Open WebUI Frontend.
#
----------------------------------------------------------------
"""





"""
----------------------------------------------------------------
# MODULES AND IMPORTS
----------------------------------------------------------------
# 
# In the first step we import the necessary modules.
#  
#  - os is used for environment variable management
#  - fastapi is the web framework for building the API
#  - dotenv is used to load environment variables from a .env 
#    file
#  - chat_router is the router for chat-related endpoints
#  - setup_logging and get_logger are utility functions for 
#    logging setup
#  - configure_langfuse and get_langfuse_availability are 
#    utility functions for Langfuse observability configuration
# 
----------------------------------------------------------------
"""

import os
from fastapi import FastAPI
# from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from app.api.chat import router as chat_router
from app.utils.logging import setup_logging, get_logger
from app.utils.langfuse_config import configure_langfuse, get_langfuse_availability





"""
----------------------------------------------------------------
# APPLICATION INITIALIZATION
----------------------------------------------------------------
# 
# In this section, we initialize the FastAPI application, 
# configure logging, and set up Langfuse observability.
# 
----------------------------------------------------------------
"""


"""
----------------------------------------------------------------
# STEP 1: Load Environment Variables
----------------------------------------------------------------
# 
# This step loads environment variables from a .env file to 
# configure the application.
# 
----------------------------------------------------------------
"""

load_dotenv()



"""
----------------------------------------------------------------
# STEP 2: Configure Logging
----------------------------------------------------------------
# 
# This step sets up logging for the application, allowing for 
# different log levels based on environment variables.
# 
----------------------------------------------------------------
"""

setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)


"""
----------------------------------------------------------------
# STEP 3: Configure Langfuse Observability
----------------------------------------------------------------
# 
# This step configures Langfuse observability if the package is 
# available and the necessary environment variables are set.
# 
----------------------------------------------------------------
"""

langfuse_available = configure_langfuse()
if langfuse_available:
    logger.info("Langfuse observability is enabled")
else:
    logger.info("Langfuse observability is disabled (not configured or package not available)")


"""
----------------------------------------------------------------
# STEP 4: Initialize FastAPI Application
----------------------------------------------------------------
# 
# This step initializes the FastAPI application with metadata 
# such as title, description, and version.
# 
----------------------------------------------------------------
"""

app = FastAPI(
    title="Twinly API",
    description="A simple FastAPI application for LLM Powered Chatbots and agents",
    version="1.0.0"
)


"""
----------------------------------------------------------------
# STEP 5: Register Routers
----------------------------------------------------------------
# 
# This step registers the chat router for handling chat-related
# endpoints.
# 
----------------------------------------------------------------
"""

app.include_router(chat_router)


"""
----------------------------------------------------------------
# STEP 6: Application Startup Event
----------------------------------------------------------------
# 
# This step defines an application startup event to log that the
# API has been initialized successfully.
# 
----------------------------------------------------------------
"""

logger.info("Twinly API initialized successfully")




"""
----------------------------------------------------------------
# MAIN APPLICATION ENTRY POINT
----------------------------------------------------------------
# 
# This section defines the main entry point for the FastAPI 
# application, allowing it to be run with Uvicorn.
# 
----------------------------------------------------------------
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
