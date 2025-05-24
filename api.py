from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr, validator
from llamafirewall import LlamaFirewall, UserMessage, Role, ScannerType
from typing import Optional, Dict, List
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import os
import json
import logging
import logging.handlers
from datetime import datetime

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = f"llmfirewall_api_{datetime.now().strftime('%Y%m%d')}.log"

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure root logger
logger = logging.getLogger("llmfirewall_api")
logger.setLevel(getattr(logging, LOG_LEVEL))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(console_handler)

# File handler with rotation
file_handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join("logs", LOG_FILE),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(file_handler)

# Get thread pool configuration from environment
THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "4"))  # Default to 4 workers if not specified
logger.info(f"Initializing thread pool with {THREAD_POOL_WORKERS} workers")

# Create thread pool executor at startup
thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)

app = FastAPI(
    title="LlamaFirewall API",
    description="API for scanning user messages using LlamaFirewall and OpenAI moderation",
    version="1.0.0"
)

def parse_scanners_config() -> Dict[Role, List[ScannerType]]:
    """
    Parse scanner configuration from environment variable.
    Default configuration if not set:
    {
        "USER": ["PROMPT_GUARD"]
    }
    """
    global moderation
    moderation = False  # Reset moderation flag
    default_config = {
        Role.USER: [ScannerType.PROMPT_GUARD]
    }

    config_str = os.getenv("LLAMAFIREWALL_SCANNERS", "{}")
    logger.debug(f"Parsing scanner configuration: {config_str}")

    try:
        # Parse the JSON configuration
        config_dict = json.loads(config_str)

        # Validate configuration structure
        if not isinstance(config_dict, dict):
            logger.error("Invalid scanner configuration: must be a JSON object")
            raise ValueError("Invalid scanner configuration: must be a JSON object")

        # Convert string keys to Role enum and string values to ScannerType enum
        scanners = {}
        for role_str, scanner_list in config_dict.items():
            if not isinstance(scanner_list, list):
                logger.error(f"Invalid scanner list for role {role_str}: must be an array")
                raise ValueError(f"Invalid scanner list for role {role_str}: must be an array")

            try:
                role = Role[role_str]
                # Handle MODERATION scanner type separately
                scanners[role] = []
                for scanner in scanner_list:
                    if not isinstance(scanner, str):
                        logger.error(f"Invalid scanner type: {scanner}")
                        raise ValueError(f"Invalid scanner type: {scanner}")
                    if scanner == "MODERATION":
                        # Check if OpenAI API key is configured
                        if not os.getenv("OPENAI_API_KEY"):
                            logger.error("OPENAI_API_KEY environment variable is required when using MODERATION scanner")
                            raise ValueError("OPENAI_API_KEY environment variable is required when using MODERATION scanner")
                        moderation = True
                        logger.info("OpenAI moderation enabled")
                    else:
                        scanners[role].append(ScannerType[scanner])
            except KeyError as e:
                logger.error(f"Invalid role or scanner type: {e}")
                raise ValueError(f"Invalid role or scanner type: {e}")
        
        logger.info(f"Scanner configuration loaded: {scanners if scanners else default_config}")
        return scanners if scanners else default_config
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in LLAMAFIREWALL_SCANNERS, using default configuration")
        return default_config

# Cache scanner configuration at startup
SCANNER_CONFIG = parse_scanners_config()

# Initialize LlamaFirewall with cached config
llamafirewall = LlamaFirewall(scanners=SCANNER_CONFIG)

# Initialize OpenAI client
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

class ModerationResult(BaseModel, frozen=True):
    """Model for a single moderation result."""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]

class OpenAIModerationResponse(BaseModel, frozen=True):
    """Model for OpenAI moderation response."""
    id: str
    model: str
    results: List[ModerationResult]

class ScanRequest(BaseModel, frozen=True):
    """Request model for scanning messages."""
    content: constr(min_length=1, max_length=10000)  # Constrain content length between 1 and 10000 characters

    @validator('content')
    def validate_content(cls, v):
        """Validate content for potential security issues."""
        # Check for common injection patterns
        injection_patterns = [
            "<?php", "<script", "javascript:", "data:", "vbscript:",
            "onerror=", "onload=", "onclick=", "onmouseover="
        ]
        for pattern in injection_patterns:
            if pattern.lower() in v.lower():
                raise ValueError(f"Content contains potentially unsafe pattern: {pattern}")
        return v

class ScanResponse(BaseModel, frozen=True):
    """Unified response model for both scan types."""
    is_safe: bool
    risk_score: Optional[float] = None
    details: Optional[dict] = None
    moderation_results: Optional[OpenAIModerationResponse] = None
    scan_type: str

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def perform_openai_moderation(content: str) -> OpenAIModerationResponse:
    """Perform OpenAI moderation with retry logic."""
    try:
        logger.debug("Performing OpenAI moderation")
        response = await async_client.moderations.create(
            model="omni-moderation-latest",
            input=content
        )

        result = OpenAIModerationResponse(
            id=response.id,
            model=response.model,
            results=[
                ModerationResult(
                    flagged=result.flagged,
                    categories=result.categories.model_dump(),
                    category_scores=result.category_scores.model_dump()
                )
                for result in response.results
            ]
        )

        if any(r.flagged for r in result.results):
            logger.warning(f"Content flagged by OpenAI moderation: {result}")
        else:
            logger.debug("Content passed OpenAI moderation")

        return result
    except Exception as e:
        logger.error(f"Error during content moderation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error during content moderation"
        )

@app.post("/scan", response_model=ScanResponse)
async def scan_message(request: ScanRequest):
    """
    Scan a user message for potential security risks using LlamaFirewall and optionally OpenAI moderation.
    LlamaFirewall is always used, and OpenAI moderation is added when moderation=True.
    
    Args:
        request: ScanRequest containing the message content
        
    Returns:
        ScanResponse containing the scan results and metadata
        
    Raises:
        HTTPException: If there's an error during scanning
    """
    logger.info("Received scan request")
    try:
        # Always use LlamaFirewall first
        message = UserMessage(content=request.content)
        logger.debug("Starting LlamaFirewall scan")

        # Use thread pool for LlamaFirewall scan
        llama_result = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: llamafirewall.scan(message)
        )

        logger.info(f"LlamaFirewall scan result: decision={llama_result.decision}, score={llama_result.score}")

        # Initialize response with LlamaFirewall results
        response = ScanResponse(
            is_safe=True if llama_result.decision == "allow" else False,
            risk_score=llama_result.score,
            details={"reason": llama_result.reason},
            scan_type="llamafirewall"
        )
        
        # Add OpenAI moderation if enabled
        if moderation:
            logger.debug("Starting OpenAI moderation")
            # Perform OpenAI moderation asynchronously
            moderation_response = await perform_openai_moderation(request.content)

            # Update response with moderation results
            response = ScanResponse(
                is_safe=response.is_safe and not any(r.flagged for r in moderation_response.results),
                risk_score=response.risk_score,
                details={
                    "reason": response.details.get("reason", ""),
                    "flagged_categories": {
                        category: score
                        for result in moderation_response.results
                        for category, score in result.category_scores.items()
                        if score > 0.5
                    } if any(r.flagged for r in moderation_response.results) else None
                },
                moderation_results=moderation_response,
                scan_type="llamafirewall+openai_moderation"
            )
            logger.info(f"Final scan result with moderation: is_safe={response.is_safe}")

        return response

    except Exception as e:
        logger.error(f"Error processing scan request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing scan request"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    logger.debug("Health check requested")
    return {"status": "healthy"}

@app.get("/config")
async def get_config():
    """Get the current scanner configuration."""
    logger.debug("Config requested")
    config = {
        "scanners": {
            role.name: [scanner.name for scanner in scanners]
            for role, scanners in llamafirewall.scanners.items()
        }
    }

    # Add MODERATION to the list if enabled
    if moderation:
        for role in config["scanners"]:
            if "MODERATION" not in config["scanners"][role]:
                config["scanners"][role].append("MODERATION")

    logger.debug(f"Returning config: {config}")
    return config

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")
    thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")