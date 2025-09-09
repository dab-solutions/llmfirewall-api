from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, ValidationError
from llamafirewall import LlamaFirewall, UserMessage, Role, ScannerType, ScanDecision
from typing import Optional, Dict, List, Pattern, Tuple, Any
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import asyncio
import os
import json
import logging
import logging.handlers
from datetime import datetime
import re
import shutil
import uuid
from collections import deque
import time
import aiohttp
from urllib.parse import urlparse

# Import configuration manager
from config_manager import (
    ConfigurationManager, EndpointConfiguration, ConfigurationRecord,
    CreateConfigurationRequest, UpdateConfigurationRequest,
    get_config_manager, close_config_manager
)

# Load environment variables from .env file
load_dotenv()

# LLM Guard imports - conditionally imported based on configuration
try:
    from llm_guard import scan_prompt
    from llm_guard.input_scanners import (
        Anonymize, BanCompetitors, BanSubstrings, BanTopics, Code,
        Gibberish, InvisibleText, Language, PromptInjection, Regex, Secrets,
        Sentiment, TokenLimit, Toxicity
    )
    from llm_guard.vault import Vault
    LLMGUARD_AVAILABLE = True
except ImportError:
    LLMGUARD_AVAILABLE = False
    # Set to None to avoid unbound variable errors - will be checked at runtime
    scan_prompt = None
    # Type: ignore for these assignments since they're only used when LLMGUARD_AVAILABLE is True
    Anonymize = BanCompetitors = BanSubstrings = BanTopics = Code = None  # type: ignore
    Gibberish = InvisibleText = Language = PromptInjection = Regex = None  # type: ignore
    Secrets = Sentiment = TokenLimit = Toxicity = Vault = None  # type: ignore

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

def update_log_level(log_level: Optional[str] = None) -> None:
    """
    Update the logging level dynamically.
    
    Args:
        log_level: The new log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  If None, reads from environment variable.
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    else:
        log_level = log_level.upper()
    
    try:
        # Validate log level
        numeric_level = getattr(logging, log_level)
        
        # Update logger level
        logger.setLevel(numeric_level)
        
        # Update all handlers
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
        
        logger.info(f"Log level updated to: {log_level}")
    except AttributeError:
        logger.error(f"Invalid log level: {log_level}")
        raise ValueError(f"Invalid log level: {log_level}")

# Initialize log level
update_log_level(LOG_LEVEL)

# Get thread pool configuration from environment
THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "4"))  # Default to 4 workers if not specified
logger.info(f"Initializing thread pool with {THREAD_POOL_WORKERS} workers")

# Create thread pool executor at startup
thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting up application")

    # Initialize configuration manager
    try:
        await get_config_manager()
        logger.info("Configuration manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize configuration manager: {e}")

    yield

    # Shutdown
    logger.info("Shutting down application")
    thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")

    # Close configuration manager
    try:
        await close_config_manager()
        logger.info("Configuration manager closed")
    except Exception as e:
        logger.error(f"Error closing configuration manager: {e}")

app = FastAPI(
    title="LLM Firewall API",
    description="API for scanning user messages using LlamaFirewall and OpenAI moderation",
    version="1.1.0",
    lifespan=lifespan
)

# Mount static files for the web UI
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Pre-compile regex patterns
INJECTION_PATTERNS: List[Pattern] = [
    re.compile(pattern, re.IGNORECASE) for pattern in [
        r'<\?php',
        r'<script',
        r'javascript:',
        r'data:',
        r'vbscript:',
        r'on\w+=',
        r'exec\s*\(',
        r'eval\s*\(',
        r'system\s*\(',
        r'base64_decode\s*\(',
        r'from\s+import\s+',
        r'__import__\s*\(',
    ]
]

def sanitize_log_data(data: dict) -> dict:
    """Sanitize sensitive data for logging."""
    if not data:
        return data
    sanitized = data.copy()
    sensitive_fields = {'content', 'api_key', 'token', 'error'}
    for field in sensitive_fields:
        if field in sanitized:
            sanitized[field] = '[REDACTED]'
    return sanitized

# Request tracking system
class RequestTracker:
    """Thread-safe request tracker for monitoring API usage."""
    
    def __init__(self, max_requests: int = 1000):
        self.max_requests = max_requests
        self.requests = deque(maxlen=max_requests)
        self._lock = asyncio.Lock()
    
    async def add_request(self, request_data: dict):
        """Add a new request to the tracking system."""
        async with self._lock:
            self.requests.append(request_data)
    
    async def get_requests(self, limit: int = 100) -> List[dict]:
        """Get the most recent requests."""
        async with self._lock:
            return list(self.requests)[-limit:]
    
    async def clear_requests(self):
        """Clear all tracked requests."""
        async with self._lock:
            self.requests.clear()

# Global request tracker instance
request_tracker = RequestTracker(max_requests=1000)

# Global variables for configuration
moderation = False
LLMGUARD_ENABLED = False
LLMGUARD_SCANNERS = []
LLMGUARD_CONFIG = {}
SCANNER_CONFIG = {}
llamafirewall = None
async_client = None

# Configuration reload tracking
config_reload_in_progress = False
config_reload_last_status = "ready"

async def get_forwarding_enabled_endpoints():
    """
    Get all endpoint configurations that have forwarding enabled.
    
    Returns:
        List of endpoint configurations with forwarding enabled
    """
    try:
        config_manager = await get_config_manager()
        # Get all endpoint configurations
        all_configs = await config_manager.list_configurations(config_type="endpoint")
        
        # Filter for configurations with forwarding enabled
        forwarding_configs = []
        for config_record in all_configs:
            try:
                endpoint_config = EndpointConfiguration(**config_record.config_data)
                if endpoint_config.forwarding_enabled:
                    forwarding_configs.append((config_record.name, endpoint_config))
            except Exception as e:
                logger.warning(f"Failed to parse endpoint configuration {config_record.name}: {e}")
                continue
        
        return forwarding_configs
    except Exception as e:
        logger.error(f"Failed to get forwarding enabled endpoints: {e}")
        return []

def parse_llmguard_config() -> Tuple[bool, List[Any], Dict[str, Any]]:
    """
    Parse LLM Guard configuration from environment variables.
    
    Returns:
        Tuple of (enabled, scanners_list, config_dict)
    """
    enabled = os.getenv("LLMGUARD_ENABLED", "false").lower() == "true"
    logger.debug(f"LLM Guard enabled: {enabled}")
    logger.debug(f"LLM Guard available: {LLMGUARD_AVAILABLE}")

    if not enabled or not LLMGUARD_AVAILABLE:
        return False, [], {}
    
    # Parse input scanners
    input_scanners_str = os.getenv("LLMGUARD_INPUT_SCANNERS", "[]")
    try:
        input_scanners_list = json.loads(input_scanners_str)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in LLMGUARD_INPUT_SCANNERS: {input_scanners_str}")
        return False, [], {}
    
    # Configuration for scanner thresholds and settings
    config = {
        'toxicity_threshold': float(os.getenv("LLMGUARD_TOXICITY_THRESHOLD", "0.7")),
        'prompt_injection_threshold': float(os.getenv("LLMGUARD_PROMPT_INJECTION_THRESHOLD", "0.8")),
        'sentiment_threshold': float(os.getenv("LLMGUARD_SENTIMENT_THRESHOLD", "0.5")),
        'bias_threshold': float(os.getenv("LLMGUARD_BIAS_THRESHOLD", "0.7")),
        'token_limit': int(os.getenv("LLMGUARD_TOKEN_LIMIT", "4096")),
        'fail_fast': os.getenv("LLMGUARD_FAIL_FAST", "true").lower() == "true",
        'anonymize_entities': json.loads(os.getenv("LLMGUARD_ANONYMIZE_ENTITIES", '["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]')),
        'code_languages': json.loads(os.getenv("LLMGUARD_CODE_LANGUAGES", '["python", "javascript", "java", "sql"]')),
        'competitors': json.loads(os.getenv("LLMGUARD_COMPETITORS", '["openai", "anthropic", "google"]')),
        'banned_substrings': json.loads(os.getenv("LLMGUARD_BANNED_SUBSTRINGS", '["hack", "exploit", "bypass"]')),
        'banned_topics': json.loads(os.getenv("LLMGUARD_BANNED_TOPICS", '["violence", "illegal"]')),
        'valid_languages': json.loads(os.getenv("LLMGUARD_VALID_LANGUAGES", '["en"]')),
        'regex_patterns': json.loads(os.getenv("LLMGUARD_REGEX_PATTERNS", '["^(?!.*password).*$"]'))
    }
    
    # Build scanner instances
    scanners = []
    vault = None
    
    # Only try to build scanners if LLM Guard is actually available
    if not LLMGUARD_AVAILABLE:
        logger.warning("LLM Guard classes not available, cannot build scanners")
        return False, [], {}
    
    # Assert that the classes are available (helps type checker)
    assert Vault is not None and Anonymize is not None and Code is not None
    assert BanCompetitors is not None and BanSubstrings is not None and BanTopics is not None
    assert Gibberish is not None and InvisibleText is not None and Language is not None
    assert PromptInjection is not None and Regex is not None and Secrets is not None
    assert Sentiment is not None and TokenLimit is not None and Toxicity is not None
    
    for scanner_name in input_scanners_list:
        try:
            if scanner_name == "Anonymize":
                if vault is None:
                    vault = Vault()
                scanners.append(Anonymize(vault, entity_types=config['anonymize_entities']))
            elif scanner_name == "Code":
                languages = config.get('code_languages', ['python', 'javascript', 'java', 'sql'])
                scanners.append(Code(languages=languages))
            elif scanner_name == "BanCompetitors":
                scanners.append(BanCompetitors(competitors=config['competitors']))
            elif scanner_name == "BanSubstrings":
                scanners.append(BanSubstrings(substrings=config['banned_substrings']))
            elif scanner_name == "BanTopics":
                scanners.append(BanTopics(topics=config['banned_topics']))
            elif scanner_name == "Gibberish":
                scanners.append(Gibberish())
            elif scanner_name == "InvisibleText":
                scanners.append(InvisibleText())
            elif scanner_name == "Language":
                scanners.append(Language(valid_languages=config['valid_languages']))
            elif scanner_name == "PromptInjection":
                scanners.append(PromptInjection(threshold=config['prompt_injection_threshold']))
            elif scanner_name == "Regex":
                scanners.append(Regex(patterns=config['regex_patterns']))
            elif scanner_name == "Secrets":
                scanners.append(Secrets())
            elif scanner_name == "Sentiment":
                scanners.append(Sentiment(threshold=config['sentiment_threshold']))
            elif scanner_name == "TokenLimit":
                scanners.append(TokenLimit(limit=config['token_limit']))
            elif scanner_name == "Toxicity":
                scanners.append(Toxicity(threshold=config['toxicity_threshold']))
            else:
                logger.warning(f"Unknown LLM Guard scanner: {scanner_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Guard scanner {scanner_name}: {e}")
            # Continue with other scanners instead of failing completely
    
    return enabled, scanners, config

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

    if "MODERATION" in config_str:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when using MODERATION scanner")
        moderation = True
        logger.info("OpenAI moderation enabled")

    # Parse the JSON configuration
    try:
        config_dict = json.loads(config_str)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in scanner configuration", extra=sanitize_log_data({"error": str(e)}))
        raise ValueError("Invalid JSON format in scanner configuration") from e

    # Validate configuration structure
    if not isinstance(config_dict, dict):
        raise ValueError("Invalid scanner configuration: must be a JSON object")

    # Convert string keys to Role enum and string values to ScannerType enum
    scanners = {}
    for role_str, scanner_list in config_dict.items():
        if not isinstance(scanner_list, list):
            raise ValueError(f"Invalid scanner list for role {role_str}")

        try:
            role = Role[role_str]
            scanners[role] = []
            for scanner in scanner_list:
                if not isinstance(scanner, str):
                    raise ValueError(f"Invalid scanner type format: {scanner}")
                if scanner != "MODERATION":
                    scanners[role].append(ScannerType[scanner])
        except KeyError as e:
            raise ValueError(f"Invalid scanner configuration: {e}")

    logger.info("Scanner configuration loaded successfully")
    return scanners if scanners else default_config

async def reload_configuration():
    """
    Reload application configuration from environment variables.
    This function updates all global configuration variables dynamically.
    """
    global moderation, LLMGUARD_ENABLED, LLMGUARD_SCANNERS, LLMGUARD_CONFIG
    global SCANNER_CONFIG, llamafirewall, async_client
    global config_reload_in_progress, config_reload_last_status

    if config_reload_in_progress:
        logger.warning("Configuration reload already in progress")
        return {"status": "already_in_progress", "message": "Configuration reload already in progress"}

    config_reload_in_progress = True
    config_reload_last_status = "reloading"

    try:
        logger.info("Starting configuration reload")

        # Reload environment variables from .env file
        load_dotenv(override=True)

        # Update log level if changed
        new_log_level = os.getenv("LOG_LEVEL", "INFO")
        update_log_level(new_log_level)

        # Parse scanner configuration
        logger.info("Reloading scanner configuration")
        SCANNER_CONFIG = parse_scanners_config()

        # Reinitialize LlamaFirewall with new config
        logger.info("Reinitializing LlamaFirewall")
        llamafirewall = LlamaFirewall(scanners=SCANNER_CONFIG)

        # Parse and update LLM Guard configuration
        logger.info("Reloading LLM Guard configuration")
        LLMGUARD_ENABLED, LLMGUARD_SCANNERS, LLMGUARD_CONFIG = parse_llmguard_config()
        if LLMGUARD_ENABLED:
            logger.info(f"LLM Guard enabled with {len(LLMGUARD_SCANNERS)} scanners")
        else:
            logger.info("LLM Guard disabled")

        # Update OpenAI client if moderation settings changed
        if moderation:
            logger.info("Initializing OpenAI client for moderation")
            async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        else:
            logger.info("OpenAI moderation disabled")
            async_client = None

        config_reload_last_status = "success"
        logger.info("Configuration reload completed successfully")

        return {
            "status": "success", 
            "message": "Configuration reloaded successfully",
            "details": {
                "scanner_config": len(SCANNER_CONFIG),
                "llmguard_enabled": LLMGUARD_ENABLED,
                "llmguard_scanners": len(LLMGUARD_SCANNERS) if LLMGUARD_SCANNERS else 0,
                "moderation_enabled": moderation
            }
        }

    except Exception as e:
        config_reload_last_status = f"error: {str(e)}"
        logger.error(f"Failed to reload configuration: {e}", exc_info=True)
        return {
            "status": "error", 
            "message": f"Failed to reload configuration: {str(e)}"
        }
    finally:
        config_reload_in_progress = False

# Initialize application with proper error handling
try:
    # Cache scanner configuration at startup
    logger.info("Initializing scanner configuration")
    SCANNER_CONFIG = parse_scanners_config()

    # Initialize LlamaFirewall with cached config
    logger.info("Initializing LlamaFirewall")
    llamafirewall = LlamaFirewall(scanners=SCANNER_CONFIG)

    # Initialize LLM Guard if enabled
    logger.info("Initializing LLM Guard configuration")
    LLMGUARD_ENABLED, LLMGUARD_SCANNERS, LLMGUARD_CONFIG = parse_llmguard_config()
    if LLMGUARD_ENABLED:
        logger.info(f"LLM Guard enabled with {len(LLMGUARD_SCANNERS)} scanners")
    else:
        logger.info("LLM Guard disabled")

    # Initialize OpenAI client if moderation is enabled
    if moderation:
        logger.info("Initializing OpenAI client")
        async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    else:
        async_client = None

    config_reload_last_status = "initialized"

except ValueError as e:
    logger.error(f"Failed to initialize application: {e}", extra=sanitize_log_data({"error": str(e)}))
    exit(1)
except Exception as e:
    logger.error(f"Unexpected error during initialization: {e}", extra=sanitize_log_data({"error": str(e)}))
    exit(1)

class ModerationResult(BaseModel):
    """Model for a single moderation result."""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    
    model_config = {"frozen": True}

class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration."""
    config_data: Dict[str, Any] = Field(description="Configuration data to update")
    
    model_config = {"frozen": True}
    
    @field_validator('config_data')
    @classmethod
    def validate_config_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration data for security."""
        if not isinstance(v, dict):
            raise ValueError("Configuration data must be a dictionary")
        
        # Check for reasonable size limits to prevent DoS
        if len(str(v)) > 50000:  # 50KB limit
            raise ValueError("Configuration data too large")
            
        return v

class ConfigResponse(BaseModel):
    """Response model for configuration operations."""
    success: bool
    message: str
    config: Optional[Dict[str, Any]] = None

    model_config = {"frozen": True}

class OpenAIModerationResponse(BaseModel):
    """Model for OpenAI moderation response."""
    id: str
    model: str
    results: List[ModerationResult]

    model_config = {"frozen": True}

class ScanRequest(BaseModel):
    """Request model for scanning messages."""
    content: str = Field(min_length=1, max_length=10000)  # Constrain content length between 1 and 10000 characters

    model_config = {"frozen": True}

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content for potential security issues."""
        # Check for injection patterns
        for pattern in INJECTION_PATTERNS:
            if pattern.search(v):
                raise ValueError("Content contains potentially unsafe patterns")

        # Check for excessive whitespace (potential DoS)
        if len(v.strip()) == 0:
            raise ValueError("Content cannot be empty or whitespace only")

        # Check for excessive repeated characters (potential DoS)
        if any(c * 100 in v for c in set(v)):
            raise ValueError("Content contains excessive repeated characters")

        return v

class ForwardingResponse(BaseModel):
    """Response model for forwarding results."""
    success: bool
    status_code: Optional[int] = None
    response_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    model_config = {"frozen": True}

class ScanResponse(BaseModel):
    """Unified response model for both scan types."""
    is_safe: bool
    risk_score: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    moderation_results: Optional[OpenAIModerationResponse] = None
    llmguard_results: Optional[Dict[str, Any]] = None
    forwarding_result: Optional[ForwardingResponse] = None
    scan_type: str

    model_config = {"frozen": True}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def perform_openai_moderation(content: str) -> OpenAIModerationResponse:
    """Perform OpenAI moderation with retry logic."""
    if async_client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI client not initialized"
        )

    try:
        logger.debug("Performing OpenAI moderation", extra=sanitize_log_data({"content": content}))
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
            logger.warning("Content flagged by OpenAI moderation", extra=sanitize_log_data({
                "flagged": True,
                "categories": {k: v for r in result.results for k, v in r.categories.items() if v}
            }))
        else:
            logger.debug("Content passed OpenAI moderation")

        return result
    except Exception as e:
        logger.error("Error during content moderation", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
        raise HTTPException(
            status_code=500,
            detail="Error during content moderation"
        )

async def perform_llmguard_scan(content: str) -> Tuple[str, Dict[str, bool], Dict[str, float]]:
    """
    Perform LLM Guard scanning with retry logic.

    Returns:
        Tuple of (sanitized_content, validation_results, risk_scores)
    """
    if not LLMGUARD_ENABLED or not LLMGUARD_SCANNERS:
        return content, {}, {}

    if scan_prompt is None:
        logger.error("LLM Guard scan_prompt function not available")
        return content, {}, {}

    try:
        logger.debug("Performing LLM Guard scan", extra=sanitize_log_data({"content": content}))

        # Call the actual scan function directly (we've checked it's not None above)
        def _do_scan():
            # At this point we know scan_prompt is not None due to the check above
            return scan_prompt(LLMGUARD_SCANNERS, content, fail_fast=LLMGUARD_CONFIG.get('fail_fast', True))  # type: ignore

        # Use thread pool for LLM Guard scan since it's CPU intensive
        result = await asyncio.get_event_loop().run_in_executor(thread_pool, _do_scan)

        sanitized_content, validation_results, risk_scores = result

        # Log results
        failed_scanners = [name for name, valid in validation_results.items() if not valid]
        if failed_scanners:
            logger.warning("Content flagged by LLM Guard scanners", extra=sanitize_log_data({
                "failed_scanners": failed_scanners,
                "risk_scores": {k: v for k, v in risk_scores.items() if k in failed_scanners}
            }))
        else:
            logger.debug("Content passed LLM Guard scan")
        
        return sanitized_content, validation_results, risk_scores

    except Exception as e:
        logger.error("Error during LLM Guard scan", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
        # Don't fail the request if LLM Guard fails, just log and continue
        return content, {}, {}

async def forward_request_to_endpoint(
    content: str, 
    endpoint_url: Optional[str] = None,
    endpoint_config: Optional[EndpointConfiguration] = None,
    additional_headers: Optional[Dict[str, str]] = None,
    is_safe: bool = True
) -> ForwardingResponse:
    """
    Forward the scanned content to the specified endpoint using configuration.

    Args:
        content: The original content that was scanned
        endpoint_url: Direct URL to forward to (overrides config)
        endpoint_config: EndpointConfiguration object with forwarding settings
        additional_headers: Additional headers to include
        is_safe: Whether the content was deemed safe by scanning

    Returns:
        ForwardingResponse containing the result of the forwarding operation
    """
    url = "unknown"  # Initialize for error handling

    try:
        # Determine which configuration to use
        if endpoint_url:
            # Use direct URL with default configuration
            url = endpoint_url
            headers = {"Content-Type": "application/json", "User-Agent": "LlamaFirewall-API/1.1.0"}
            timeout_seconds = 30
            method = "POST"
            verify_ssl = True
            include_scan_results = True
            forward_on_unsafe = False
        elif endpoint_config:
            # Use stored configuration
            url = endpoint_config.url
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "LlamaFirewall-API/1.1.0",
                **endpoint_config.headers
            }
            timeout_seconds = endpoint_config.timeout
            method = endpoint_config.method
            verify_ssl = endpoint_config.verify_ssl
            include_scan_results = endpoint_config.include_scan_results
            forward_on_unsafe = endpoint_config.forward_on_unsafe
        else:
            raise ValueError("Either endpoint_url or endpoint_config must be provided")

        # Check if we should forward unsafe content
        if not is_safe and not forward_on_unsafe:
            logger.info(f"Skipping forwarding to {url} because content is unsafe and forward_on_unsafe=False")
            return ForwardingResponse(
                success=False,
                status_code=None,
                response_data=None,
                error="Content marked as unsafe and forward_on_unsafe is disabled"
            )

        # Add additional headers if provided
        if additional_headers:
            # Filter out dangerous headers (already done in validation)
            headers.update(additional_headers)
        
        logger.info(f"Forwarding request to endpoint: {url}")

        # Prepare the payload to send
        payload: Dict[str, Any] = {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "source": "llama_firewall_api"
        }

        # Include scan results if configured
        if include_scan_results:
            payload["is_safe"] = is_safe

        # Set up HTTP client with security configurations
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        connector = aiohttp.TCPConnector(
            limit=10,  # Limit concurrent connections
            ssl=verify_ssl   # Verify SSL certificates based on config
        )

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        ) as session:
            # Use the configured HTTP method
            async with session.request(method, url, json=payload) as response:
                # Read response content
                try:
                    response_data = await response.json()
                except aiohttp.ContentTypeError:
                    # If response is not JSON, read as text
                    response_text = await response.text()
                    response_data = {"response": response_text}

                success = 200 <= response.status < 300

                if success:
                    logger.info(f"Successfully forwarded request to {url} (status: {response.status})")
                else:
                    logger.warning(f"Forwarding request failed with status {response.status}: {response_data}")

                return ForwardingResponse(
                    success=success,
                    status_code=response.status,
                    response_data=response_data,
                    error=None if success else f"HTTP {response.status}: {response_data}"
                )

    except aiohttp.ClientError as e:
        error_msg = f"Network error while forwarding to {url}: {str(e)}"
        logger.error(error_msg)
        return ForwardingResponse(
            success=False,
            status_code=None,
            response_data=None,
            error=error_msg
        )
    except asyncio.TimeoutError:
        error_msg = f"Timeout while forwarding to {url}"
        logger.error(error_msg)
        return ForwardingResponse(
            success=False,
            status_code=None,
            response_data=None,
            error=error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error while forwarding to {url}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return ForwardingResponse(
            success=False,
            status_code=None,
            response_data=None,
            error=error_msg
        )

@app.post("/scan", response_model=ScanResponse)
async def scan_message(request: ScanRequest, http_request: Request):
    """
    Scan a user message for potential security risks using LlamaFirewall, LLM Guard, and optionally OpenAI moderation.
    
    Args:
        request: ScanRequest containing the message content
        http_request: FastAPI Request object for tracking
        
    Returns:
        ScanResponse containing the scan results and metadata
        
    Raises:
        HTTPException: If there's an error during scanning
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Track the incoming request
    request_data = {
        "id": request_id,
        "timestamp": datetime.now().isoformat(),
        "method": "POST",
        "endpoint": "/scan",
        "client_ip": http_request.client.host if http_request.client else "unknown",
        "user_agent": http_request.headers.get("user-agent", "unknown"),
        "content_length": len(request.content),
        "content_preview": request.content[:100] + "..." if len(request.content) > 100 else request.content,
        "status": "processing",
        "response": None,
        "processing_time_ms": None,
        "error": None
    }

    logger.info("Received scan request", extra=sanitize_log_data({"content": request.content}))

    response = None  # Initialize response variable
    llama_result = None  # Initialize llama_result variable
    try:
        # Step 1: Always use LlamaFirewall first
        message = UserMessage(content=request.content)
        logger.debug("Starting LlamaFirewall scan")

        try:
            # Use thread pool for LlamaFirewall scan
            if llamafirewall is None:
                raise HTTPException(
                    status_code=503,
                    detail="LlamaFirewall not initialized. Please reload configuration."
                )

            # Now we know llamafirewall is not None
            firewall_instance = llamafirewall
            llama_result = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: firewall_instance.scan(message)
            )
        except Exception as e:
            logger.error("LlamaFirewall scan failed", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable"
            )

        # Log sanitized result
        logger.info("LlamaFirewall scan completed", extra=sanitize_log_data({
            "decision": llama_result.decision,
            "score": llama_result.score,
            "reason": llama_result.reason
        }))

        is_safe = llama_result.decision == ScanDecision.ALLOW
        details: Dict[str, Any] = {"reason": llama_result.reason}
        scan_types = ["llamafirewall"]

        # Step 2: Perform LLM Guard scan if enabled
        llmguard_results = None
        if LLMGUARD_ENABLED:
            try:
                logger.debug("Starting LLM Guard scan")
                sanitized_content, validation_results, risk_scores = await perform_llmguard_scan(request.content)
                
                llmguard_results = {
                    "sanitized_content": sanitized_content,
                    "validation_results": validation_results,
                    "risk_scores": risk_scores
                }
                
                # Update safety status based on LLM Guard results
                llmguard_safe = all(validation_results.values()) if validation_results else True
                is_safe = is_safe and llmguard_safe
                
                if not llmguard_safe:
                    failed_scanners = [name for name, valid in validation_results.items() if not valid]
                    details["llmguard_failed"] = failed_scanners
                    details["llmguard_scores"] = {k: v for k, v in risk_scores.items() if k in failed_scanners}
                
                scan_types.append("llm_guard")
                logger.info("LLM Guard scan completed", extra=sanitize_log_data({
                    "is_safe": llmguard_safe,
                    "failed_scanners": details.get("llmguard_failed", [])
                }))
            except Exception as e:
                logger.error("LLM Guard scan failed, continuing without it", exc_info=True, extra=sanitize_log_data({"error": str(e)}))

        response = ScanResponse(
            is_safe=is_safe,
            risk_score=llama_result.score,
            details=details,
            llmguard_results=llmguard_results,
            forwarding_result=None,  # Will be updated if forwarding is requested
            scan_type="+".join(scan_types)
        )

        # Step 3: Perform OpenAI moderation if enabled
        if moderation:
            try:
                logger.debug("Starting OpenAI moderation")
                # Perform OpenAI moderation asynchronously
                moderation_response = await perform_openai_moderation(request.content)

                # Update response with moderation results
                moderation_safe = not any(r.flagged for r in moderation_response.results)
                flagged_categories = {
                    category: score
                    for result in moderation_response.results
                    for category, score in result.category_scores.items()
                    if score > 0.5
                } if not moderation_safe else None
                
                updated_details = dict(response.details) if response.details else {}
                if flagged_categories:
                    updated_details["flagged_categories"] = flagged_categories
                
                response = ScanResponse(
                    is_safe=response.is_safe and moderation_safe,
                    risk_score=response.risk_score,
                    details=updated_details,
                    moderation_results=moderation_response,
                    llmguard_results=response.llmguard_results,
                    forwarding_result=response.forwarding_result,
                    scan_type=f"{response.scan_type}+openai_moderation"
                )
                logger.info("Moderation scan completed", extra=sanitize_log_data({
                    "is_safe": moderation_safe,
                    "flagged_categories": flagged_categories
                }))
            except HTTPException:
                raise
            except Exception as e:
                logger.error("OpenAI moderation failed", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable"
                )

        # Step 4: Forward request to all endpoints with forwarding enabled
        forwarding_enabled_endpoints = await get_forwarding_enabled_endpoints()
        if forwarding_enabled_endpoints:
            # Forward to all enabled endpoints
            forwarding_results = []
            for endpoint_name, endpoint_config in forwarding_enabled_endpoints:
                logger.debug(f"Forwarding to endpoint: {endpoint_name}")
   
                try:
                    # Forward the request using the endpoint configuration
                    forwarding_result = await forward_request_to_endpoint(
                        content=request.content,
                        endpoint_url=None,  # Use configuration URL
                        endpoint_config=endpoint_config,
                        additional_headers={},  # No additional headers from request
                        is_safe=response.is_safe
                    )

                    forwarding_results.append({
                        "endpoint": endpoint_name,
                        "result": forwarding_result
                    })

                    logger.info("Request forwarding completed", extra=sanitize_log_data({
                        "endpoint": endpoint_name,
                        "success": forwarding_result.success,
                        "status_code": forwarding_result.status_code
                    }))

                except Exception as forward_error:
                    logger.error(f"Forwarding failed for endpoint {endpoint_name}", exc_info=True, extra=sanitize_log_data({"error": str(forward_error)}))
                    # Continue with other endpoints even if one fails
                    forwarding_results.append({
                        "endpoint": endpoint_name,
                        "result": ForwardingResponse(
                            success=False,
                            status_code=None,
                            response_data=None,
                            error=f"Forwarding error: {str(forward_error)}"
                        )
                    })

            # Update response with forwarding results (use the first successful result for backward compatibility)
            main_forwarding_result = None
            for fr in forwarding_results:
                if fr["result"].success:
                    main_forwarding_result = fr["result"]
                    break

            # If no successful results, use the first result
            if not main_forwarding_result and forwarding_results:
                main_forwarding_result = forwarding_results[0]["result"]

            if main_forwarding_result:
                response = ScanResponse(
                    is_safe=response.is_safe,
                    risk_score=response.risk_score,
                    details=response.details,
                    moderation_results=response.moderation_results,
                    llmguard_results=response.llmguard_results,
                    forwarding_result=main_forwarding_result,
                    scan_type=f"{response.scan_type}+forwarded"
                )

        return response

    except HTTPException as e:
        # Update request tracking with error
        processing_time = (time.time() - start_time) * 1000
        request_data.update({
            "status": "error", 
            "processing_time_ms": round(processing_time, 2),
            "error": f"HTTP {e.status_code}: {e.detail}"
        })
        await request_tracker.add_request(request_data)
        raise
    except ValueError as e:
        # Handle validation errors
        processing_time = (time.time() - start_time) * 1000
        request_data.update({
            "status": "error", 
            "processing_time_ms": round(processing_time, 2),
            "error": f"Validation error: {str(e)}"
        })
        await request_tracker.add_request(request_data)
        logger.error("Validation error", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
        raise HTTPException(
            status_code=400,
            detail="Invalid request format"
        )
    except Exception as e:
        # Handle unexpected errors
        processing_time = (time.time() - start_time) * 1000
        request_data.update({
            "status": "error", 
            "processing_time_ms": round(processing_time, 2),
            "error": f"Internal error: {str(e)}"
        })
        await request_tracker.add_request(request_data)
        logger.error("Unexpected error", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
    finally:
        # Update request tracking with final result (success or error)
        if request_data["status"] == "processing" and response is not None:
            processing_time = (time.time() - start_time) * 1000

            # Extract risks found from the response details
            risks_found = []
            if response.details:
                if "reason" in response.details and response.details["reason"]:
                    risks_found.append(response.details["reason"])
                if "llmguard_failed" in response.details:
                    risks_found.extend(response.details["llmguard_failed"])
            
            # Build detailed scan results for monitoring
            scan_results = {}
            
            # Add LlamaFirewall results
            if llama_result is not None:
                scan_results["llamafirewall"] = {
                    "decision": llama_result.decision.value if hasattr(llama_result, 'decision') else "unknown",
                    "score": llama_result.score if hasattr(llama_result, 'score') else None,
                    "reason": llama_result.reason if hasattr(llama_result, 'reason') else None,
                    "is_safe": llama_result.decision == ScanDecision.ALLOW if hasattr(llama_result, 'decision') else None
                }
            else:
                scan_results["llamafirewall"] = {
                    "decision": "error",
                    "score": None,
                    "reason": "LlamaFirewall scan failed",
                    "is_safe": False
                }

            # Add LLM Guard results if available
            if response.llmguard_results:
                scan_results["llmguard"] = {
                    "enabled": True,
                    "validation_results": response.llmguard_results.get("validation_results", {}),
                    "risk_scores": response.llmguard_results.get("risk_scores", {}),
                    "failed_scanners": response.details.get("llmguard_failed", []) if response.details else [],
                    "is_safe": all(response.llmguard_results.get("validation_results", {}).values()) if response.llmguard_results.get("validation_results") else True
                }
            else:
                scan_results["llmguard"] = {"enabled": False}

            # Add OpenAI Moderation results if available
            if response.moderation_results:
                moderation_safe = not any(r.flagged for r in response.moderation_results.results)
                flagged_categories = {}
                if not moderation_safe:
                    for result in response.moderation_results.results:
                        for category, score in result.category_scores.items():
                            if score > 0.5:
                                flagged_categories[category] = score

                scan_results["openai_moderation"] = {
                    "enabled": True,
                    "is_safe": moderation_safe,
                    "flagged_categories": flagged_categories,
                    "model": response.moderation_results.model
                }
            else:
                scan_results["openai_moderation"] = {"enabled": False}

            # Add forwarding results if available
            if response.forwarding_result:
                # Get the endpoint name from the forwarding enabled endpoints
                forwarding_endpoints = await get_forwarding_enabled_endpoints()
                endpoint_name = forwarding_endpoints[0][0] if forwarding_endpoints else "unknown"

                scan_results["forwarding"] = {
                    "enabled": True,
                    "endpoint": endpoint_name,
                    "success": response.forwarding_result.success,
                    "status_code": response.forwarding_result.status_code,
                    "error": response.forwarding_result.error
                }
            else:
                scan_results["forwarding"] = {"enabled": False}

            request_data.update({
                "status": "success",
                "processing_time_ms": round(processing_time, 2),
                "response": {
                    "is_safe": response.is_safe,
                    "score": response.risk_score,  # Store as 'score' to match UI expectations
                    "risks_found": risks_found,    # Store risks found for UI display
                    "scan_type": response.scan_type,
                    "has_moderation": response.moderation_results is not None,
                    "has_llmguard": response.llmguard_results is not None,
                    "has_forwarding": response.forwarding_result is not None,
                    "details": response.details,
                    "scan_results": scan_results  # Detailed breakdown by system
                }
            })

        # Always add the final request data (even for errors, which are handled in except blocks)
        if request_data["status"] != "processing":  # Only add if status was updated
            await request_tracker.add_request(request_data)

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the configuration web UI."""
    return FileResponse("static/index.html")

@app.get("/api/env-config")
async def get_env_config():
    """Get current environment configuration (sanitized for security)."""
    logger.debug("Environment config requested")

    try:
        # Read current .env file if it exists
        env_file_path = ".env"
        config = {}

        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Sanitize sensitive values for display (exclude token limits and other config values)
                        sensitive_patterns = ['_TOKEN', '_KEY', '_SECRET']
                        exclude_patterns = ['_TOKEN_LIMIT', '_LIMIT']

                        is_sensitive = any(pattern in key.upper() for pattern in sensitive_patterns)
                        is_excluded = any(pattern in key.upper() for pattern in exclude_patterns)

                        if is_sensitive and not is_excluded:
                            config[key] = '[REDACTED]' if value and value != 'your_huggingface_token_here' else ''
                        else:
                            config[key] = value

        # Provide default values for missing configuration keys
        defaults = {
            'LOG_LEVEL': 'INFO',
            'THREAD_POOL_WORKERS': '4',
            'HF_TOKEN': '',
            'TOGETHER_API_KEY': '',
            'OPENAI_API_KEY': '',
            'LLAMAFIREWALL_SCANNERS': '{"USER": ["PROMPT_GUARD"]}',
            'LLMGUARD_ENABLED': 'false',
            'LLMGUARD_INPUT_SCANNERS': '["PromptInjection", "Toxicity", "Secrets"]',
            'LLMGUARD_TOXICITY_THRESHOLD': '0.7',
            'LLMGUARD_PROMPT_INJECTION_THRESHOLD': '0.8',
            'LLMGUARD_SENTIMENT_THRESHOLD': '0.5',
            'LLMGUARD_BIAS_THRESHOLD': '0.7',
            'LLMGUARD_TOKEN_LIMIT': '4096',
            'LLMGUARD_FAIL_FAST': 'true',
            'LLMGUARD_ANONYMIZE_ENTITIES': '["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]',
            'LLMGUARD_CODE_LANGUAGES': '["python", "javascript", "java", "sql"]',
            'LLMGUARD_COMPETITORS': '["openai", "anthropic", "google"]',
            'LLMGUARD_BANNED_SUBSTRINGS': '["hack", "exploit", "bypass"]',
            'LLMGUARD_BANNED_TOPICS': '["violence", "illegal"]',
            'LLMGUARD_VALID_LANGUAGES': '["en"]',
            'LLMGUARD_REGEX_PATTERNS': '["^(?!.*password).*$"]',
            'TOKENIZERS_PARALLELISM': 'false'
        }

        # Add default values for missing keys
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value

        return ConfigResponse(
            success=True,
            message="Configuration retrieved successfully",
            config=config
        )
    except Exception as e:
        logger.error(f"Error reading configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error reading configuration"
        )

def validate_configuration(config_data: Dict[str, str]) -> None:
    """
    Validate configuration data for common issues.
    
    Args:
        config_data: Dictionary of configuration key-value pairs
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate required fields (skip if value is [REDACTED] as it means the token exists)
    required_fields = ['HF_TOKEN']
    for field in required_fields:
        value = config_data.get(field, '')
        if not value or (value in ['', 'your_huggingface_token_here'] and value != '[REDACTED]'):
            raise ValueError(f"{field} is required and cannot be empty")

    # Validate numeric fields
    numeric_fields = ['THREAD_POOL_WORKERS', 'LLMGUARD_TOKEN_LIMIT']
    for field in numeric_fields:
        if field in config_data:
            try:
                value = int(config_data[field])
                if value <= 0:
                    raise ValueError(f"{field} must be a positive integer")
            except ValueError:
                raise ValueError(f"{field} must be a valid positive integer")
    
    # Validate LOG_LEVEL
    if 'LOG_LEVEL' in config_data:
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config_data['LOG_LEVEL'].upper() not in valid_log_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(valid_log_levels)}")

    # Validate threshold fields (0.0 to 1.0)
    threshold_fields = [
        'LLMGUARD_TOXICITY_THRESHOLD',
        'LLMGUARD_PROMPT_INJECTION_THRESHOLD',
        'LLMGUARD_SENTIMENT_THRESHOLD',
        'LLMGUARD_BIAS_THRESHOLD'
    ]
    for field in threshold_fields:
        if field in config_data:
            try:
                value = float(config_data[field])
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{field} must be between 0.0 and 1.0")
            except ValueError:
                raise ValueError(f"{field} must be a valid float between 0.0 and 1.0")

    # Validate JSON fields
    json_fields = [
        'LLAMAFIREWALL_SCANNERS',
        'LLMGUARD_INPUT_SCANNERS',
        'LLMGUARD_ANONYMIZE_ENTITIES',
        'LLMGUARD_CODE_LANGUAGES',
        'LLMGUARD_COMPETITORS',
        'LLMGUARD_BANNED_SUBSTRINGS',
        'LLMGUARD_BANNED_TOPICS',
        'LLMGUARD_VALID_LANGUAGES',
        'LLMGUARD_REGEX_PATTERNS'
    ]
    for field in json_fields:
        if field in config_data:
            try:
                json.loads(config_data[field])
            except json.JSONDecodeError:
                raise ValueError(f"{field} must be valid JSON")

@app.post("/api/env-config", response_model=ConfigResponse)
async def update_env_config(request: ConfigUpdateRequest):
    """Update environment configuration with validation and security checks."""
    logger.info("Environment configuration update requested")

    try:
        # Validate configuration data
        config_data = request.config_data

        # Ensure all values are strings (convert objects to JSON strings if needed)
        validation_config = {}
        for key, value in config_data.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid key: {key}")

            # Convert dictionary or list values to JSON strings
            if isinstance(value, (dict, list)):
                validation_config[key] = json.dumps(value)
            elif not isinstance(value, str):
                validation_config[key] = str(value)
            else:
                validation_config[key] = value

            # Check for malicious patterns
            if any(pattern in str(validation_config[key]) for pattern in ['../', '..\\', '<script', 'javascript:', 'data:']):
                raise ValueError(f"Potentially unsafe value for {key}")

        # Read existing configuration to preserve sensitive values
        env_file_path = ".env"
        existing_config = {}
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        existing_config[key] = value

        # Merge new config with existing, preserving sensitive values if they weren't actually changed
        merged_config = existing_config.copy()
        for key, value in config_data.items():
            # If the value is [REDACTED], keep the existing value
            if value == '[REDACTED]' and key in existing_config:
                merged_config[key] = existing_config[key]
            else:
                merged_config[key] = value

        # Validate using the new validation function (but skip validation for [REDACTED] values)
        validation_config = {k: v for k, v in merged_config.items() if v != '[REDACTED]'}
        validate_configuration(validation_config)

        # Create backup of current .env file
        if os.path.exists(env_file_path):
            shutil.copy2(env_file_path, f"{env_file_path}.backup")

        # Write new configuration
        with open(env_file_path, 'w') as f:
            f.write("# LLM Firewall API Configuration\n")
            f.write(f"# Updated: {datetime.now().isoformat()}\n\n")

            for key, value in sorted(merged_config.items()):
                f.write(f"{key}={value}\n")

        # Update log level dynamically if it was changed
        if 'LOG_LEVEL' in config_data:
            try:
                update_log_level(config_data['LOG_LEVEL'])
                logger.info(f"Log level dynamically updated to: {config_data['LOG_LEVEL']}")
            except ValueError as e:
                logger.warning(f"Failed to update log level: {e}")

        logger.info("Environment configuration updated successfully")
        return ConfigResponse(
            success=True,
            message="Configuration updated successfully. Use /api/reload-config to apply changes immediately.",
            config=None  # Don't return the full config for security
        )

    except ValueError as e:
        logger.error(f"Validation error in configuration update: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating configuration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error updating configuration"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    logger.debug("Health check requested")
    return {"status": "healthy"}

@app.post("/api/reload-config")
async def reload_config():
    """Reload application configuration from environment variables."""
    logger.info("Configuration reload requested")
    try:
        result = await reload_configuration()
        return result
    except Exception as e:
        logger.error(f"Error during configuration reload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading configuration: {str(e)}"
        )

@app.get("/api/reload-config/status")
async def get_reload_status():
    """Get the current status of configuration reload."""
    global config_reload_in_progress, config_reload_last_status
    return {
        "in_progress": config_reload_in_progress,
        "last_status": config_reload_last_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/requests")
async def get_requests(limit: int = 100):
    """Get the most recent API requests for monitoring."""
    logger.debug(f"Requests endpoint accessed, limit: {limit}")
    try:
        requests = await request_tracker.get_requests(limit)
        return {
            "total": len(requests),
            "requests": requests
        }
    except Exception as e:
        logger.error(f"Error retrieving requests: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving request history")

@app.post("/api/requests/clear")
async def clear_requests():
    """Clear all tracked requests."""
    logger.info("Clearing request history")
    try:
        await request_tracker.clear_requests()
        return {"message": "Request history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing requests: {e}")
        raise HTTPException(status_code=500, detail="Error clearing request history")

@app.post("/api/test-endpoint")
async def test_forward_endpoint(request: dict):
    """
    Test a forward endpoint by sending a simple test request.

    Args:
        request: Dictionary containing 'endpoint' key with the URL to test

    Returns:
        ForwardingResponse with the test result
    """
    logger.info("Testing forward endpoint")
    
    try:
        endpoint = request.get("endpoint")
        if not endpoint:
            raise HTTPException(status_code=400, detail="Endpoint URL is required")
        
        # Basic URL validation
        try:
            from urllib.parse import urlparse
            parsed = urlparse(endpoint)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL format")
            if parsed.scheme not in ['http', 'https']:
                raise ValueError("URL must use HTTP or HTTPS protocol")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid endpoint URL: {str(e)}")

        # Send a test request
        test_result = await forward_request_to_endpoint(
            content="This is a test message from LLM Firewall API",
            endpoint_url=endpoint,
            is_safe=True
        )

        logger.info(f"Forward endpoint test completed: {test_result.success}")
        return test_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing forward endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error testing forward endpoint")

@app.get("/config")
async def get_config():
    """Get the current scanner configuration."""
    logger.debug("Config requested")

    # Build scanners config safely
    scanners_config = {}
    if llamafirewall is not None:
        scanners_config = {
            role.name if hasattr(role, 'name') else str(role): [
                getattr(scanner, 'name', str(scanner)) for scanner in scanner_list
            ]
            for role, scanner_list in llamafirewall.scanners.items()
        }

    config_response = {
        "llamafirewall": {
            "scanners": scanners_config
        },
        "llmguard": {
            "enabled": LLMGUARD_ENABLED,
            "scanners": [type(scanner).__name__ for scanner in LLMGUARD_SCANNERS] if LLMGUARD_ENABLED else [],
            "config": LLMGUARD_CONFIG if LLMGUARD_ENABLED else {}
        },
        "openai_moderation": {
            "enabled": moderation
        }
    }

    # Add MODERATION to the list if enabled
    if moderation:
        for role in config_response["llamafirewall"]["scanners"]:
            if "MODERATION" not in config_response["llamafirewall"]["scanners"][role]:
                config_response["llamafirewall"]["scanners"][role].append("MODERATION")

    return config_response

# Endpoint Configuration Management API Endpoints

@app.post("/api/configurations", status_code=201)
async def create_configuration(request: CreateConfigurationRequest):
    """
    Create a new configuration.

    Args:
        request: Configuration creation request

    Returns:
        Dictionary with success status and configuration ID
    """
    try:
        config_manager = await get_config_manager()

        # Special handling for endpoint configurations
        if request.config_type == "endpoint":
            # Validate endpoint configuration
            endpoint_config = EndpointConfiguration(**request.config_data)
            config_id = await config_manager.create_endpoint_configuration(
                name=request.name,
                endpoint_config=endpoint_config,
                description=request.description,
                tags=request.tags
            )
        else:
            # General configuration
            config_id = await config_manager.create_configuration(
                name=request.name,
                config_type=request.config_type,
                config_data=request.config_data,
                description=request.description,
                tags=request.tags
            )

        logger.info(f"Created configuration: {request.name} (ID: {config_id})")
        return {
            "success": True,
            "message": "Configuration created successfully",
            "config_id": config_id
        }
        
    except ValueError as e:
        logger.error(f"Validation error creating configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error creating configuration")

@app.get("/api/configurations")
async def list_configurations(
    config_type: Optional[str] = None,
    active_only: bool = True,
    tags: Optional[str] = None
):
    """
    List configurations with optional filtering.

    Args:
        config_type: Filter by configuration type
        active_only: Only return active configurations
        tags: Comma-separated list of tags to filter by

    Returns:
        List of configuration records
    """
    try:
        config_manager = await get_config_manager()

        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        configurations = await config_manager.list_configurations(
            config_type=config_type,
            active_only=active_only,
            tags=tag_list
        )

        return {
            "success": True,
            "configurations": [config.model_dump() for config in configurations]
        }
        
    except Exception as e:
        logger.error(f"Error listing configurations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing configurations")

@app.get("/api/configurations/{config_id}")
async def get_configuration(config_id: str):
    """
    Get a configuration by ID.

    Args:
        config_id: Configuration ID

    Returns:
        Configuration record
    """
    try:
        config_manager = await get_config_manager()
        configuration = await config_manager.get_configuration(config_id)

        if not configuration:
            raise HTTPException(status_code=404, detail="Configuration not found")

        return {
            "success": True,
            "configuration": configuration.model_dump()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving configuration")

@app.get("/api/configurations/by-name/{config_name}")
async def get_configuration_by_name(config_name: str, config_type: Optional[str] = None):
    """
    Get a configuration by name.

    Args:
        config_name: Configuration name
        config_type: Optional configuration type filter

    Returns:
        Configuration record
    """
    try:
        config_manager = await get_config_manager()
        configuration = await config_manager.get_configuration_by_name(config_name, config_type)
        
        if not configuration:
            raise HTTPException(status_code=404, detail="Configuration not found")

        return {
            "success": True,
            "configuration": configuration.model_dump()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving configuration by name: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving configuration")

@app.put("/api/configurations/{config_id}")
async def update_configuration(config_id: str, request: UpdateConfigurationRequest):
    """
    Update a configuration.

    Args:
        config_id: Configuration ID
        request: Update request data

    Returns:
        Success status
    """
    try:
        config_manager = await get_config_manager()

        # Prepare update data
        updates = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.description is not None:
            updates["description"] = request.description
        if request.config_data is not None:
            updates["config_data"] = request.config_data
        if request.is_active is not None:
            updates["is_active"] = request.is_active
        if request.tags is not None:
            updates["tags"] = request.tags

        success = await config_manager.update_configuration(config_id, updates)

        if not success:
            raise HTTPException(status_code=404, detail="Configuration not found")

        logger.info(f"Updated configuration: {config_id}")
        return {
            "success": True,
            "message": "Configuration updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating configuration")

@app.delete("/api/configurations/{config_id}")
async def delete_configuration(config_id: str):
    """
    Delete a configuration (soft delete).

    Args:
        config_id: Configuration ID

    Returns:
        Success status
    """
    try:
        config_manager = await get_config_manager()
        success = await config_manager.delete_configuration(config_id)

        if not success:
            raise HTTPException(status_code=404, detail="Configuration not found")

        logger.info(f"Deleted configuration: {config_id}")
        return {
            "success": True,
            "message": "Configuration deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting configuration")

@app.get("/api/endpoint-configurations")
async def list_endpoint_configurations():
    """
    List all endpoint configurations.

    Returns:
        List of endpoint configuration records
    """
    try:
        config_manager = await get_config_manager()
        configurations = await config_manager.list_configurations(
            config_type="endpoint",
            active_only=True
        )

        # Parse endpoint configurations
        endpoint_configs = []
        for config in configurations:
            try:
                endpoint_config = EndpointConfiguration(**config.config_data)
                endpoint_configs.append({
                    "id": config.id,
                    "name": config.name,
                    "description": config.description,
                    "created_at": config.created_at.isoformat(),
                    "updated_at": config.updated_at.isoformat(),
                    "tags": config.tags,
                    "endpoint_config": endpoint_config.model_dump()
                })
            except Exception as e:
                logger.warning(f"Failed to parse endpoint configuration {config.name}: {e}")
                continue
        
        return {
            "success": True,
            "endpoint_configurations": endpoint_configs
        }

    except Exception as e:
        logger.error(f"Error listing endpoint configurations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing endpoint configurations")

@app.post("/api/endpoint-configurations/test/{config_name}")
async def test_endpoint_configuration(config_name: str):
    """
    Test an endpoint configuration by sending a test request.

    Args:
        config_name: Name of the endpoint configuration to test

    Returns:
        Test result
    """
    try:
        config_manager = await get_config_manager()
        endpoint_config = await config_manager.get_endpoint_configuration(config_name)
        
        if not endpoint_config:
            raise HTTPException(status_code=404, detail="Endpoint configuration not found")
        
        # Send test request
        test_result = await forward_request_to_endpoint(
            content="This is a test message from LLM Firewall API",
            endpoint_config=endpoint_config,
            is_safe=True
        )

        logger.info(f"Endpoint configuration test completed: {config_name}, success: {test_result.success}")
        return {
            "success": True,
            "test_result": test_result.model_dump(),
            "configuration_name": config_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing endpoint configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error testing endpoint configuration")


# Endpoint-specific configuration routes for the web UI
@app.post("/api/configurations/endpoints")
async def create_endpoint_configuration_web(request: dict):
    """
    Create a new endpoint configuration (Web UI specific route).

    Args:
        request: Dictionary containing name, description, and endpoint_config

    Returns:
        Configuration details
    """
    try:
        config_manager = await get_config_manager()

        name = request.get("name")
        description = request.get("description")
        endpoint_config_data = request.get("endpoint_config", {})

        if not name:
            raise HTTPException(status_code=400, detail="Configuration name is required")

        # Validate endpoint configuration
        endpoint_config = EndpointConfiguration(**endpoint_config_data)

        config_id = await config_manager.create_endpoint_configuration(
            name=name,
            endpoint_config=endpoint_config,
            description=description
        )

        logger.info(f"Created endpoint configuration via web UI: {name}")

        return {
            "id": config_id,
            "name": name,
            "message": "Endpoint configuration created successfully"
        }

    except ValidationError as e:
        logger.error(f"Validation error creating endpoint configuration: {e}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating endpoint configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error creating endpoint configuration")

@app.get("/api/configurations/endpoints/{config_name}")
async def get_endpoint_configuration_web(config_name: str):
    """
    Get an endpoint configuration by name (Web UI specific route).

    Args:
        config_name: Name of the configuration to retrieve

    Returns:
        Endpoint configuration data
    """
    try:
        config_manager = await get_config_manager()

        endpoint_config = await config_manager.get_endpoint_configuration(config_name)

        if not endpoint_config:
            raise HTTPException(status_code=404, detail="Endpoint configuration not found")

        return endpoint_config.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving endpoint configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving endpoint configuration")

@app.put("/api/configurations/endpoints/{config_name}")
async def update_endpoint_configuration_web(config_name: str, request: dict):
    """
    Update an endpoint configuration (Web UI specific route).

    Args:
        config_name: Name of the configuration to update
        request: Dictionary containing description and endpoint_config

    Returns:
        Updated configuration details
    """
    try:
        config_manager = await get_config_manager()
        
        description = request.get("description")
        endpoint_config_data = request.get("endpoint_config", {})
        
        # Validate endpoint configuration
        endpoint_config = EndpointConfiguration(**endpoint_config_data)
        
        success = await config_manager.update_endpoint_configuration(
            name=config_name,
            endpoint_config=endpoint_config,
            description=description
        )

        if not success:
            raise HTTPException(status_code=404, detail="Endpoint configuration not found")
        
        logger.info(f"Updated endpoint configuration via web UI: {config_name}")
        
        return {
            "name": config_name,
            "message": "Endpoint configuration updated successfully"
        }

    except ValidationError as e:
        logger.error(f"Validation error updating endpoint configuration: {e}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating endpoint configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating endpoint configuration")

@app.delete("/api/configurations/endpoints/{config_name}")
async def delete_endpoint_configuration_web(config_name: str):
    """
    Delete an endpoint configuration (Web UI specific route).

    Args:
        config_name: Name of the configuration to delete

    Returns:
        Deletion confirmation
    """
    try:
        config_manager = await get_config_manager()

        success = await config_manager.delete_endpoint_configuration(config_name)

        if not success:
            raise HTTPException(status_code=404, detail="Endpoint configuration not found")

        logger.info(f"Deleted endpoint configuration via web UI: {config_name}")

        return {
            "message": f"Endpoint configuration '{config_name}' deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting endpoint configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting endpoint configuration")