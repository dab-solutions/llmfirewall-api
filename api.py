from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
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
from pydantic import BaseModel, Field, field_validator
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
import aiohttp
from urllib.parse import urlparse

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
    yield
    # Shutdown
    logger.info("Shutting down application")
    thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")

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
    forward_endpoint: Optional[str] = Field(None, description="Optional endpoint to forward the request to after scanning")

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
    
    @field_validator('forward_endpoint')
    @classmethod
    def validate_forward_endpoint(cls, v: Optional[str]) -> Optional[str]:
        """Validate forward endpoint URL for security."""
        if v is None:
            return v
        
        # Ensure the URL is valid and uses secure protocols
        try:
            parsed = urlparse(v)
            if not parsed.scheme:
                raise ValueError("Forward endpoint must include a protocol (http:// or https://)")
            
            if parsed.scheme not in ['http', 'https']:
                raise ValueError("Forward endpoint must use HTTP or HTTPS protocol")
            
            if not parsed.netloc:
                raise ValueError("Forward endpoint must include a valid host")
            
            # Prevent SSRF attacks by blocking private IP ranges and localhost
            hostname = parsed.hostname
            if hostname:
                # Block localhost variants
                #if hostname.lower() in ['localhost', '127.0.0.1', '::1']:
                #    raise ValueError("Forward endpoint cannot point to localhost")
                
                # Block common private IP ranges (this is a basic check)
                if (hostname.startswith('192.168.') or 
                    hostname.startswith('10.') or 
                    hostname.startswith('172.16.') or
                    hostname.startswith('172.17.') or
                    hostname.startswith('172.18.') or
                    hostname.startswith('172.19.') or
                    hostname.startswith('172.20.') or
                    hostname.startswith('172.21.') or
                    hostname.startswith('172.22.') or
                    hostname.startswith('172.23.') or
                    hostname.startswith('172.24.') or
                    hostname.startswith('172.25.') or
                    hostname.startswith('172.26.') or
                    hostname.startswith('172.27.') or
                    hostname.startswith('172.28.') or
                    hostname.startswith('172.29.') or
                    hostname.startswith('172.30.') or
                    hostname.startswith('172.31.')):
                    raise ValueError("Forward endpoint cannot point to private IP addresses")
                    
            return v.strip()
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Invalid forward endpoint URL: {str(e)}")

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

async def forward_request_to_endpoint(content: str, endpoint: str, is_safe: bool) -> ForwardingResponse:
    """
    Forward the scanned content to the specified endpoint.
    
    Args:
        content: The original content that was scanned
        endpoint: The target endpoint URL to forward to
        is_safe: Whether the content was deemed safe by scanning
        
    Returns:
        ForwardingResponse containing the result of the forwarding operation
    """
    try:
        logger.info(f"Forwarding request to endpoint: {endpoint}")
        
        # Prepare the payload to send
        payload = {
            "content": content,
            "is_safe": is_safe,
            "timestamp": datetime.now().isoformat(),
            "source": "llama_firewall_api"
        }
        
        # Set up HTTP client with security configurations
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        connector = aiohttp.TCPConnector(
            limit=10,  # Limit concurrent connections
            ssl=True   # Verify SSL certificates
        )
        
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "LlamaFirewall-API/1.1.0"
            }
        ) as session:
            async with session.post(endpoint, json=payload) as response:
                # Read response content
                try:
                    response_data = await response.json()
                except aiohttp.ContentTypeError:
                    # If response is not JSON, read as text
                    response_text = await response.text()
                    response_data = {"response": response_text}
                
                success = 200 <= response.status < 300
                
                if success:
                    logger.info(f"Successfully forwarded request to {endpoint} (status: {response.status})")
                else:
                    logger.warning(f"Forwarding request failed with status {response.status}: {response_data}")
                
                return ForwardingResponse(
                    success=success,
                    status_code=response.status,
                    response_data=response_data,
                    error=None if success else f"HTTP {response.status}: {response_data}"
                )
                
    except aiohttp.ClientError as e:
        error_msg = f"Network error while forwarding to {endpoint}: {str(e)}"
        logger.error(error_msg)
        return ForwardingResponse(
            success=False,
            status_code=None,
            response_data=None,
            error=error_msg
        )
    except asyncio.TimeoutError:
        error_msg = f"Timeout while forwarding to {endpoint}"
        logger.error(error_msg)
        return ForwardingResponse(
            success=False,
            status_code=None,
            response_data=None,
            error=error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error while forwarding to {endpoint}: {str(e)}"
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
            llama_result = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: llamafirewall.scan(message)
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

        # Step 4: Forward request if endpoint is provided
        if request.forward_endpoint:
            try:
                logger.debug(f"Forwarding request to: {request.forward_endpoint}")
                forwarding_result = await forward_request_to_endpoint(
                    content=request.content,
                    endpoint=request.forward_endpoint,
                    is_safe=response.is_safe
                )
                
                # Update response with forwarding result
                response = ScanResponse(
                    is_safe=response.is_safe,
                    risk_score=response.risk_score,
                    details=response.details,
                    moderation_results=response.moderation_results,
                    llmguard_results=response.llmguard_results,
                    forwarding_result=forwarding_result,
                    scan_type=f"{response.scan_type}+forwarded"
                )
                
                logger.info("Request forwarding completed", extra=sanitize_log_data({
                    "endpoint": request.forward_endpoint,
                    "success": forwarding_result.success,
                    "status_code": forwarding_result.status_code
                }))
                
            except Exception as e:
                logger.error("Forwarding failed", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
                # Don't fail the entire request if forwarding fails
                response = ScanResponse(
                    is_safe=response.is_safe,
                    risk_score=response.risk_score,
                    details=response.details,
                    moderation_results=response.moderation_results,
                    llmguard_results=response.llmguard_results,
                    forwarding_result=ForwardingResponse(
                        success=False,
                        status_code=None,
                        response_data=None,
                        error=f"Forwarding error: {str(e)}"
                    ),
                    scan_type=f"{response.scan_type}+forwarding_failed"
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
                scan_results["forwarding"] = {
                    "enabled": True,
                    "endpoint": request.forward_endpoint,
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
            message="Configuration updated successfully. Restart the application to apply changes.",
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
        
        # Validate the endpoint using the same validation as ScanRequest
        try:
            # Use a temporary ScanRequest to validate the endpoint
            temp_request = ScanRequest(content="test", forward_endpoint=endpoint)
            validated_endpoint = temp_request.forward_endpoint
            if not validated_endpoint:
                raise ValueError("Endpoint validation failed")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Send a test request
        test_result = await forward_request_to_endpoint(
            content="This is a test message from LLM Firewall API",
            endpoint=validated_endpoint,
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
    config_response = {
        "llamafirewall": {
            "scanners": {
                role.name if hasattr(role, 'name') else str(role): [
                    getattr(scanner, 'name', str(scanner)) for scanner in scanner_list
                ]
                for role, scanner_list in llamafirewall.scanners.items()
            }
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

    logger.debug(f"Returning config: {config_response}")
    return config_response