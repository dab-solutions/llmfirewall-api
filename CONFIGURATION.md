# Configuration Management System

The LLM Firewall API now includes a powerful configuration management system that allows you to store and manage endpoint configurations, HTTP headers, and other settings in an external database.

## Features

- **Database-backed Configuration**: Store configurations in SQLite (easily extensible to other databases)
- **Endpoint Management**: Configure forwarding endpoints with custom headers, timeouts, and security settings
- **Flexible Parameters**: Support for custom HTTP methods, SSL verification, retry logic, and conditional forwarding
- **RESTful API**: Full CRUD operations for configuration management
- **Security**: Built-in validation and SSRF protection
- **Future-ready**: Extensible schema for additional configuration types

## Quick Start

### 1. Basic Usage with Stored Configuration

```python
import aiohttp
import asyncio

async def create_and_use_config():
    # Create an endpoint configuration
    config_data = {
        "name": "my-webhook",
        "description": "My custom webhook endpoint",
        "config_type": "endpoint",
        "config_data": {
            "url": "https://my-api.example.com/webhook",
            "headers": {
                "Authorization": "Bearer your-token",
                "X-Source": "llm-firewall"
            },
            "timeout": 30,
            "method": "POST",
            "verify_ssl": True,
            "include_scan_results": True,
            "forward_on_unsafe": False
        },
        "tags": ["webhook", "production"]
    }
    
    # Create the configuration
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/configurations",
            json=config_data
        ) as response:
            result = await response.json()
            print(f"Configuration created: {result['config_id']}")
    
    # Use the configuration in a scan request
    scan_request = {
        "content": "Hello, this is a test message.",
        "endpoint_config_name": "my-webhook",
        "additional_headers": {
            "X-Request-ID": "12345"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/scan",
            json=scan_request
        ) as response:
            result = await response.json()
            print(f"Scan completed: {result}")

asyncio.run(create_and_use_config())
```

### 2. Scanning with Direct Endpoint (Legacy Support)

```python
# You can still use direct endpoints as before
scan_request = {
    "content": "Test message",
    "forward_endpoint": "https://webhook.site/your-id",
    "additional_headers": {
        "X-Custom": "value"
    }
}
```

## Configuration Schema

### Endpoint Configuration

```json
{
  "url": "https://api.example.com/webhook",
  "headers": {
    "Authorization": "Bearer token",
    "Content-Type": "application/json"
  },
  "timeout": 30,
  "retry_attempts": 3,
  "retry_delay": 1.0,
  "verify_ssl": true,
  "method": "POST",
  "include_scan_results": true,
  "forward_on_unsafe": false
}
```

### Configuration Fields

- **url** (required): Target endpoint URL
- **headers** (optional): Additional HTTP headers to include
- **timeout** (optional, default: 30): Request timeout in seconds
- **retry_attempts** (optional, default: 3): Number of retry attempts
- **retry_delay** (optional, default: 1.0): Delay between retries in seconds
- **verify_ssl** (optional, default: true): Whether to verify SSL certificates
- **method** (optional, default: "POST"): HTTP method to use
- **include_scan_results** (optional, default: true): Include scan results in payload
- **forward_on_unsafe** (optional, default: false): Forward even if content is unsafe

## API Endpoints

### Configuration Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/configurations` | Create a new configuration |
| GET | `/api/configurations` | List all configurations |
| GET | `/api/configurations/{id}` | Get configuration by ID |
| GET | `/api/configurations/by-name/{name}` | Get configuration by name |
| PUT | `/api/configurations/{id}` | Update a configuration |
| DELETE | `/api/configurations/{id}` | Delete a configuration |

### Endpoint-Specific Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/endpoint-configurations` | List endpoint configurations |
| POST | `/api/endpoint-configurations/test/{name}` | Test an endpoint configuration |

### Enhanced Scan Endpoint

The `/scan` endpoint now supports these additional parameters:

```json
{
  "content": "Message to scan",
  "forward_endpoint": "https://direct-url.com",
  "endpoint_config_name": "stored-config-name",
  "additional_headers": {
    "X-Custom": "value"
  }
}
```

## Security Features

### SSRF Protection

The system includes built-in protection against Server-Side Request Forgery (SSRF) attacks:

- URL validation and parsing
- Blocking of private IP ranges
- Protocol restrictions (HTTP/HTTPS only)
- Host validation

### Header Security

- Automatic filtering of potentially dangerous headers
- Prevention of header injection attacks
- Safe header merging with request-specific headers

### Input Validation

- Comprehensive validation of all configuration parameters
- JSON schema validation for complex fields
- Size limits to prevent DoS attacks

## Database Storage

The system uses SQLite by default but can be easily configured for other databases:

```python
# Default SQLite configuration
DATABASE_URL = "sqlite+aiosqlite:///config.db"

# PostgreSQL example
DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/llm_firewall"

# MySQL example  
DATABASE_URL = "mysql+aiomysql://user:pass@localhost/llm_firewall"
```

Set the `CONFIG_DB_PATH` environment variable to customize the SQLite database location:

```bash
export CONFIG_DB_PATH="/path/to/your/config.db"
```

## Example Configurations

### 1. Secure API Webhook

```json
{
  "name": "secure-api",
  "description": "Production API webhook with authentication",
  "config_type": "endpoint",
  "config_data": {
    "url": "https://api.production.com/webhooks/firewall",
    "headers": {
      "Authorization": "Bearer prod-token-12345",
      "X-API-Version": "v1",
      "X-Source": "llm-firewall"
    },
    "timeout": 15,
    "verify_ssl": true,
    "forward_on_unsafe": false,
    "include_scan_results": true
  },
  "tags": ["production", "secure"]
}
```

### 2. Logging and Monitoring

```json
{
  "name": "audit-logger",
  "description": "Audit logging endpoint for all scan results",
  "config_type": "endpoint", 
  "config_data": {
    "url": "https://logs.company.com/api/events",
    "headers": {
      "X-Log-Level": "INFO",
      "X-Component": "llm-firewall"
    },
    "timeout": 10,
    "forward_on_unsafe": true,
    "include_scan_results": true
  },
  "tags": ["logging", "audit"]
}
```

### 3. Slack Notifications

```json
{
  "name": "slack-alerts",
  "description": "Slack webhook for security alerts",
  "config_type": "endpoint",
  "config_data": {
    "url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
    "timeout": 5,
    "forward_on_unsafe": true,
    "include_scan_results": false
  },
  "tags": ["notification", "slack"]
}
```

## Migration Guide

### From Direct Endpoints

If you're currently using direct endpoints in your scan requests:

**Before:**
```json
{
  "content": "test message",
  "forward_endpoint": "https://webhook.site/unique-id"
}
```

**After (Option 1 - Still Supported):**
```json
{
  "content": "test message", 
  "forward_endpoint": "https://webhook.site/unique-id",
  "additional_headers": {
    "X-Custom": "header"
  }
}
```

**After (Option 2 - Recommended):**

1. Create a configuration:
```bash
curl -X POST http://localhost:8000/api/configurations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-webhook",
    "config_type": "endpoint",
    "config_data": {
      "url": "https://webhook.site/unique-id",
      "headers": {"X-Custom": "header"}
    }
  }'
```

2. Use the configuration:
```json
{
  "content": "test message",
  "endpoint_config_name": "my-webhook"
}
```

## Best Practices

1. **Use Stored Configurations**: Prefer stored configurations over direct endpoints for better security and maintainability

2. **Tag Your Configurations**: Use tags to organize configurations by environment, purpose, or team

3. **Test Configurations**: Always test new configurations before using them in production

4. **Monitor Forwarding**: Check forwarding results in scan responses to ensure successful delivery

5. **Secure Headers**: Store sensitive headers (like API keys) in the database rather than passing them in each request

6. **Environment Separation**: Use different configuration names/tags for different environments (dev, staging, prod)

7. **Regular Cleanup**: Periodically review and clean up unused configurations

## Troubleshooting

### Common Issues

1. **Configuration Not Found**
   - Check the configuration name spelling
   - Verify the configuration is active
   - Use the list endpoint to see available configurations

2. **Forwarding Failures**
   - Test the configuration with the test endpoint
   - Check URL accessibility and SSL certificate validity
   - Verify authentication headers and permissions

3. **Database Issues**
   - Check database file permissions for SQLite
   - Verify database connection string for other databases
   - Check application logs for detailed error messages

### Debug Tips

1. Enable debug logging: `LOG_LEVEL=DEBUG`
2. Use the test endpoint to validate configurations
3. Check the forwarding_result in scan responses
4. Monitor application logs for detailed error information

## Future Enhancements

The configuration system is designed to be extensible. Planned features include:

- Configuration versioning and rollback
- Configuration templates
- Bulk operations
- Configuration import/export
- Advanced filtering and search
- Configuration change notifications
- Role-based access control
