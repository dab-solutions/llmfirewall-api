# Configuration Management System

The LLM Firewall API includes a comprehensive configuration management system that allows you to store and manage endpoint configurations, HTTP headers, and other settings in a local database. This system provides a secure, scalable way to manage multiple endpoint configurations for automatic request forwarding.

## Features

- **Database-backed Configuration**: Store configurations in SQLite (easily extensible to other databases)
- **Endpoint Management**: Configure forwarding endpoints with custom headers, timeouts, and security settings
- **Flexible Parameters**: Support for custom HTTP methods, SSL verification, and conditional forwarding
- **RESTful API**: Full CRUD operations for configuration management
- **Web Interface**: User-friendly configuration management through the Forwarding tab
- **Security**: Built-in validation and SSRF protection
- **Automatic Forwarding**: Enable automatic forwarding to configured endpoints
- **Testing**: Built-in endpoint testing and validation

## Quick Start

### 1. Create an Endpoint Configuration

**Via Web Interface:**
1. Navigate to http://localhost:8000
2. Click the **Forwarding** tab
3. Click "Create New Configuration"
4. Fill in the configuration details
5. Enable "Enable automatic forwarding for scan requests"
6. Save the configuration

**Via API:**
```bash
curl -X POST "http://localhost:8000/api/configurations" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-webhook",
    "description": "My webhook endpoint",
    "config_type": "endpoint",
    "config_data": {
      "url": "https://webhook.site/your-unique-id",
      "headers": {
        "Authorization": "Bearer your-token",
        "X-Source": "llm-firewall"
      },
      "timeout": 30,
      "forwarding_enabled": true,
      "forward_on_unsafe": false,
      "include_scan_results": true
    },
    "tags": ["webhook", "production"]
  }'
```

### 2. Test the Configuration

```bash
curl -X POST "http://localhost:8000/api/endpoint-configurations/test/my-webhook"
```

### 3. Scan Content (Automatic Forwarding)

Once you have configurations with `forwarding_enabled=true`, all scan requests are automatically forwarded:

```bash
curl -X POST "http://localhost:8000/scan" \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, this is a test message."}'
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
  "forward_on_unsafe": false,
  "forwarding_enabled": false
}
```

### Configuration Fields

- **url** (required): Target endpoint URL
- **headers** (optional): Additional HTTP headers to include
- **timeout** (optional, default: 30): Request timeout in seconds (1-300)
- **retry_attempts** (optional, default: 3): Number of retry attempts
- **retry_delay** (optional, default: 1.0): Delay between retries in seconds
- **verify_ssl** (optional, default: true): Whether to verify SSL certificates
- **method** (optional, default: "POST"): HTTP method (GET, POST, PUT, PATCH, DELETE)
- **include_scan_results** (optional, default: true): Include detailed scan results in payload
- **forward_on_unsafe** (optional, default: false): Forward even if content is unsafe
- **forwarding_enabled** (optional, default: false): **Enable automatic forwarding for all scan requests**

### Configuration Record Fields

- **name** (required): Unique identifier for the configuration
- **description** (optional): Human-readable description
- **config_type** (required): Type of configuration ("endpoint")
- **config_data** (required): The endpoint configuration object
- **tags** (optional): Array of tags for organization and filtering
- **is_active** (optional, default: true): Whether the configuration is active

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

The `/scan` endpoint automatically forwards requests to all endpoint configurations with `forwarding_enabled=true`:

```json
{
  "content": "Message to scan"
}
```

**Automatic Forwarding Behavior:**
- All configurations with `forwarding_enabled=true` receive the scan request
- Forwarding occurs after the security scan is completed
- Multiple endpoints can receive the same request
- Forwarding failures do not affect the scan results
- The response includes forwarding results for monitoring

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
    "forwarding_enabled": true,
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
    "forwarding_enabled": true,
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
    "forwarding_enabled": true,
    "forward_on_unsafe": true,
    "include_scan_results": false
  },
  "tags": ["notification", "slack"]
}
```

### 4. Development Testing

```json
{
  "name": "webhook-test",
  "description": "Development testing endpoint",
  "config_type": "endpoint",
  "config_data": {
    "url": "https://webhook.site/your-unique-id",
    "headers": {
      "X-Environment": "development"
    },
    "timeout": 30,
    "forwarding_enabled": false,
    "forward_on_unsafe": true,
    "include_scan_results": true,
    "verify_ssl": true
  },
  "tags": ["development", "testing"]
}
```

## Migration Guide

### Current Implementation

The current implementation uses **automatic forwarding** based on endpoint configurations. Here's how to set it up:

**Step 1: Create Endpoint Configuration**
```bash
curl -X POST http://localhost:8000/api/configurations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-webhook",
    "config_type": "endpoint",
    "config_data": {
      "url": "https://webhook.site/unique-id",
      "headers": {"X-Custom": "header"},
      "forwarding_enabled": true
    }
  }'
```

**Step 2: Use Normal Scan Endpoint**
```bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"content": "test message"}'
```

The request will automatically be forwarded to all endpoints with `forwarding_enabled=true`.

### For Multiple Endpoints

Create multiple configurations with `forwarding_enabled=true` to forward to multiple endpoints:

```bash
# Create first endpoint
curl -X POST http://localhost:8000/api/configurations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "production-webhook",
    "config_type": "endpoint",
    "config_data": {
      "url": "https://prod.example.com/webhook",
      "forwarding_enabled": true,
      "forward_on_unsafe": false
    }
  }'

# Create second endpoint
curl -X POST http://localhost:8000/api/configurations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "audit-logger",
    "config_type": "endpoint",
    "config_data": {
      "url": "https://logs.example.com/api/events",
      "forwarding_enabled": true,
      "forward_on_unsafe": true
    }
  }'
```

Now all scan requests will be forwarded to both endpoints automatically.

## Best Practices

1. **Use Automatic Forwarding**: Set `forwarding_enabled=true` for endpoints that should receive all scan requests

2. **Configure Safety Filtering**: Use `forward_on_unsafe=false` to prevent forwarding of potentially harmful content to production systems

3. **Tag Your Configurations**: Use tags to organize configurations by environment, purpose, or team

4. **Test Configurations**: Always test new configurations before enabling forwarding:
   ```bash
   curl -X POST "http://localhost:8000/api/endpoint-configurations/test/config-name"
   ```

5. **Monitor Forwarding**: Check forwarding results in scan responses and the monitoring dashboard

6. **Secure Headers**: Store sensitive headers (like API keys) in the configuration rather than passing them per request

7. **Environment Separation**: Use different configuration names and tags for different environments

8. **Regular Cleanup**: Periodically review and clean up unused configurations

9. **Use HTTPS**: Always use HTTPS endpoints for production configurations

10. **Configure Timeouts**: Set appropriate timeouts based on your endpoint's expected response time

## Troubleshooting

### Common Issues

1. **Configuration Not Found**
   - Check the configuration name spelling
   - Verify the configuration exists: `GET /api/configurations`
   - Ensure the configuration is active (`is_active=true`)

2. **Forwarding Not Occurring**
   - Verify `forwarding_enabled=true` in the endpoint configuration
   - Check that the configuration is active
   - Use the monitoring dashboard to check forwarding status

3. **Forwarding Failures**
   - Test the configuration: `POST /api/endpoint-configurations/test/{name}`
   - Check URL accessibility and SSL certificate validity
   - Verify authentication headers and permissions
   - Check the endpoint's response time (must be under configured timeout)

4. **Multiple Endpoints Receiving Requests**
   - This is expected behavior - all endpoints with `forwarding_enabled=true` receive requests
   - Disable forwarding (`forwarding_enabled=false`) on endpoints that shouldn't receive all requests

5. **Database Issues**
   - Check database file permissions for SQLite
   - Verify the database path is writable
   - Check application logs for detailed error messages

6. **SSL/HTTPS Issues**
   - Ensure valid SSL certificates for HTTPS endpoints
   - Set `verify_ssl=false` for testing (not recommended for production)
   - Check for certificate chain issues

### Debug Tips

1. **Enable Debug Logging**: Set `LOG_LEVEL=DEBUG` for verbose configuration and forwarding information

2. **Use Test Endpoints**: Always test configurations before enabling forwarding:
   ```bash
   curl -X POST "http://localhost:8000/api/endpoint-configurations/test/my-config"
   ```

3. **Check Monitoring Dashboard**: Use the monitoring tab in the web interface to see forwarding results

4. **Review Configuration**: Get configuration details to verify settings:
   ```bash
   curl "http://localhost:8000/api/configurations/by-name/my-config"
   ```

5. **Monitor Application Logs**: Check logs for detailed error information and SSRF protection alerts

### Configuration Validation

The system performs extensive validation on configurations:
- **URL Validation**: Ensures proper format and protocol
- **SSRF Protection**: Blocks private IP addresses (with warnings in development)
- **Header Validation**: Prevents dangerous header injection
- **Timeout Limits**: Enforces reasonable timeout values (1-300 seconds)
- **Method Validation**: Only allows safe HTTP methods

## Future Enhancements

The configuration system is designed to be extensible. Planned and potential features include:

- **Configuration versioning and rollback**
- **Configuration templates and inheritance**
- **Bulk operations and batch management**
- **Configuration import/export functionality**
- **Advanced filtering and search capabilities**
- **Configuration change notifications and webhooks**
- **Role-based access control (RBAC)**
- **Configuration approval workflows**
- **Performance analytics and optimization**
- **Integration with external configuration management systems**

The current implementation provides a solid foundation for these advanced features while maintaining simplicity and reliability for everyday use.
