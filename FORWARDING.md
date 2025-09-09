# Request Forwarding Feature

## Overview

The LLM Firewall API supports automatic request forwarding to external endpoints through a configuration-based system. This feature allows you to integrate the firewall with downstream services by automatically sending scanned content and security assessments to configured endpoints.

## How It Works

1. **Configuration-Based**: Create endpoint configurations through the web UI or API that specify forwarding settings
2. **Automatic Forwarding**: When `forwarding_enabled=true` is set on an endpoint configuration, all scan requests are automatically forwarded
3. **Scanning First**: The firewall performs its complete security scan (LlamaFirewall, LLM Guard, OpenAI Moderation) as usual
4. **Forwarding After**: The API sends the original content and security assessment to configured endpoints
5. **Non-Blocking**: Forwarding failures do not affect the security scan results

## Configuration Management

### Creating Endpoint Configurations

Use the web interface or API to create endpoint configurations:

**Via Web Interface:**
1. Navigate to the **Forwarding** tab
2. Click "Create New Configuration"
3. Fill in the configuration details
4. Enable "Enable automatic forwarding for scan requests"
5. Save the configuration

**Via API:**
```bash
curl -X POST "http://localhost:8000/api/configurations" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-webhook",
    "description": "Production webhook endpoint",
    "config_type": "endpoint",
    "config_data": {
      "url": "https://api.example.com/webhook",
      "headers": {
        "Authorization": "Bearer your-token",
        "X-Source": "llm-firewall"
      },
      "timeout": 30,
      "method": "POST",
      "forwarding_enabled": true,
      "forward_on_unsafe": false,
      "include_scan_results": true,
      "verify_ssl": true
    },
    "tags": ["production", "webhook"]
  }'
```

### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | string | required | Target endpoint URL |
| `headers` | object | {} | Additional HTTP headers |
| `timeout` | integer | 30 | Request timeout in seconds |
| `method` | string | "POST" | HTTP method (POST, PUT, PATCH) |
| `forwarding_enabled` | boolean | false | Enable automatic forwarding |
| `forward_on_unsafe` | boolean | false | Forward even if content is unsafe |
| `include_scan_results` | boolean | true | Include scan results in payload |
| `verify_ssl` | boolean | true | Verify SSL certificates |

## API Usage

### Automatic Forwarding

Once you have created endpoint configurations with `forwarding_enabled=true`, all scan requests will automatically be forwarded:

```bash
curl -X POST "http://localhost:8000/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your message to scan"
  }'
```

### Response Format

The response includes a `forwarding_result` field when forwarding is active:

```json
{
  "is_safe": true,
  "risk_score": 0.1,
  "details": {...},
  "scan_type": "llamafirewall+llm_guard+forwarded",
  "forwarding_result": {
    "success": true,
    "status_code": 200,
    "response_data": {...},
    "error": null
  }
}
```

## Management APIs

### List Endpoint Configurations

```bash
curl "http://localhost:8000/api/endpoint-configurations"
```

### Get Configuration by Name

```bash
curl "http://localhost:8000/api/configurations/by-name/my-webhook"
```

### Update Configuration

```bash
curl -X PUT "http://localhost:8000/api/configurations/{config-id}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-webhook",
    "description": "Updated webhook endpoint",
    "config_data": {
      "url": "https://api.example.com/webhook",
      "forwarding_enabled": false
    }
  }'
```

### Delete Configuration

```bash
curl -X DELETE "http://localhost:8000/api/configurations/{config-id}"
```

## Forward Request Format

When forwarding is enabled, the target endpoint receives:

```json
{
  "content": "The original message content",
  "is_safe": true,
  "timestamp": "2024-01-15T10:30:00",
  "source": "llama_firewall_api"
}
```

If `include_scan_results=true` (default), additional scan details are included:

```json
{
  "content": "The original message content",
  "is_safe": true,
  "timestamp": "2024-01-15T10:30:00",
  "source": "llama_firewall_api",
  "scan_results": {
    "llamafirewall": {
      "decision": "ALLOW",
      "score": 0.1,
      "reason": "No security issues detected"
    },
    "llmguard": {
      "enabled": true,
      "is_safe": true,
      "validation_results": {
        "PromptInjection": true,
        "Toxicity": true,
        "Secrets": true
      },
      "risk_scores": {
        "PromptInjection": 0.05,
        "Toxicity": 0.02,
        "Secrets": 0.0
      }
    }
  }
}
```

**Headers sent:**
- `Content-Type: application/json`
- `User-Agent: LlamaFirewall-API/1.1.0`
- Any custom headers configured in the endpoint configuration

## Security Features

### SSRF Protection
- Only HTTP and HTTPS protocols allowed
- Localhost and private IP addresses are blocked
- URLs are validated before sending requests

### Timeouts and Limits
- 30-second request timeout
- SSL certificate verification for HTTPS
- Connection pooling with limits

### Error Handling
- Forwarding failures don't affect security scan results
- Detailed error reporting in response
- Network errors are logged but not exposed

## Testing Endpoints

### Test Endpoint Configuration

Test a stored endpoint configuration:

```bash
curl -X POST "http://localhost:8000/api/endpoint-configurations/test/my-webhook"
```

### Test Direct Endpoint

Test a direct endpoint URL for connectivity:

```bash
curl -X POST "http://localhost:8000/api/test-endpoint" \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint": "https://webhook.site/your-unique-id"
  }'
```

### Web UI Testing

1. Navigate to the **Forwarding** tab in the web interface
2. Select an existing configuration or create a new one
3. Click "Test Configuration" to verify connectivity
4. The test sends a sample message and displays the response

## Example Integration

### Webhook Server Example

Here's a simple Flask webhook server to receive forwarded requests:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def handle_firewall_webhook():
    data = request.json
    
    print(f"Received from firewall:")
    print(f"  Content: {data.get('content')}")
    print(f"  Is Safe: {data.get('is_safe')}")
    print(f"  Timestamp: {data.get('timestamp')}")
    
    # Process the data as needed
    # ...
    
    return jsonify({"status": "received", "processed": True})

if __name__ == '__main__':
    app.run(port=5000)
```

## Monitoring

Forwarding results are included in the monitoring dashboard:

- Success/failure rates for forwarding requests
- Response times and status codes
- Error details for debugging

## Best Practices

1. **Use Configuration Management**: Prefer stored configurations over direct endpoints for better security and maintainability
2. **Enable Forwarding Selectively**: Only set `forwarding_enabled=true` for endpoints that should receive all scan requests
3. **Secure Your Endpoints**: Always use HTTPS and implement proper authentication on your webhook endpoints
4. **Handle Failures Gracefully**: Forwarding is supplementary to security scanning - design your system to handle forwarding failures
5. **Monitor Forwarding Success**: Use the built-in monitoring to track forwarding success rates and response times
6. **Test Configurations**: Always test endpoint configurations before enabling automatic forwarding
7. **Use Appropriate Headers**: Configure authentication and identification headers in your endpoint configurations
8. **Consider Safety Filtering**: Use `forward_on_unsafe=false` to prevent forwarding of potentially harmful content

## Configuration Management

Endpoint configurations are stored in a SQLite database and provide:
- **Persistence**: Configurations survive server restarts
- **Security**: Built-in SSRF protection and input validation
- **Flexibility**: Support for custom headers, timeouts, and forwarding rules
- **Monitoring**: Track configuration usage and success rates
- **Versioning**: Update configurations without downtime

The configuration system is accessible through:
- **Web Interface**: User-friendly configuration management in the Forwarding tab
- **REST API**: Full CRUD operations for programmatic management
- **Database**: Direct access to the SQLite database for advanced use cases

No additional environment variables are required. The forwarding feature is enabled by default and controlled through endpoint configurations.

## Limitations

- Maximum 30-second timeout per forwarding request (configurable per endpoint)
- Response data is limited to prevent memory issues
- Private networks and localhost are blocked for security (with warnings in development)
- Only HTTP and HTTPS protocols are supported
- Configurations are stored locally in SQLite (easily extensible to other databases)

## Troubleshooting

### Common Issues

1. **"Configuration not found"**
   - Solution: Check configuration name spelling and ensure it exists

2. **"Forwarding failed"**
   - Solution: Test the endpoint configuration using the test API
   - Check endpoint accessibility and SSL certificate validity
   - Verify authentication headers and permissions

3. **"URL cannot point to private IP addresses"**
   - Solution: Use a public endpoint or configure proper network access
   - Note: In development, private IPs generate warnings but are allowed

4. **"URL must use HTTP or HTTPS protocol"**
   - Solution: Ensure your URL starts with `http://` or `https://`

5. **Timeout errors**
   - Solution: Ensure your endpoint responds within the configured timeout
   - Adjust timeout in endpoint configuration if needed

6. **SSL certificate errors**
   - Solution: Use valid SSL certificates for HTTPS endpoints
   - Set `verify_ssl=false` for testing (not recommended for production)

### Configuration Issues

1. **Forwarding not occurring**
   - Check that `forwarding_enabled=true` in your endpoint configuration
   - Verify the configuration is active and properly saved
   - Use the monitoring dashboard to check forwarding status

2. **Multiple endpoints receiving requests**
   - This is expected behavior - all endpoints with `forwarding_enabled=true` receive requests
   - Disable forwarding on endpoints that shouldn't receive all requests

3. **Headers not being sent**
   - Verify headers are properly configured in the endpoint configuration
   - Check for header conflicts with system headers

### Debug Information

Check the API logs for detailed forwarding information:
- Configuration loading and validation
- Request forwarding attempts and URLs
- Response status codes and timing
- Error details and stack traces
- SSRF protection alerts

Enable debug logging with `LOG_LEVEL=DEBUG` for verbose forwarding information.
