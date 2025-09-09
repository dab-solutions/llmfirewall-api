# Request Forwarding Feature

## Overview

The LLM Firewall API now supports automatic request forwarding to external endpoints. This feature allows you to integrate the firewall with downstream services by automatically sending scanned content and security assessments to configured endpoints.

## How It Works

1. **Scanning First**: The firewall performs its complete security scan (LlamaFirewall, LLM Guard, OpenAI Moderation) as usual
2. **Forwarding After**: If a forward endpoint is specified, the API sends the original content and security assessment to the target URL
3. **Non-Blocking**: Forwarding failures do not affect the security scan results

## API Usage

### Scan with Forwarding

```bash
curl -X POST "http://localhost:8000/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your message to scan",
    "forward_endpoint": "https://api.example.com/webhook"
  }'
```

### Response Format

The response includes a new `forwarding_result` field:

```json
{
  "is_safe": true,
  "risk_score": 0.1,
  "details": {...},
  "scan_type": "llamafirewall+forwarded",
  "forwarding_result": {
    "success": true,
    "status_code": 200,
    "response_data": {...},
    "error": null
  }
}
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

**Headers sent:**
- `Content-Type: application/json`
- `User-Agent: LlamaFirewall-API/1.1.0`

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

Use the new `/api/test-endpoint` endpoint to validate your configuration:

```bash
curl -X POST "http://localhost:8000/api/test-endpoint" \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint": "https://httpbin.org/post"
  }'
```

### Web UI Testing

1. Navigate to the **Forwarding** tab in the web interface
2. Enter your endpoint URL in the "Forward Endpoint URL" field
3. Click "Test Endpoint" to verify connectivity
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

1. **Always use HTTPS** for production endpoints
2. **Implement authentication** on your webhook endpoints
3. **Handle failures gracefully** - forwarding is supplementary to security scanning
4. **Monitor forwarding success rates** using the built-in monitoring
5. **Test endpoints thoroughly** before deploying to production

## Configuration

No additional environment variables are required. The forwarding feature is enabled by default and controlled per-request via the `forward_endpoint` parameter.

## Limitations

- Maximum 30-second timeout per forwarding request
- No retry mechanism for failed forwards
- Response data is limited to prevent memory issues
- Private networks and localhost are blocked for security

## Troubleshooting

### Common Issues

1. **"Forward endpoint cannot point to localhost"**
   - Solution: Use a public endpoint or configure proper network access

2. **"Forward endpoint must use HTTP or HTTPS protocol"**
   - Solution: Ensure your URL starts with `http://` or `https://`

3. **Timeout errors**
   - Solution: Ensure your endpoint responds within 30 seconds

4. **SSL certificate errors**
   - Solution: Use valid SSL certificates for HTTPS endpoints

### Debug Information

Check the API logs for detailed forwarding information:
- Request attempts and URLs
- Response status codes
- Error details and stack traces
