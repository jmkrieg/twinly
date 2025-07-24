# Langfuse Observability Integration

## Overview

This document describes the Langfuse observability integration for the Twinly API. Langfuse provides comprehensive observability for LLM applications, enabling tracing, monitoring, and analysis of agent workflows.

## Features

- **Comprehensive Tracing**: Full observability of the plan-reason-respond agent workflow
- **Metadata Capture**: Detailed request and response metadata for analysis
- **Error Tracking**: Automatic error capture and scoring
- **Performance Monitoring**: Trace timing and execution metrics
- **Graceful Fallback**: Application continues to work when Langfuse is not configured

## Configuration

### Environment Variables

Configure Langfuse using the following environment variables:

```bash
# Required for Langfuse integration
LANGFUSE_PUBLIC_KEY=pk_lf_...
LANGFUSE_SECRET_KEY=sk_lf_...

# Optional - defaults to https://cloud.langfuse.com
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional - defaults to twinly
LANGFUSE_SESSION_ID=your-session-id

# Optional - defaults to system
LANGFUSE_USER_ID=your-user-id
```

### Getting Langfuse Credentials

1. Sign up at [Langfuse Cloud](https://cloud.langfuse.com) or deploy your own instance
2. Create a new project
3. Copy the public and secret keys from the project settings
4. Set the environment variables in your deployment

## Architecture

### Components

1. **Configuration (`app/utils/langfuse_config.py`)**
   - Environment-based configuration
   - Availability checking
   - Graceful initialization

2. **Safe Decorators (`app/utils/langfuse_decorators.py`)**
   - Wrapper functions for Langfuse decorators
   - Graceful fallback when Langfuse is unavailable
   - Type-safe implementations

3. **Agent Controller Integration**
   - Full workflow tracing with `@safe_observe` decorators
   - Metadata capture for each step
   - Error handling and scoring

4. **API Integration**
   - Session and user tracking
   - Request metadata capture
   - Trace context management

### Trace Structure

The agent workflow creates a hierarchical trace structure:

```
Agent Execution (run_agent)
├── Planning Step (_simple_planner)
├── Reasoning Step (_think_step_by_step)
└── Response Generation (_generate_final_response)
```

Each trace includes:
- Input/output data
- Execution timing
- Step-specific metadata
- Error information (if applicable)
- Success/failure scoring

## Usage Examples

### Basic Agent Request

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent-mode",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": false
  }'
```

This request will create a complete trace in Langfuse showing:
1. The planning step determining the goal
2. The reasoning step with chain-of-thought
3. The final response generation
4. All intermediate LLM calls and responses

### Health Check with Observability Status

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "healthy",
  "service": "Twinly API",
  "version": "1.0.0",
  "observability": {
    "langfuse_enabled": true,
    "tracing": "enabled"
  }
}
```

## Monitoring and Analysis

### Langfuse Dashboard

Once configured, you can view traces in the Langfuse dashboard:

1. **Traces**: See all agent executions with timing and success rates
2. **Sessions**: Group related conversations
3. **Users**: Track user-specific interactions
4. **Generations**: Analyze individual LLM calls
5. **Scores**: Monitor success rates and quality metrics

### Key Metrics

- **Agent Success Rate**: Percentage of successful agent executions
- **Step Performance**: Timing for planning, reasoning, and response steps
- **Error Rates**: Failed executions and common error patterns
- **LLM Usage**: Token consumption and costs per step

## Development

### Running Tests

```bash
# Test Langfuse integration
python -m pytest test_langfuse_integration.py -v

# Test main functionality (includes observability)
python -m pytest test_main.py -v
```

### Debugging

Enable debug logging for more detailed information:

```python
import logging
logging.getLogger("langfuse").setLevel(logging.DEBUG)
```

### Local Development

For local development, you can run without Langfuse by simply not setting the environment variables. The application will log that observability is disabled and continue to work normally.

## Best Practices

### Production Deployment

1. **Always set credentials**: Configure proper Langfuse credentials for production
2. **Use session tracking**: Implement proper session and user ID management
3. **Monitor performance**: Check that observability doesn't impact response times
4. **Set up alerts**: Configure alerts for high error rates or performance issues

### Security

1. **Keep secrets secure**: Store Langfuse credentials securely (never in code)
2. **Use environment variables**: Configure through environment variables or secure secret management
3. **Monitor access**: Regularly review Langfuse project access and permissions

### Data Privacy

1. **Review data retention**: Configure appropriate data retention policies in Langfuse
2. **Sanitize sensitive data**: Be careful about logging sensitive user information
3. **Comply with regulations**: Ensure observability practices comply with relevant data protection regulations

## Troubleshooting

### Common Issues

1. **"Langfuse observability is disabled"**
   - Check environment variables are set correctly
   - Verify Langfuse credentials are valid
   - Check network connectivity to Langfuse host

2. **"No trace found in the current context"**
   - This is expected for standalone function calls
   - Traces are created at the agent execution level

3. **High memory usage**
   - Check Langfuse flush intervals
   - Consider adjusting trace sampling in high-volume scenarios

### Support

- **Langfuse Documentation**: https://langfuse.com/docs
- **GitHub Issues**: Report issues in the Twinly repository
- **Community**: Join the Langfuse Discord for community support

## Version History

- **v1.0.0**: Initial Langfuse integration with comprehensive agent workflow tracing
- Full plan-reason-respond paradigm observability
- Safe decorator patterns with graceful fallback
- Health endpoint enhancement with observability status
