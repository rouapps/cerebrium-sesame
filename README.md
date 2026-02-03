# Cerebrium Sesame CSM Deployment

This folder contains the deployment files for running Sesame CSM (Conversational Speech Model) on Cerebrium's serverless GPU platform.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

You are free to:
- Use this software for any purpose
- Copy, modify, and distribute it
- Use it commercially and privately

The only requirement is to include a copy of the license and copyright notice.

## Prerequisites

1. **Cerebrium Account**: Sign up at [dashboard.cerebrium.ai](https://dashboard.cerebrium.ai/register)

2. **HuggingFace Access**: You need access to these gated models:
   - [sesame/csm-1b](https://huggingface.co/sesame/csm-1b) - Request access
   - [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) - Request access

3. **HuggingFace Token**: Generate at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Deployment Steps

### 1. Install Cerebrium CLI

```bash
pip install cerebrium --upgrade
```

### 2. Login to Cerebrium

```bash
cerebrium login
```

### 3. Set Environment Variables

In your Cerebrium dashboard, go to **Secrets** and add:

- `HF_TOKEN`: Your HuggingFace token
- `HF_HUB_ENABLE_HF_TRANSFER`: `1`
- `HF_HOME`: `/persistent-storage/.cache/huggingface/hub`

### 4. Deploy

From this directory:

```bash
cerebrium deploy
```

The first deployment takes 5-10 minutes to download models and build the container.

### 5. Get Your Endpoint URL

After deployment, you'll receive an endpoint URL like:
```
https://api.cortex.cerebrium.ai/v4/YOUR_PROJECT_ID/sesame-csm-streaming
```

Use this URL in the main application's setup page.

## API Endpoints

### Non-Streaming Generation

```python
import requests
import base64

response = requests.post(
    "https://api.cortex.cerebrium.ai/v4/YOUR_PROJECT/sesame-csm-streaming/predict",
    json={
        "text": "Hello, this is a test.",
        "speaker": 0,
        "stream": False
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

result = response.json()
audio_bytes = base64.b64decode(result["result"]["audio_data"])
```

### Streaming Generation (SSE)

```python
import requests
import json
import base64

response = requests.post(
    "https://api.cortex.cerebrium.ai/v4/YOUR_PROJECT/sesame-csm-streaming/predict",
    json={
        "text": "Hello, this is a streaming test.",
        "speaker": 0,
        "stream": True
    },
    headers={
        "Authorization": "Bearer YOUR_API_KEY",
        "Accept": "text/event-stream"
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b"data: "):
        data = json.loads(line[6:])
        if data.get("done"):
            break
        audio_chunk = base64.b64decode(data["audio"])
        # Process audio chunk...
```

## Configuration

Edit `cerebrium.toml` to adjust:

- **Hardware**: Change `compute` for different GPUs (A10, A100, etc.)
- **Scaling**: Adjust `min_replicas`, `max_replicas` for autoscaling
- **Memory**: Increase `memory` if needed

## Files

- `cerebrium.toml` - Deployment configuration
- `main.py` - API endpoint handlers
- `generator.py` - CSM model wrapper with streaming
- `models.py` - Model architecture definitions
- `requirements.txt` - Python dependencies

## Costs

Approximate costs on Cerebrium:
- A10 GPU: ~$0.0001/second
- Cold start: ~30-60 seconds (first request after idle)
- Warm inference: ~1-2 seconds for first audio chunk

Set `min_replicas = 1` to avoid cold starts (but incurs idle costs).

## Troubleshooting

**"Model not found" error**: Ensure your HF_TOKEN has access to the gated models.

**Out of memory**: Increase `memory` in cerebrium.toml or use a larger GPU.

**Slow first response**: This is the cold start. Set `min_replicas = 1` for faster responses.
