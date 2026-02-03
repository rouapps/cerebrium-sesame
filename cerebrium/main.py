"""
Cerebrium Sesame CSM Streaming API Endpoint.

This module provides a streaming TTS endpoint using Server-Sent Events (SSE)
for real-time audio generation with Sesame's CSM model.
"""
import os
import json
import base64
import logging
from typing import Optional, List
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from generator import load_csm_1b, Segment, Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

logger.info(f"Using device: {device}")

# Load the model at startup (stays in memory)
generator: Optional[Generator] = None


def get_generator() -> Generator:
    """Get or initialize the generator singleton."""
    global generator
    if generator is None:
        logger.info("Loading CSM-1B model...")
        generator = load_csm_1b(device=device)
        logger.info("Model loaded successfully")
    return generator


# Pre-load model on import
generator = get_generator()

# Default reference audio for voice context - SINGLE context for faster processing
# Using just one speaker context reduces tokenization time significantly
DEFAULT_SPEAKERS = [0]
DEFAULT_TRANSCRIPTS = [
    (
        "like revising for an exam I'd have to try and like keep up the momentum because I'd "
        "start really early I'd be like okay I'm gonna start revising now"
    ),
]

# Download default reference audio (single file for speed)
DEFAULT_AUDIO_PATHS = [
    hf_hub_download(repo_id="sesame/csm-1b", filename="prompts/conversational_a.wav"),
]


def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    """Load and resample audio file."""
    # Use soundfile directly to avoid torchcodec/FFmpeg issues
    audio_np, sample_rate = sf.read(audio_path)
    # Convert to tensor (soundfile returns float64, we need float32)
    audio_tensor = torch.from_numpy(audio_np.astype(np.float32))
    # Handle stereo -> mono if needed
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=-1)
    # Resample if needed
    if sample_rate != target_sample_rate:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
        )
    return audio_tensor


# Cache the default context (loaded once at startup)
_cached_context: Optional[List[Segment]] = None

def get_default_context() -> List[Segment]:
    """Get default voice context segments (cached for speed)."""
    global _cached_context
    
    if _cached_context is not None:
        return _cached_context
    
    gen = get_generator()
    segments = []
    
    for transcript, speaker, audio_path in zip(
        DEFAULT_TRANSCRIPTS, DEFAULT_SPEAKERS, DEFAULT_AUDIO_PATHS
    ):
        audio = load_prompt_audio(audio_path, gen.sample_rate)
        segments.append(Segment(
            text=transcript,
            speaker=speaker,
            audio=audio
        ))
    
    _cached_context = segments
    logger.info("Default context cached for fast inference")
    return segments

# Pre-load context at startup
_cached_context = get_default_context()


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for audio generation."""
    text: str
    speaker: int = 0
    max_audio_length_ms: int = 30000
    temperature: float = 0.9  # Official default
    topk: int = 50  # Official default
    use_default_context: bool = True
    # Optional: custom context audio as base64
    context_audio_b64: Optional[str] = None
    context_text: Optional[str] = None


class AudioChunk(BaseModel):
    """Single audio chunk in SSE stream."""
    audio: str  # Base64 encoded int16 PCM
    sample_rate: int
    chunk_index: int


class GenerateResponse(BaseModel):
    """Response model for non-streaming generation."""
    audio_data: str  # Base64 encoded WAV
    format: str = "wav"
    encoding: str = "base64"
    sample_rate: int


def generate_audio(text: str, speaker: int = 0, max_audio_length_ms: int = 30000,
                   temperature: float = 0.9, topk: int = 50) -> dict:
    """
    Generate complete audio (non-streaming).
    
    Args:
        text: Text to synthesize
        speaker: Speaker ID (0 or 1)
        max_audio_length_ms: Maximum audio length
        temperature: Sampling temperature
        topk: Top-k sampling
        
    Returns:
        Dict with base64 encoded audio
    """
    gen = get_generator()
    context = get_default_context()
    
    audio = gen.generate(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        topk=topk
    )
    
    # Save to temporary file and encode using soundfile directly
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    # Convert tensor to numpy and save with soundfile
    audio_np = audio.cpu().numpy()
    sf.write(temp_path, audio_np, gen.sample_rate)
    
    with open(temp_path, "rb") as f:
        wav_data = f.read()
    
    os.remove(temp_path)
    
    return {
        "audio_data": base64.b64encode(wav_data).decode("utf-8"),
        "format": "wav",
        "encoding": "base64",
        "sample_rate": gen.sample_rate
    }


def generate_audio_stream(text: str, speaker: int = 0, max_audio_length_ms: int = 30000,
                          temperature: float = 0.9, topk: int = 50):
    """
    Generate audio with streaming via Server-Sent Events.
    
    Args:
        text: Text to synthesize
        speaker: Speaker ID
        max_audio_length_ms: Maximum audio length
        temperature: Sampling temperature
        topk: Top-k sampling
        
    Yields:
        SSE formatted strings with base64 encoded audio chunks
    """
    gen = get_generator()
    context = get_default_context()
    
    chunk_index = 0
    
    for audio_chunk in gen.generate_stream(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        topk=topk
    ):
        # Convert to int16 PCM bytes
        audio_np = audio_chunk.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Encode as base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Format as SSE event
        event_data = {
            "audio": audio_b64,
            "sample_rate": gen.sample_rate,
            "chunk_index": chunk_index,
            "done": False
        }
        
        yield f"data: {json.dumps(event_data)}\n\n"
        chunk_index += 1
    
    # Send completion event
    yield f'data: {{"done": true, "total_chunks": {chunk_index}}}\n\n'


# For Cerebrium, we need to expose these as callable functions
# The main entry point is either generate_audio or generate_audio_stream

def predict(text: str, speaker: int = 0, stream: bool = False, 
            max_audio_length_ms: int = 30000, temperature: float = 0.9, 
            topk: int = 50) -> dict:
    """
    Main prediction endpoint for Cerebrium.
    
    Args:
        text: Text to synthesize
        speaker: Speaker ID
        stream: Whether to return streaming response
        max_audio_length_ms: Maximum audio length
        temperature: Sampling temperature
        topk: Top-k sampling
        
    Returns:
        For stream=False: Dict with complete audio
        For stream=True: Generator yielding SSE events
    """
    if stream:
        # Return generator for streaming
        return generate_audio_stream(
            text=text,
            speaker=speaker,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk
        )
    else:
        # Return complete audio
        return generate_audio(
            text=text,
            speaker=speaker,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk
        )


# Health check
def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "device": device
    }


if __name__ == "__main__":
    # Test the generation
    print("Testing audio generation...")
    result = generate_audio("Hello, this is a test of the Sesame CSM model on Cerebrium.")
    print(f"Generated audio: {len(result['audio_data'])} bytes (base64)")
    print("Test complete!")
