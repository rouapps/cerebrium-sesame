"""
Cerebrium Sesame CSM Streaming API Endpoint.

This module provides a streaming TTS endpoint using Server-Sent Events (SSE)
for real-time audio generation with Sesame's CSM model.

Optimized for maximum inference speed with:
- torch.compile optimization
- CUDA TF32/Flash Attention
- True streaming generation
- Pre-warmed model
"""
import os
import json
import base64
import logging
import time
from typing import Optional, List
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

from generator import load_csm_1b, load_csm_1b_fast, Segment, Generator

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
        logger.info("Loading CSM-1B model with optimizations...")
        # Use compiled model for best inference speed
        # Set compile_model=False if you encounter CUDA graph conflicts
        generator = load_csm_1b(device=device, compile_model=True)
        logger.info("Model loaded successfully")
    return generator


# Pre-load model on import
generator = get_generator()

# Default reference audio for voice context - SINGLE context for faster processing
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
    audio_np, sample_rate = sf.read(audio_path)
    audio_tensor = torch.from_numpy(audio_np.astype(np.float32))
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=-1)
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
    temperature: float = 0.9
    topk: int = 50
    use_default_context: bool = True
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
    Generate complete audio (non-streaming, optimized).
    
    Args:
        text: Text to synthesize
        speaker: Speaker ID (0 or 1)
        max_audio_length_ms: Maximum audio length
        temperature: Sampling temperature
        topk: Top-k sampling
        
    Returns:
        Dict with base64 encoded audio and performance metrics
    """
    gen = get_generator()
    context = get_default_context()
    
    start_time = time.time()
    
    # Use streaming internally for better performance
    audio = gen.generate(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        topk=topk,
        stream=True  # Use streaming generation internally
    )
    
    generation_time = time.time() - start_time
    audio_duration = audio.shape[0] / gen.sample_rate
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
    
    logger.info(f"Generated {audio_duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.3f}x)")
    
    # Save to temporary file and encode
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    audio_np = audio.cpu().numpy()
    sf.write(temp_path, audio_np, gen.sample_rate)
    
    with open(temp_path, "rb") as f:
        wav_data = f.read()
    
    os.remove(temp_path)
    
    return {
        "audio_data": base64.b64encode(wav_data).decode("utf-8"),
        "format": "wav",
        "encoding": "base64",
        "sample_rate": gen.sample_rate,
        "audio_duration_seconds": audio_duration,
        "generation_time_seconds": generation_time,
        "real_time_factor": rtf
    }


def generate_audio_stream(text: str, speaker: int = 0, max_audio_length_ms: int = 30000,
                          temperature: float = 0.9, topk: int = 50):
    """
    Generate audio with TRUE streaming via Server-Sent Events.
    
    This uses the optimized streaming generation that yields audio chunks
    as they are generated, enabling real-time playback with minimal latency.
    
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
    start_time = time.time()
    first_chunk_time = None
    total_audio_samples = 0
    
    for audio_chunk in gen.generate_stream(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        topk=topk,
    ):
        current_time = time.time()
        
        # Track first chunk latency
        if first_chunk_time is None:
            first_chunk_time = current_time - start_time
            logger.info(f"First chunk latency: {first_chunk_time*1000:.1f}ms")
        
        # Convert to int16 PCM bytes
        audio_np = audio_chunk.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Track audio duration
        total_audio_samples += len(audio_np)
        
        # Encode as base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Format as SSE event
        event_data = {
            "audio": audio_b64,
            "sample_rate": gen.sample_rate,
            "chunk_index": chunk_index,
            "chunk_duration_ms": len(audio_np) / gen.sample_rate * 1000,
            "done": False
        }
        
        yield f"data: {json.dumps(event_data)}\n\n"
        chunk_index += 1
    
    # Calculate final metrics
    total_time = time.time() - start_time
    total_audio_duration = total_audio_samples / gen.sample_rate
    rtf = total_time / total_audio_duration if total_audio_duration > 0 else float('inf')
    
    # Send completion event with metrics
    completion_data = {
        "done": True,
        "total_chunks": chunk_index,
        "first_chunk_latency_ms": first_chunk_time * 1000 if first_chunk_time else 0,
        "total_generation_time_seconds": total_time,
        "total_audio_duration_seconds": total_audio_duration,
        "real_time_factor": rtf
    }
    
    logger.info(f"Streaming complete: {total_audio_duration:.2f}s audio in {total_time:.2f}s (RTF: {rtf:.3f}x)")
    
    yield f'data: {json.dumps(completion_data)}\n\n'


def generate_audio_stream_raw(text: str, speaker: int = 0, max_audio_length_ms: int = 30000,
                              temperature: float = 0.9, topk: int = 50):
    """
    Generate audio with streaming - yields raw audio bytes directly.
    
    Use this for direct audio streaming without SSE wrapper.
    
    Args:
        text: Text to synthesize
        speaker: Speaker ID
        max_audio_length_ms: Maximum audio length
        temperature: Sampling temperature
        topk: Top-k sampling
        
    Yields:
        Raw int16 PCM audio bytes
    """
    gen = get_generator()
    context = get_default_context()
    
    for audio_chunk in gen.generate_stream(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        topk=topk,
    ):
        # Convert to int16 PCM bytes
        audio_np = audio_chunk.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        yield audio_int16.tobytes()


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
        return generate_audio_stream(
            text=text,
            speaker=speaker,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk
        )
    else:
        return generate_audio(
            text=text,
            speaker=speaker,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk
        )


def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "device": device,
        "optimizations": {
            "torch_compile": True,
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            "cudnn_benchmark": torch.backends.cudnn.benchmark if torch.cuda.is_available() else False,
        }
    }


def benchmark(text: str = "Hello, this is a benchmark test for the Sesame CSM model.",
              num_runs: int = 3) -> dict:
    """
    Run benchmark to measure inference speed.
    
    Args:
        text: Text to generate
        num_runs: Number of benchmark runs
        
    Returns:
        Dict with benchmark results
    """
    gen = get_generator()
    context = get_default_context()
    
    times = []
    audio_durations = []
    
    # Warmup run
    _ = gen.generate(
        text=text,
        speaker=0,
        context=context,
        max_audio_length_ms=10000,
        stream=True
    )
    
    # Benchmark runs
    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        audio = gen.generate(
            text=text,
            speaker=0,
            context=context,
            max_audio_length_ms=30000,
            stream=True
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        duration = audio.shape[0] / gen.sample_rate
        
        times.append(elapsed)
        audio_durations.append(duration)
        
        logger.info(f"Benchmark run {i+1}: {duration:.2f}s audio in {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    avg_duration = sum(audio_durations) / len(audio_durations)
    avg_rtf = avg_time / avg_duration
    
    return {
        "num_runs": num_runs,
        "average_generation_time_seconds": avg_time,
        "average_audio_duration_seconds": avg_duration,
        "average_real_time_factor": avg_rtf,
        "all_times": times,
        "all_durations": audio_durations,
        "device": device
    }


if __name__ == "__main__":
    # Test the generation
    print("Testing optimized audio generation...")
    
    # Test non-streaming
    print("\n=== Non-streaming test ===")
    result = generate_audio("Hello, this is a test of the optimized Sesame CSM model.")
    print(f"Generated audio: {len(result['audio_data'])} bytes (base64)")
    print(f"Audio duration: {result.get('audio_duration_seconds', 'N/A')}s")
    print(f"Generation time: {result.get('generation_time_seconds', 'N/A')}s")
    print(f"RTF: {result.get('real_time_factor', 'N/A')}")
    
    # Test streaming
    print("\n=== Streaming test ===")
    chunk_count = 0
    for event in generate_audio_stream("This is a streaming test."):
        chunk_count += 1
        if chunk_count <= 3:  # Print first 3 chunks
            print(f"Chunk {chunk_count}: {len(event)} chars")
    print(f"Total chunks: {chunk_count}")
    
    # Run benchmark
    print("\n=== Benchmark ===")
    bench_result = benchmark(num_runs=2)
    print(f"Average RTF: {bench_result['average_real_time_factor']:.3f}x")
    
    print("\nTest complete!")
