"""
Cerebrium-optimized Sesame CSM Generator with TRUE streaming support.

Based on official SesameAILabs/csm implementation with optimizations from csm-streaming.
Key optimizations:
- torch.compile with reduce-overhead mode
- CUDA TF32, Flash Attention, cuDNN optimizations
- True streaming with MIMI's streaming context
- Frame batching for better GPU utilization
- Tokenization caching with LRU cache
- Pre-allocated tensors for reduced allocation overhead
"""
from dataclasses import dataclass
from typing import List, Tuple, Generator as PyGenerator, Optional, Callable
from functools import lru_cache
from collections import OrderedDict
import time
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """Audio segment with text and speaker information."""
    speaker: int
    text: str
    audio: torch.Tensor
    sample_rate: int = 24_000


def load_llama3_tokenizer():
    """Load and configure the Llama 3.2 tokenizer."""
    tokenizer_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )
    return tokenizer


class Generator:
    """
    CSM audio generator with extreme performance optimizations.
    
    Optimizations:
    - MIMI streaming context for true streaming decode
    - Frame batching (20 frames per batch)
    - Pre-allocated tensors
    - Tokenization caching
    """
    
    def __init__(self, model: Model):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        
        # Load MIMI codec
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        
        # Get num_codebooks from model config
        self._num_codebooks = model.config.audio_num_codebooks
        mimi.set_num_codebooks(self._num_codebooks)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device
        
        # Streaming configuration
        self._stream_buffer_size = 20  # Frames to buffer before decode
        self.max_seq_len = 2048
        
        # Caches
        self._text_token_cache = {}
        self._cache = OrderedDict()
        
        # Pre-allocate common tensors for reduced allocation overhead
        self._zeros_1_1 = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._zeros_mask_1_1 = torch.zeros(1, 1, dtype=torch.bool, device=device)
        
        # Performance settings
        torch.set_num_threads(16)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text segment with caching for reduced latency."""
        # Check cache first
        cache_key = f"{speaker}:{text}"
        
        if cache_key in self._text_token_cache:
            return self._text_token_cache[cache_key]

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), self._num_codebooks + 1, dtype=torch.long, device=self.device)
        text_frame_mask = torch.zeros(len(text_tokens), self._num_codebooks + 1, dtype=torch.bool, device=self.device)
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True

        result = (text_frame, text_frame_mask)
        
        # Cache result (limit cache size)
        if len(self._text_token_cache) > 1024:
            # Remove oldest entry
            oldest_key = next(iter(self._text_token_cache))
            del self._text_token_cache[oldest_key]
        self._text_token_cache[cache_key] = result
        
        return result

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize audio using MIMI codec."""
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        
        # Limit to the number of codebooks
        audio_tokens = audio_tokens[:self._num_codebooks, :]
        
        # Add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1, device=self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), self._num_codebooks + 1, dtype=torch.long, device=self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self._num_codebooks + 1, dtype=torch.bool, device=self.device)
        audio_frame[:, :self._num_codebooks] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :self._num_codebooks] = True

        return audio_frame, audio_frame_mask

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a complete segment (text + audio)."""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        
        total_len = text_tokens.size(0) + audio_tokens.size(0)
        
        # Truncate if too long
        if total_len > self.max_seq_len:
            overflow = total_len - self.max_seq_len
            if text_tokens.size(0) > overflow:
                text_tokens = text_tokens[overflow:]
                text_masks = text_masks[overflow:]
            else:
                audio_overflow = overflow - text_tokens.size(0)
                text_tokens = text_tokens[0:0]
                text_masks = text_masks[0:0]
                audio_tokens = audio_tokens[audio_overflow:]
                audio_masks = audio_masks[audio_overflow:]
        
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        stream: bool = False,
    ) -> torch.Tensor:
        """
        Generate audio with optional streaming optimization.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: List of context segments for voice cloning
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            stream: If True, uses streaming generation internally
            
        Returns:
            Complete audio tensor
        """
        if stream:
            # Collect streaming chunks into complete audio
            audio_chunks = []
            for chunk in self.generate_stream(
                text=text,
                speaker=speaker,
                context=context,
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                topk=topk,
            ):
                audio_chunks.append(chunk)
            
            if not audio_chunks:
                return torch.tensor([])
            return torch.cat(audio_chunks)
        
        # Non-streaming generation (optimized)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        # Truncate if needed
        if prompt_tokens.size(0) > self.max_seq_len:
            prompt_tokens = prompt_tokens[-self.max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-self.max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=self.device).unsqueeze(0).long()

        samples = []
        
        # Use MIMI streaming context for optimized decode
        with self._audio_tokenizer.streaming(1):
            for _ in range(max_generation_len):
                with torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cuda', dtype=torch.bfloat16):
                    sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                
                if torch.all(sample == 0):
                    break  # EOS

                samples.append(sample)

                curr_tokens = torch.cat([sample, self._zeros_1_1], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample, dtype=torch.bool), self._zeros_mask_1_1], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

        if not samples:
            return torch.tensor([])

        # Decode ALL frames at once
        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        return audio

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        chunk_frames: int = 20,  # Number of frames per chunk
        on_chunk_generated: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> PyGenerator[torch.Tensor, None, None]:
        """
        Generate audio with TRUE streaming - yields audio chunks as they're generated.
        
        This is the optimized streaming approach that generates frames in batches,
        decodes them using MIMI's streaming context, and yields audio chunks
        for immediate playback or transmission.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: List of context segments for voice cloning
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            chunk_frames: Number of frames to batch before decoding (default 20)
            on_chunk_generated: Optional callback for each generated chunk
            
        Yields:
            Audio chunks as torch tensors (CPU, float32)
        """
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        
        # Streaming parameters - optimized for low latency
        initial_batch_size = 20  # Frames to generate before first decode
        normal_batch_size = 20   # Frames per batch after first chunk
        buffer_size = initial_batch_size
        expected_frame_count = buffer_size
        first_chunk_delivered = False

        tokens, tokens_mask = [], []

        # Tokenize context
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        # Tokenize generation prompt
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        # Truncate if needed
        if prompt_tokens.size(0) > self.max_seq_len:
            prompt_tokens = prompt_tokens[-self.max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-self.max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=self.device).unsqueeze(0).long()

        frame_buffer = []
        
        # Pre-allocated tensors for token updates
        zeros_1_1 = self._zeros_1_1
        zeros_mask_1_1 = self._zeros_mask_1_1

        def update_tokens(sample):
            nonlocal curr_tokens, curr_tokens_mask, curr_pos
            ones = torch.ones_like(sample, dtype=torch.bool)
            curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones, zeros_mask_1_1], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        generation_start = time.time()
        i = 0
        
        # Use MIMI streaming context for efficient chunk-by-chunk decode
        with self._audio_tokenizer.streaming(1):
            while i < max_generation_len:
                batch_end = min(i + buffer_size, max_generation_len)
                batch_samples = []

                # Generate a batch of frames
                for _ in range(batch_end - i):
                    with torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cuda', dtype=torch.bfloat16):
                        sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                    
                    # Check for EOS
                    if torch.all(sample == 0):
                        break
                    
                    batch_samples.append(sample)
                    update_tokens(sample)

                if not batch_samples:
                    break

                frame_buffer.extend(batch_samples)
                i += len(batch_samples)

                # Decode and yield when buffer is full
                if len(frame_buffer) >= expected_frame_count:
                    frames_to_process = frame_buffer[:expected_frame_count]
                    
                    # Pad if necessary (for consistent decode shapes)
                    if len(frames_to_process) < expected_frame_count:
                        padding_frames = [
                            torch.zeros_like(frames_to_process[0])
                            for _ in range(expected_frame_count - len(frames_to_process))
                        ]
                        frames_to_process = frames_to_process + padding_frames
                    
                    # Decode chunk
                    frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                    audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                    
                    # Keep remaining frames
                    frame_buffer = frame_buffer[expected_frame_count:]
                    
                    # Process chunk
                    cpu_chunk = audio_chunk.float().cpu()
                    
                    if on_chunk_generated:
                        on_chunk_generated(cpu_chunk)
                    
                    # Switch to normal batch size after first chunk
                    if not first_chunk_delivered:
                        buffer_size = normal_batch_size
                        expected_frame_count = buffer_size
                        first_chunk_delivered = True
                    
                    yield cpu_chunk

            # Process remaining frames
            if frame_buffer:
                actual_frame_count = len(frame_buffer)
                
                # Pad to expected size for decode
                if actual_frame_count < expected_frame_count:
                    padding_frames = [
                        torch.zeros_like(frame_buffer[0])
                        for _ in range(expected_frame_count - actual_frame_count)
                    ]
                    frames_to_process = frame_buffer + padding_frames
                else:
                    frames_to_process = frame_buffer
                
                frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                
                # Trim to actual length (remove padding audio)
                if actual_frame_count < expected_frame_count:
                    actual_samples = int(audio_chunk.shape[0] * actual_frame_count / expected_frame_count)
                    audio_chunk = audio_chunk[:actual_samples]
                
                cpu_chunk = audio_chunk.float().cpu()
                
                if on_chunk_generated:
                    on_chunk_generated(cpu_chunk)
                
                yield cpu_chunk

        # Log performance metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - generation_start
        frames_generated = i
        audio_seconds = frames_generated * 0.08  # 80ms per frame
        rtf = total_time / audio_seconds if audio_seconds > 0 else float('inf')
        logger.info(f"Generated {frames_generated} frames ({audio_seconds:.2f}s audio) in {total_time:.2f}s (RTF: {rtf:.3f}x)")


def warmup_generator(gen: Generator, warmup_text: str = "Hello, this is a warmup.", speaker_id: int = 0):
    """
    Perform warmup to reduce first-generation latency.
    
    This exercises all model components to ensure CUDA kernels are compiled
    and caches are warm.
    """
    logger.info("Starting generator warmup...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Create dummy audio for context
    dummy_audio = torch.zeros(24000, device=gen.device)  # 1 second
    dummy_segment = Segment(
        speaker=speaker_id,
        text="Warmup context segment.",
        audio=dummy_audio
    )
    
    # Exercise tokenization
    gen._tokenize_text_segment(warmup_text, speaker_id)
    gen._tokenize_segment(dummy_segment)
    
    # Exercise frame generation with different parameters
    with torch.inference_mode():
        dummy_tokens = torch.ones(1, 10, gen._num_codebooks + 1, dtype=torch.long, device=gen.device)
        dummy_mask = torch.ones(1, 10, gen._num_codebooks + 1, dtype=torch.bool, device=gen.device)
        dummy_pos = torch.arange(0, 10, device=gen.device).unsqueeze(0)
        
        for temp in [0.7, 0.8, 0.9]:
            for topk in [30, 40, 50]:
                _ = gen._model.generate_frame(dummy_tokens, dummy_mask, dummy_pos, temp, topk)
    
    # Run a short generation to fully warm up
    try:
        _ = gen.generate(
            text=warmup_text,
            speaker=speaker_id,
            context=[dummy_segment],
            max_audio_length_ms=2000,  # Short warmup
            temperature=0.9,
            topk=50,
        )
    except Exception as e:
        logger.warning(f"Warmup generation warning: {e}")
    
    # Clear tokenization cache to avoid warmup entries
    gen._text_token_cache.clear()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    logger.info("Generator warmup complete")


def load_csm_1b(device: str = "cuda", compile_model: bool = True) -> Generator:
    """
    Load the CSM-1B model with extreme performance optimizations.
    
    Optimizations applied:
    - CUDA TF32 matmul
    - Flash SDP (Scaled Dot Product) attention
    - cuDNN benchmarking
    - torch.compile with reduce-overhead mode (optional)
    - Model warmup
    
    Args:
        device: Device to load model on
        compile_model: Whether to apply torch.compile (recommended for repeated inference)
        
    Returns:
        Optimized Generator instance
    """
    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.info("Loading CSM-1B model with extreme optimizations...")
    
    model = Model.from_pretrained("sesame/csm-1b")
    
    # Determine best dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    
    # Apply torch.compile for faster inference
    # Note: This may conflict with MIMI's CUDA graphs in some cases
    if compile_model:
        try:
            logger.info("Applying torch.compile optimization...")
            model.backbone = torch.compile(
                model.backbone,
                mode='reduce-overhead',
                fullgraph=True,
                backend='inductor'
            )
            model.decoder = torch.compile(
                model.decoder,
                mode='reduce-overhead',
                fullgraph=True,
                backend='inductor'
            )
            logger.info("torch.compile applied successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without: {e}")
    
    model.to(device=device, dtype=dtype)
    
    logger.info("Creating generator...")
    generator = Generator(model)
    
    # Perform warmup
    warmup_generator(generator)
    
    logger.info("CSM-1B model loaded and optimized successfully")
    
    return generator


def load_csm_1b_fast(device: str = "cuda") -> Generator:
    """
    Load CSM-1B without torch.compile for faster cold start.
    
    Use this if you need fast startup and are okay with slightly
    slower inference. Good for development/testing.
    """
    return load_csm_1b(device=device, compile_model=False)
