"""
Cerebrium-optimized Sesame CSM Generator with streaming support.
"""
from dataclasses import dataclass
from typing import List, Tuple, Generator as PyGenerator, Optional
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
    """CSM audio generator with streaming capabilities."""
    
    def __init__(self, model: Model):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()
        device = next(model.parameters()).device

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        
        num_codebooks = model.config.audio_num_codebooks
        mimi.set_num_codebooks(num_codebooks)
        self._num_codebooks = num_codebooks
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

        self._first_chunk_size = 5     # First chunk FAST (~400ms) for low latency
        self._stream_buffer_size = 25  # Subsequent chunks larger for smooth audio
        self.max_seq_len = 2048
        self._text_token_cache = {}

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text segment with caching."""
        cache_key = f"{speaker}:{text}"
        
        if cache_key in self._text_token_cache:
            return self._text_token_cache[cache_key]

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), self._num_codebooks + 1, dtype=torch.long, device=self.device)
        text_frame_mask = torch.zeros(len(text_tokens), self._num_codebooks + 1, dtype=torch.bool, device=self.device)
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True

        result = (text_frame, text_frame_mask)
        self._text_token_cache[cache_key] = result
        return result

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize audio using MIMI codec."""
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        audio_tokens = audio_tokens[:self._num_codebooks, :]
        
        # Add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), self._num_codebooks + 1).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self._num_codebooks + 1).bool().to(self.device)
        audio_frame[:, :self._num_codebooks] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :self._num_codebooks] = True

        return audio_frame, audio_frame_mask

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a complete segment (text + audio)."""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        total_len = text_tokens.size(0) + audio_tokens.size(0)

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
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.8,
        topk: int = 50,
    ) -> PyGenerator[torch.Tensor, None, None]:
        """
        Generate audio in a streaming fashion.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: List of context segments for voice cloning
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            
        Yields:
            Audio chunks as torch tensors
        """
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []

        batch_size = 10  # Process frames in small batches
        first_chunk_size = 5   # FAST first chunk (~400ms audio)
        buffer_size = 25  # Larger subsequent chunks for smooth audio
        is_first_chunk = True

        # Tokenize context segments
        if context:
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

        # Tokenize generation text
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        # Truncate if too long
        max_seq_len = 2048
        if prompt_tokens.size(0) > max_seq_len:
            prompt_tokens = prompt_tokens[-max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        frame_buffer = []
        zeros_1_1 = torch.zeros(1, 1).long().to(self.device)
        zeros_mask_1_1 = torch.zeros(1, 1).bool().to(self.device)

        def update_tokens(sample):
            nonlocal curr_tokens, curr_tokens_mask, curr_pos
            ones = torch.ones_like(sample).bool()
            curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones, zeros_mask_1_1], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        with self._audio_tokenizer.streaming(1):
            i = 0
            
            while i < max_generation_len:
                batch_end = min(i + batch_size, max_generation_len)
                batch_samples = []

                for _ in range(batch_end - i):
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        sample = self._model.generate_frame(
                            curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                        )
                    
                    if torch.all(sample == 0):
                        break

                    batch_samples.append(sample)
                    update_tokens(sample)

                if not batch_samples:
                    break

                frame_buffer.extend(batch_samples)
                i += len(batch_samples)

                # Use smaller size for FIRST chunk (low latency), larger for rest (smooth)
                current_buffer_target = first_chunk_size if is_first_chunk else buffer_size

                # Yield audio chunk when buffer reaches target
                if len(frame_buffer) >= current_buffer_target:
                    frames_to_process = frame_buffer[:current_buffer_target]
                    frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                    audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                    frame_buffer = frame_buffer[current_buffer_target:]
                    is_first_chunk = False  # Switch to larger chunks after first
                    yield audio_chunk.cpu()

            # Process remaining frames
            if frame_buffer:
                current_size = buffer_size  # Use standard size for padding calc
                if len(frame_buffer) < current_size:
                    # Pad with zeros for decoder
                    padding_frames = [
                        torch.zeros_like(frame_buffer[0]) 
                        for _ in range(current_size - len(frame_buffer))
                    ]
                    frames_to_process = frame_buffer + padding_frames
                    actual_frame_count = len(frame_buffer)
                else:
                    frames_to_process = frame_buffer[:current_size]
                    actual_frame_count = current_size
                    
                frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                
                # Return only non-padded portion
                if len(frame_buffer) < current_size:
                    actual_samples = int(audio_chunk.shape[0] * actual_frame_count / current_size)
                    audio_chunk = audio_chunk[:actual_samples]
                    
                yield audio_chunk.cpu()

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.5,
        topk: int = 30,
    ) -> torch.Tensor:
        """
        Generate complete audio by collecting streaming chunks.
        Uses streaming internally but returns complete audio.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: List of context segments
            max_audio_length_ms: Maximum audio length
            temperature: Sampling temperature
            topk: Top-k sampling
            
        Returns:
            Complete audio tensor
        """
        chunks = list(self.generate_stream(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk
        ))
        
        if not chunks:
            return torch.tensor([])
        
        return torch.cat(chunks)


def load_csm_1b(device: str = "cuda") -> Generator:
    """
    Load the CSM-1B model with optimizations for low latency.
    
    Args:
        device: Device to load model on
        
    Returns:
        Generator instance
    """
    logger.info("Loading CSM-1B model with optimizations...")
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    model = Model.from_pretrained("sesame/csm-1b")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model.to(device=device, dtype=dtype)
    
    # Compile model for faster inference (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("Model compiled with torch.compile for faster inference")
    except Exception as e:
        logger.warning(f"torch.compile not available: {e}")
    
    generator = Generator(model)
    logger.info("CSM-1B model loaded successfully")
    
    return generator
