"""
Cerebrium-optimized Sesame CSM Generator with streaming support.

Based on official SesameAILabs/csm implementation.
Key insight: MIMI decoder must decode ALL frames at once, not frame-by-frame.
Streaming happens AFTER decoding, not during.
"""
from dataclasses import dataclass
from typing import List, Tuple, Generator as PyGenerator
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
    CSM audio generator following official implementation pattern.
    
    IMPORTANT: MIMI decoder requires all frames at once. Do NOT attempt
    frame-by-frame decoding - it causes CUDA graph failures and audio artifacts.
    """
    
    def __init__(self, model: Model):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        
        # Load MIMI exactly as official implementation does - no dtype changes
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)  # Official uses 32
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text segment."""
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        return text_frame.to(self.device), text_frame_mask.to(self.device)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize audio using MIMI codec."""
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        
        # Add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        return audio_frame, audio_frame_mask

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a complete segment (text + audio)."""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
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
    ) -> torch.Tensor:
        """
        Generate complete audio (official implementation pattern).
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: List of context segments for voice cloning
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            
        Returns:
            Complete audio tensor
        """
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

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            # Truncate context if too long
            curr_tokens = curr_tokens[:, -max_context_len:]
            curr_tokens_mask = curr_tokens_mask[:, -max_context_len:]
            curr_pos = torch.arange(0, curr_tokens.size(1)).unsqueeze(0).long().to(self.device)

        # Generate all frames
        for _ in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # EOS

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        if not samples:
            return torch.tensor([])

        # Decode ALL frames at once - this is critical!
        # MIMI decoder uses CUDA graphs that expect full sequence
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
        chunk_size_samples: int = 24000,  # 1 second chunks
    ) -> PyGenerator[torch.Tensor, None, None]:
        """
        Generate audio and stream output in chunks.
        
        IMPORTANT: This generates ALL frames first, decodes once, then 
        streams the decoded audio in chunks. Frame-by-frame decoding 
        does NOT work with MIMI.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID
            context: List of context segments for voice cloning
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            chunk_size_samples: Size of audio chunks to yield (samples)
            
        Yields:
            Audio chunks as torch tensors
        """
        # Generate complete audio first
        audio = self.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk
        )
        
        if audio.numel() == 0:
            return
        
        # Stream the decoded audio in chunks
        total_samples = audio.shape[0]
        for start in range(0, total_samples, chunk_size_samples):
            end = min(start + chunk_size_samples, total_samples)
            yield audio[start:end].float().cpu()


def load_csm_1b(device: str = "cuda") -> Generator:
    """
    Load the CSM-1B model following official implementation.
    
    Args:
        device: Device to load model on
        
    Returns:
        Generator instance
    """
    logger.info("Loading CSM-1B model...")
    
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)
    
    # NOTE: No torch.compile - conflicts with MIMI's CUDA graphs
    
    generator = Generator(model)
    logger.info("CSM-1B model loaded successfully")
    
    return generator
