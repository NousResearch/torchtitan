# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference server that exposes the TorchTitan training model as an HTTP API.

This enables true on-policy training by serving inference requests from the
SAME model instance that is being trained - shared memory, no weight copying.

The server implements an OpenAI-compatible API (subset) so Atropos environments
can connect to it as their inference endpoint.

Also implements vLLM's native /generate endpoint for compatibility with Atropos
environments that use the vLLM server type.
"""

import json
import threading
import time
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass
class InferenceConfig:
    """Configuration for inference server."""

    port: int = 9000
    host: str = "0.0.0.0"
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled


class InferenceServer:
    """
    HTTP server that exposes the training model for inference.

    This ensures the inference model and training model share the same weights
    in memory - critical for on-policy RL where the policy generating samples
    must be the same as the policy being updated.

    The server runs in a background thread, so training can continue while
    serving inference requests.
    
    Supports both OpenAI-compatible endpoints (/v1/completions) and vLLM's
    native /generate endpoint for Atropos compatibility.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: InferenceConfig | None = None,
        vllm_engine: Optional[Any] = None,
    ):
        """
        Initialize inference server.

        Args:
            model: The TorchTitan model (shared reference, not a copy!)
            tokenizer: HuggingFace tokenizer
            config: Server configuration
            vllm_engine: Optional VLLMRolloutEngine for fast generation with /generate endpoint
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()
        self.vllm_engine = vllm_engine
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        # Model info
        self.model_name = getattr(model, "name", "torchtitan-model")

    def start(self) -> None:
        """Start the inference server in a background thread."""
        if self._running:
            return

        handler = self._create_handler()
        self._server = HTTPServer((self.config.host, self.config.port), handler)
        self._running = True

        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

        print(f"[InferenceServer] Started on http://{self.config.host}:{self.config.port}")
        print(f"[InferenceServer] OpenAI-compatible endpoints available at /v1/...")

    def stop(self) -> None:
        """Stop the inference server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        print("[InferenceServer] Stopped")

    def _serve(self) -> None:
        """Server main loop."""
        while self._running and self._server:
            self._server.handle_request()

    def _create_handler(self):
        """Create HTTP request handler with access to model."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Suppress default logging
                pass

            def do_GET(self):
                if self.path == "/v1/models":
                    self._handle_models()
                elif self.path == "/health":
                    self._send_json({"status": "healthy"})
                else:
                    self._send_error(404, "Not found")

            def do_POST(self):
                if self.path == "/v1/completions":
                    self._handle_completions()
                elif self.path == "/v1/chat/completions":
                    self._handle_chat_completions()
                elif self.path == "/generate":
                    self._handle_generate()
                else:
                    self._send_error(404, "Not found")

            def _handle_models(self):
                """List available models."""
                self._send_json({
                    "object": "list",
                    "data": [{
                        "id": server.model_name,
                        "object": "model",
                        "owned_by": "torchtitan",
                    }]
                })

            def _handle_completions(self):
                """Handle /v1/completions endpoint."""
                try:
                    data = self._read_json()
                    prompt = data.get("prompt", "")
                    max_tokens = data.get("max_tokens", server.config.max_new_tokens)
                    temperature = data.get("temperature", server.config.temperature)
                    logprobs = data.get("logprobs", 0)
                    echo = data.get("echo", False)

                    # Tokenize prompt
                    if isinstance(prompt, str):
                        prompts = [prompt]
                    else:
                        prompts = prompt

                    results = []
                    for p in prompts:
                        text, tokens, token_logprobs, top_logprobs = server._generate(
                            p, max_tokens, temperature, logprobs
                        )
                        results.append({
                            "text": text,
                            "tokens": tokens,
                            "logprobs": token_logprobs,
                            "top_logprobs": top_logprobs,
                        })

                    response = {
                        "id": f"cmpl-{int(time.time())}",
                        "object": "text_completion",
                        "model": server.model_name,
                        "choices": [{
                            "text": r["text"],
                            "index": i,
                            "logprobs": {
                                "tokens": r["tokens"],
                                "token_logprobs": r["logprobs"],
                                "top_logprobs": r["top_logprobs"],
                            } if logprobs else None,
                            "finish_reason": "length",
                        } for i, r in enumerate(results)],
                    }
                    self._send_json(response)

                except Exception as e:
                    self._send_error(500, str(e))

            def _handle_chat_completions(self):
                """Handle /v1/chat/completions endpoint."""
                try:
                    data = self._read_json()
                    messages = data.get("messages", [])
                    max_tokens = data.get("max_tokens", server.config.max_new_tokens)
                    temperature = data.get("temperature", server.config.temperature)
                    logprobs = data.get("logprobs", False)
                    top_logprobs = data.get("top_logprobs", 0)

                    # Convert messages to prompt
                    prompt = server._format_chat_prompt(messages)

                    # Generate
                    text, tokens, token_logprobs, top_lps = server._generate(
                        prompt, max_tokens, temperature, top_logprobs if logprobs else 0
                    )

                    response = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "model": server.model_name,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": text,
                            },
                            "logprobs": {
                                "content": [
                                    {
                                        "token": t,
                                        "logprob": lp,
                                        "top_logprobs": top_lps[i] if top_lps else [],
                                    }
                                    for i, (t, lp) in enumerate(zip(tokens, token_logprobs))
                                ]
                            } if logprobs else None,
                            "finish_reason": "length",
                        }],
                        "usage": {
                            "prompt_tokens": len(server.tokenizer.encode(prompt)),
                            "completion_tokens": len(tokens),
                            "total_tokens": len(server.tokenizer.encode(prompt)) + len(tokens),
                        },
                    }
                    self._send_json(response)

                except Exception as e:
                    self._send_error(500, str(e))

            def _handle_generate(self):
                """
                Handle /generate endpoint - vLLM native format for Atropos compatibility.
                
                This endpoint uses the vLLM engine for fast generation with KV cache.
                
                Request format (vLLM native):
                {
                    "prompt": {"prompt_token_ids": [1, 2, 3, ...]},  # or string prompt
                    "max_tokens": 512,
                    "temperature": 1.0,
                    "n": 1,
                    "logprobs": 0,  # Number of top logprobs (0 = return 1 logprob per token)
                    ...
                }
                
                Response format (matches vLLM /generate):
                {
                    "text": ["completion1", "completion2", ...],
                    "logprobs": [[[{token_id: logprob}], ...], ...],  # Per sample, per position
                    "finish_reasons": ["stop", "length", ...],
                }
                """
                try:
                    data = self._read_json()
                    
                    # Parse prompt - can be string or {"prompt_token_ids": [...]}
                    prompt_data = data.get("prompt", "")
                    if isinstance(prompt_data, dict):
                        prompt_token_ids = prompt_data.get("prompt_token_ids", [])
                        # Decode token IDs to text for vLLM
                        prompt_text = server.tokenizer.decode(prompt_token_ids)
                    else:
                        prompt_text = prompt_data
                        prompt_token_ids = server.tokenizer.encode(prompt_text)
                    
                    # Parse sampling params (vLLM format - directly in request, not nested)
                    max_tokens = data.get("max_tokens", server.config.max_new_tokens)
                    temperature = data.get("temperature", server.config.temperature)
                    n_samples = data.get("n", 1)
                    stop_sequences = data.get("stop", None)
                    top_p = data.get("top_p", server.config.top_p)
                    
                    # Whether to return logprobs (0 means return 1 logprob per token)
                    logprobs_requested = data.get("logprobs", 0) is not None
                    
                    # Use vLLM engine if available for fast generation
                    if server.vllm_engine is not None:
                        # Use vLLM's fast generation with KV cache
                        (
                            completions,
                            log_probs_tensor,
                            token_ids_list,
                            token_log_probs_list,
                            prompt_token_ids_list,
                            finish_reasons,
                        ) = server.vllm_engine.generate(
                            prompt_texts=[prompt_text],
                            max_new_tokens=max_tokens,
                            temperature=max(temperature, 0.01),  # vLLM doesn't like temp=0
                            n_samples_per_prompt=n_samples,
                            stop=stop_sequences,
                            top_p=top_p,
                        )
                        
                        # Format response in vLLM style
                        # logprobs format: [[[{token_id: logprob}], ...], ...]
                        # - Outer list: one entry per sample
                        # - Middle list: one entry per token position  
                        # - Inner list: contains single dict {token_id: logprob}
                        formatted_logprobs = []
                        for sample_idx, (tids, tlps) in enumerate(zip(token_ids_list, token_log_probs_list)):
                            sample_logprobs = []
                            for tid, tlp in zip(tids, tlps):
                                # Each position is a list containing a single {token_id: logprob} dict
                                sample_logprobs.append([{int(tid): float(tlp)}])
                            formatted_logprobs.append(sample_logprobs)
                        
                        response = {
                            "text": completions,
                            "logprobs": formatted_logprobs,
                            "finish_reasons": finish_reasons,
                        }
                    else:
                        # Fallback to slow PyTorch generation if no vLLM engine
                        all_completions = []
                        all_logprobs = []
                        all_finish_reasons = []
                        
                        for _ in range(n_samples):
                            text, tokens, token_lps, _ = server._generate(
                                prompt_text, max_tokens, temperature, 1 if logprobs_requested else 0
                            )
                            all_completions.append(text)
                            
                            # Get token IDs for the generated tokens
                            gen_token_ids = server.tokenizer.encode(text, add_special_tokens=False)
                            
                            # Format logprobs - each position is [{token_id: logprob}]
                            if logprobs_requested and token_lps:
                                sample_logprobs = []
                                for tid, tlp in zip(gen_token_ids, token_lps):
                                    sample_logprobs.append([{int(tid): float(tlp)}])
                                all_logprobs.append(sample_logprobs)
                            else:
                                all_logprobs.append([])
                            
                            # Finish reason
                            if len(gen_token_ids) >= max_tokens:
                                all_finish_reasons.append("length")
                            else:
                                all_finish_reasons.append("stop")
                        
                        response = {
                            "text": all_completions,
                            "logprobs": all_logprobs,
                            "finish_reasons": all_finish_reasons,
                        }
                    
                    self._send_json(response)
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self._send_error(500, str(e))

            def _read_json(self) -> dict:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                return json.loads(body.decode("utf-8"))

            def _send_json(self, data: dict):
                body = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_error(self, code: int, message: str):
                body = json.dumps({"error": {"message": message, "code": code}}).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def _format_chat_prompt(self, messages: list[dict]) -> str:
        """Convert chat messages to a prompt string."""
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Simple fallback
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts) + "\nassistant:"

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k_logprobs: int = 0,
    ) -> tuple[str, list[str], list[float], list[list[dict]] | None]:
        """
        Generate text from prompt using the training model.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k_logprobs: Number of top logprobs to return (0 = none)

        Returns:
            (generated_text, token_strings, token_logprobs, top_logprobs_list)
        """
        device = next(self.model.parameters()).device

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate token by token
        generated_ids = []
        token_logprobs = []
        top_logprobs_list = [] if top_k_logprobs > 0 else None

        for _ in range(max_tokens):
            # Forward pass
            with torch.no_grad():
                logits = self.model(input_ids)

            # Get logits for last position
            next_logits = logits[0, -1, :]

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Compute log probabilities
            log_probs = F.log_softmax(next_logits.float(), dim=-1)

            # Sample next token
            if temperature > 0:
                probs = torch.exp(log_probs)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(log_probs, dim=-1, keepdim=True)

            next_token_id = next_token.item()
            token_log_prob = log_probs[next_token_id].item()

            generated_ids.append(next_token_id)
            token_logprobs.append(token_log_prob)

            # Get top-k logprobs if requested
            if top_k_logprobs > 0:
                top_values, top_indices = torch.topk(log_probs, top_k_logprobs)
                top_lps = [
                    {
                        "token": self.tokenizer.decode([idx.item()]),
                        "logprob": val.item(),
                    }
                    for val, idx in zip(top_values, top_indices)
                ]
                top_logprobs_list.append(top_lps)

            # Append to input for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

        # Decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_strings = [self.tokenizer.decode([tid]) for tid in generated_ids]

        return generated_text, token_strings, token_logprobs, top_logprobs_list
