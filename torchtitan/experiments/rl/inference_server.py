"""
HTTP inference server that wraps the TorchTitan training model.

This provides an OpenAI-compatible API that uses the SAME model instance
as training - true shared memory, always on-policy.

Usage:
    # In your training script:
    from torchtitan.experiments.rl.inference_server import InferenceServer
    
    model = Qwen3VLLMCompatModel(...)  # Your training model
    server = InferenceServer(model, tokenizer, port=9000)
    server.start()  # Starts in background thread
    
    # Train...
    # Model weights update automatically (same instance!)
    
    server.stop()
"""

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list = None


class InferenceServer:
    """
    HTTP server exposing the training model for inference.
    
    Provides OpenAI-compatible /v1/completions endpoint.
    Uses the SAME model instance as training - no weight copying!
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        host: str = "0.0.0.0",
        port: int = 9000,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self._server = None
        self._thread = None
        self._running = False
        
        # Create request handler with model reference
        self._handler = self._create_handler()
    
    def _create_handler(self):
        """Create HTTP request handler with access to model."""
        model = self.model
        tokenizer = self.tokenizer
        
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logs
            
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok"}).encode())
                else:
                    self.send_error(404)
            
            def do_POST(self):
                if self.path == "/v1/completions":
                    self._handle_completions()
                elif self.path == "/v1/chat/completions":
                    self._handle_chat_completions()
                else:
                    self.send_error(404)
            
            def _handle_completions(self):
                """Handle /v1/completions (OpenAI-compatible)."""
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                request = json.loads(body)
                
                prompt = request.get("prompt", "")
                max_tokens = request.get("max_tokens", 100)
                temperature = request.get("temperature", 1.0)
                echo = request.get("echo", False)
                logprobs = request.get("logprobs")
                
                # Tokenize
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                input_ids = input_ids.to(next(model.parameters()).device)
                
                # Generate
                with torch.no_grad():
                    generated_ids, generated_logprobs = self._generate(
                        input_ids, max_tokens, temperature, logprobs
                    )
                
                # Decode
                generated_text = tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                
                # Build response
                response = {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": "torchtitan",
                    "choices": [{
                        "text": generated_text if not echo else prompt + generated_text,
                        "index": 0,
                        "finish_reason": "length",
                    }],
                }
                
                # Add logprobs if requested
                if logprobs is not None and generated_logprobs:
                    response["choices"][0]["logprobs"] = {
                        "tokens": [tokenizer.decode([t]) for t in generated_ids[0].tolist()],
                        "token_logprobs": generated_logprobs,
                        "top_logprobs": None,  # Could add top-k here
                    }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            
            def _handle_chat_completions(self):
                """Handle /v1/chat/completions (OpenAI-compatible)."""
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                request = json.loads(body)
                
                messages = request.get("messages", [])
                max_tokens = request.get("max_tokens", 100)
                temperature = request.get("temperature", 1.0)
                
                # Convert messages to prompt
                prompt = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    prompt += f"{role}: {content}\n"
                prompt += "assistant: "
                
                # Tokenize
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                input_ids = input_ids.to(next(model.parameters()).device)
                
                # Generate
                with torch.no_grad():
                    generated_ids, _ = self._generate(
                        input_ids, max_tokens, temperature, None
                    )
                
                # Decode (only new tokens)
                new_tokens = generated_ids[0, input_ids.shape[1]:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                response = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "torchtitan",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": generated_text},
                        "finish_reason": "length",
                    }],
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            
            def _generate(self, input_ids, max_tokens, temperature, logprobs_k):
                """Generate tokens using the training model."""
                generated = input_ids.clone()
                all_logprobs = []
                
                for _ in range(max_tokens):
                    # Forward pass
                    logits = model(generated)
                    next_logits = logits[:, -1, :]
                    
                    # Apply temperature
                    if temperature > 0:
                        next_logits = next_logits / temperature
                    
                    # Sample
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Get logprob of selected token
                    if logprobs_k is not None:
                        log_probs = F.log_softmax(next_logits, dim=-1)
                        token_logprob = log_probs[0, next_token[0, 0]].item()
                        all_logprobs.append(token_logprob)
                    
                    # Append
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    # Check for EOS
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                return generated, all_logprobs
        
        return Handler
    
    def start(self):
        """Start the inference server in a background thread."""
        if self._running:
            return
        
        self._server = HTTPServer((self.host, self.port), self._handler)
        self._running = True
        
        def serve():
            print(f"Inference server started at http://{self.host}:{self.port}")
            while self._running:
                self._server.handle_request()
        
        self._thread = threading.Thread(target=serve, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the inference server."""
        self._running = False
        if self._server:
            self._server.shutdown()
