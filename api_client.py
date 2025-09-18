"""
API client module for Nebius API integration.
Provides a unified interface for both Ollama and Nebius API calls.
"""

import os
import base64
import json
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import io
from PIL import Image

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class APIClient:
    """Unified API client that can use either Ollama or Nebius API."""
    
    def __init__(self, api_type: str = "auto", api_key: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_type: "ollama", "nebius", or "auto" (auto-detect based on availability)
            api_key: Optional API key for Nebius. If provided, it overrides environment.
        """
        self.api_type = api_type
        self.api_key = api_key
        self.nebius_client = None
        self.ollama_available = OLLAMA_AVAILABLE
        
        # Try to load environment from .env if present (no external deps)
        self._load_dotenv_if_present()
        
        # Auto-detect if not specified
        if api_type == "auto":
            if OPENAI_AVAILABLE and (self.api_key or os.getenv("NEBIUS_API_KEY")):
                self.api_type = "nebius"
            elif OLLAMA_AVAILABLE:
                self.api_type = "ollama"
            else:
                raise RuntimeError("Neither Nebius API nor Ollama is available. Please install required packages and set NEBIUS_API_KEY.")
        
        # Initialize Nebius client if needed
        if self.api_type == "nebius":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI package not available. Install with: pip install openai")
            
            api_key = self.api_key or os.getenv("NEBIUS_API_KEY")
            if not api_key:
                raise RuntimeError("NEBIUS_API_KEY environment variable not set")
            
            self.nebius_client = OpenAI(
                base_url="https://api.studio.nebius.com/v1/",
                api_key=api_key
            )

    def _load_dotenv_if_present(self) -> None:
        """Lightweight .env loader: reads .env and sets env vars if not set."""
        try:
            from pathlib import Path as _Path
            candidates = [
                _Path.cwd() / ".env",
                _Path(__file__).parent.resolve() / ".env",
                _Path(__file__).parent.parent.resolve() / ".env",
            ]
            for p in candidates:
                if p.exists():
                    for raw in p.read_text(encoding="utf-8").splitlines():
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        key = k.strip()
                        val = v.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = val
                    break
        except Exception:
            # Silently ignore .env loading issues
            pass
    
    def _prepare_image_b64_for_api(self, image_path: Union[str, Path], max_side: int = 512) -> str:
        """Load image, downscale longest side to <= max_side, encode as JPEG base64."""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        scale = 1.0
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    
    def _encode_image_to_b64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request using the configured API.
        
        Args:
            messages: List of message dictionaries
            model: Model name (optional, uses defaults if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            format: Response format (e.g., "json")
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary
        """
        if self.api_type == "nebius":
            return self._nebius_chat_completion(
                messages, model, temperature, max_tokens, format, **kwargs
            )
        elif self.api_type == "ollama":
            return self._ollama_chat_completion(
                messages, model, temperature, max_tokens, format, **kwargs
            )
        else:
            raise RuntimeError(f"Unknown API type: {self.api_type}")
    
    def _nebius_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send chat completion request to Nebius API."""
        if model is None:
            model = "Qwen/Qwen2.5-VL-72B-Instruct"
        
        # Convert messages to Nebius format
        nebius_messages = []
        for msg in messages:
            if msg["role"] == "system":
                nebius_messages.append({
                    "role": "system",
                    "content": msg["content"]
                })
            elif msg["role"] == "user":
                content = msg["content"]
                images = msg.get("images", [])
                
                if images:
                    # Multi-modal message
                    content_parts = [{"type": "text", "text": content}]
                    for img_b64 in images:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        })
                    nebius_messages.append({
                        "role": "user",
                        "content": content_parts
                    })
                else:
                    # Text-only message
                    nebius_messages.append({
                        "role": "user",
                        "content": content
                    })
            elif msg["role"] == "assistant":
                nebius_messages.append({
                    "role": "assistant",
                    "content": msg["content"]
                })
        
        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": nebius_messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            request_params["max_tokens"] = max_tokens
        
        # Send request
        response = self.nebius_client.chat.completions.create(**request_params)
        
        # Convert response to Ollama-like format
        return {
            "message": {
                "content": response.choices[0].message.content
            }
        }
    
    def _ollama_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send chat completion request to Ollama."""
        if model is None:
            model = "llava:13b"
        
        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.1),
                "num_predict": max_tokens or kwargs.get("num_predict", 1024),
            }
        }
        
        if format:
            request_params["format"] = format
        
        if "keep_alive" in kwargs:
            request_params["keep_alive"] = kwargs["keep_alive"]
        
        # Send request
        try:
            return ollama.chat(**request_params)
        except Exception as e:
            msg = str(e)
            if "model \"llava\" not found" in msg or "model \"llava:13b\" not found" in msg or "status code: 404" in msg:
                raise RuntimeError("Ollama model not found. Pull it first: 'ollama pull llava:13b'. Also ensure 'ollama serve' is running.")
            raise
    
    def classify_object(
        self,
        object_path: Union[str, Path],
        system_instructions: str,
        user_prompt: str,
        max_seconds: int = 35,
        max_side: int = 512
    ) -> tuple[str, str]:
        """
        Classify an object using the configured API.
        
        Args:
            object_path: Path to the object image
            system_instructions: System prompt
            user_prompt: User prompt
            max_seconds: Maximum time to wait
            max_side: Maximum image side length
            
        Returns:
            Tuple of (predicted_class, label)
        """
        # Prepare image
        image_b64 = self._prepare_image_b64_for_api(object_path, max_side)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt, "images": [image_b64]}
        ]
        
        # Set model-specific parameters
        if self.api_type == "nebius":
            model = "Qwen/Qwen2.5-VL-72B-Instruct"
            temperature = 0.0
            max_tokens = 120000
        else:  # ollama
            model = "llava:13b"
            temperature = 0
            max_tokens = 12000 #llava might have smaller max token
        
        # Make request with timeout handling
        started = time.time()
        try:
            response = self.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                format="json" if self.api_type == "ollama" else None
            )
            
            content = response.get("message", {}).get("content", "").strip()
            elapsed = time.time() - started
            
            if elapsed > max_seconds:
                print(f"  .. took {elapsed:.1f}s (longer than {max_seconds}s timeout)", flush=True)
            
            return self._parse_classification_response(content)
            
        except Exception as e:
            print(f"  .. API error on {Path(object_path).name}: {e}", flush=True)
            return "unknown", "unknown"
    
    def _parse_classification_response(self, content: str) -> tuple[str, str]:
        """Parse classification response and extract class and label."""
        import re
        
        try:
            # Try direct JSON parsing
            data = json.loads(content)
            predicted_class = str(data.get("class", "unknown")).strip().lower()
            label = str(data.get("label", "unknown")).strip()
            return predicted_class, label
        except Exception:
            # Try to find JSON within the text
            json_match = re.search(r'\{[^}]*"class"[^}]*"label"[^}]*\}', content)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    predicted_class = str(data.get("class", "unknown")).strip().lower()
                    label = str(data.get("label", "unknown")).strip()
                    return predicted_class, label
                except Exception:
                    pass
        
        return "unknown", "unknown"
    
    def analyze_image_for_objects(
        self,
        image_path: Union[str, Path],
        prompt: str,
        model: Optional[str] = None
    ) -> str:
        """
        Analyze an image to get a list of objects.
        
        Args:
            image_path: Path to the image
            prompt: Analysis prompt
            model: Model name (optional)
            
        Returns:
            Raw response text
        """
        # Prepare image
        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Prepare messages
        messages = [{"role": "user", "content": prompt, "images": [image_b64]}]
        
        # Set model
        if model is None:
            model = "Qwen/Qwen2.5-VL-72B-Instruct" if self.api_type == "nebius" else "llava:13b"
        
        # Make request
        response = self.chat_completion(messages=messages, model=model)
        return response["message"]["content"]
    
    def generate_layout(
        self,
        contact_sheet: Image.Image,
        background_path: Union[str, Path],
        results_json_path: Union[str, Path],
        ratio: str,
        prompt: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate layout using the configured API.
        
        Args:
            contact_sheet: Contact sheet image
            background_path: Path to background image
            results_json_path: Path to results JSON
            ratio: Aspect ratio
            prompt: Layout generation prompt
            model: Model name (optional)
            
        Returns:
            Layout data dictionary
        """
        # Prepare images
        contact_b64 = self._encode_image_to_b64(contact_sheet)
        with open(background_path, "rb") as f:
            background_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a JSON generator. Output ONLY valid JSON matching the exact schema provided. No markdown, no explanations, no extra text."},
            {"role": "user", "content": prompt, "images": [contact_b64, background_b64]}
        ]
        
        # Set model
        if model is None:
            model = "Qwen/Qwen2.5-VL-72B-Instruct" if self.api_type == "nebius" else "llava:13b"
        
        # Make request
        response = self.chat_completion(messages=messages, model=model)
        content = response["message"]["content"].strip()
        
        # Parse JSON response
        return self._extract_json_from_content(content)
    
    def critique_layout(
        self,
        image_path: Union[str, Path],
        prompt: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Critique a layout using the configured API.
        
        Args:
            image_path: Path to layout image
            prompt: Critique prompt
            model: Model name (optional)
            
        Returns:
            Critique data dictionary
        """
        # Prepare image
        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Prepare messages
        messages = [{"role": "user", "content": prompt, "images": [image_b64]}]
        
        # Set model
        if model is None:
            model = "Qwen/Qwen2.5-VL-72B-Instruct" if self.api_type == "nebius" else "llava:13b"
        
        # Make request
        response = self.chat_completion(messages=messages, model=model)
        content = response["message"]["content"]
        
        # Parse JSON response
        return self._extract_json_from_content(content)
    
    def translate_critique(
        self,
        critique: str,
        current_params: Dict[str, Any],
        prompt: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate critique to parameters using the configured API.
        
        Args:
            critique: Critique text
            current_params: Current parameters
            prompt: Translation prompt
            model: Model name (optional)
            
        Returns:
            Updated parameters dictionary
        """
        # Prepare messages
        messages = [{"role": "user", "content": prompt}]
        
        # Set model
        if model is None:
            # On Nebius, prefer a strong text-only LLM for translation
            model = "openai/gpt-oss-20b" if self.api_type == "nebius" else "mistral:7b"
        
        # Make request
        response = self.chat_completion(messages=messages, model=model)
        content = response["message"]["content"]
        
        # Parse JSON response
        return self._extract_json_from_content(content)
    
    def _extract_json_from_content(self, content: str) -> Dict[str, Any]:
        """Extract JSON from response content."""
        import re
        
        # Try direct parsing first
        try:
            return json.loads(content.strip())
        except Exception:
            pass
        
        # Try to find JSON within the text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass
        
        # Fallback: return error response
        return {"error": "Could not parse JSON from response", "content": content}


# Global API clients cache (keyed by api_type and api_key)
_api_clients: dict[str, APIClient] = {}

def get_api_client(api_type: str = "auto", api_key: Optional[str] = None) -> APIClient:
    """Get or create a cached API client instance keyed by (api_type, api_key)."""
    global _api_clients
    cache_key = f"{api_type}:{api_key or ''}"
    client = _api_clients.get(cache_key)
    if client is None:
        client = APIClient(api_type=api_type, api_key=api_key)
        _api_clients[cache_key] = client
    return client
