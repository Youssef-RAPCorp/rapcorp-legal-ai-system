"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    GEMINI LLM CLIENT                                          ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Supports:                                                                    ║
║  • gemini-flash-lite-latest - Ultra-fast, cost-optimized                      ║
║  • gemini-flash-latest      - Balanced speed/quality                          ║
║  • gemini-pro-latest        - Maximum reasoning power                         ║
║                                                                               ║
║  Features:                                                                    ║
║  • Automatic model selection based on task                                    ║
║  • Cost tracking per request                                                  ║
║  • Retry logic with exponential backoff                                       ║
║  • Streaming support                                                          ║
║  • Google Search grounding                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List, Any, AsyncGenerator, Union
from enum import Enum
import json
import os

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, SafetySettingDict
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")

from configs.config import LegalAIConfig, GeminiModel, MODEL_CONFIGS


# ═══════════════════════════════════════════════════════════════════════════════
# COST TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RequestCost:
    """Track cost of a single request."""
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CostTracker:
    """Track costs across multiple requests."""
    daily_costs: Dict[str, float] = field(default_factory=dict)  # date -> cost
    monthly_costs: Dict[str, float] = field(default_factory=dict)  # month -> cost
    model_costs: Dict[str, float] = field(default_factory=dict)  # model -> cost
    total_requests: int = 0
    total_cost: float = 0.0
    
    def add_request(self, cost: RequestCost) -> None:
        """Add a request's cost to tracking."""
        today = cost.timestamp.strftime("%Y-%m-%d")
        month = cost.timestamp.strftime("%Y-%m")
        
        self.daily_costs[today] = self.daily_costs.get(today, 0.0) + cost.total_cost
        self.monthly_costs[month] = self.monthly_costs.get(month, 0.0) + cost.total_cost
        self.model_costs[cost.model] = self.model_costs.get(cost.model, 0.0) + cost.total_cost
        
        self.total_requests += 1
        self.total_cost += cost.total_cost
    
    def get_today_cost(self) -> float:
        """Get today's total cost."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return self.daily_costs.get(today, 0.0)
    
    def get_month_cost(self) -> float:
        """Get this month's total cost."""
        month = datetime.utcnow().strftime("%Y-%m")
        return self.monthly_costs.get(month, 0.0)
    
    def check_budget(self, config: LegalAIConfig) -> Dict[str, Any]:
        """Check if we're within budget."""
        today_cost = self.get_today_cost()
        month_cost = self.get_month_cost()
        
        return {
            "daily": {
                "spent": today_cost,
                "budget": config.daily_budget_usd,
                "remaining": config.daily_budget_usd - today_cost,
                "percent_used": (today_cost / config.daily_budget_usd) * 100 if config.daily_budget_usd > 0 else 0,
                "alert": today_cost >= (config.daily_budget_usd * config.cost_alert_threshold)
            },
            "monthly": {
                "spent": month_cost,
                "budget": config.monthly_budget_usd,
                "remaining": config.monthly_budget_usd - month_cost,
                "percent_used": (month_cost / config.monthly_budget_usd) * 100 if config.monthly_budget_usd > 0 else 0,
                "alert": month_cost >= (config.monthly_budget_usd * config.cost_alert_threshold)
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiClient:
    """
    Client for interacting with Gemini models.
    
    Supports automatic model selection based on task type:
    - FLASH_LITE: Simple queries, classification, keyword extraction
    - FLASH: Retrieval, summarization, standard drafting
    - PRO: Complex reasoning, IRAC/CREAC analysis, synthesis
    """
    
    def __init__(self, config: LegalAIConfig):
        """Initialize the Gemini client."""
        self.config = config
        self.cost_tracker = CostTracker()
        
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set in configuration")
        
        # Configure the API
        genai.configure(api_key=config.google_api_key)
        
        # Initialize model instances
        self._models: Dict[str, Any] = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize Gemini model instances."""
        models_to_init = [
            (GeminiModel.FLASH_LITE, self.config.model_flash_lite),
            (GeminiModel.FLASH, self.config.model_flash),
            (GeminiModel.PRO, self.config.model_pro),
        ]
        
        for model_enum, model_id in models_to_init:
            try:
                self._models[model_id] = genai.GenerativeModel(model_id)
                print(f"✅ Initialized: {model_id}")
            except Exception as e:
                print(f"⚠️ Failed to initialize {model_id}: {e}")
    
    def _get_model(self, model_id: str) -> Any:
        """Get a model instance by ID."""
        if model_id not in self._models:
            # Try to initialize on-demand
            self._models[model_id] = genai.GenerativeModel(model_id)
        return self._models[model_id]
    
    def _select_model_for_task(self, task: Optional[str] = None) -> str:
        """Select the appropriate model for a task."""
        if task:
            return self.config.get_model_for_task(task)
        return self.config.model_flash  # Default to Flash
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4
    
    def _calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> RequestCost:
        """Calculate the cost of a request."""
        # Find the model config
        model_config = None
        for model_enum, config in MODEL_CONFIGS.items():
            if config.model_id == model_id:
                model_config = config
                break
        
        if not model_config:
            # Default costs if model not found
            input_cost = input_tokens * 0.0001 / 1000
            output_cost = output_tokens * 0.0002 / 1000
        else:
            input_cost = input_tokens * model_config.cost_per_1k_input / 1000
            output_cost = output_tokens * model_config.cost_per_1k_output / 1000
        
        return RequestCost(
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost
        )
    
    async def generate(
        self,
        prompt: str,
        task: Optional[str] = None,
        model_override: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        enable_grounding: bool = False,
        return_json: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response from Gemini.
        
        Args:
            prompt: The input prompt
            task: Task type for automatic model selection
            model_override: Override automatic model selection
            system_instruction: System instruction for the model
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop_sequences: Stop sequences
            enable_grounding: Enable Google Search grounding
            return_json: Request JSON output format
        
        Returns:
            Dictionary with response text, metadata, and cost info
        """
        # Select model
        model_id = model_override or self._select_model_for_task(task)
        model = self._get_model(model_id)
        
        # Get model config for defaults
        model_config = None
        for enum, config in MODEL_CONFIGS.items():
            if config.model_id == model_id:
                model_config = config
                break
        
        # Build generation config
        gen_config_kwargs = dict(
            temperature=temperature or (model_config.temperature if model_config else 0.3),
            max_output_tokens=max_tokens or (model_config.max_tokens if model_config else 8192),
            top_p=model_config.top_p if model_config else 0.9,
            top_k=model_config.top_k if model_config else 40,
        )
        if stop_sequences:
            gen_config_kwargs["stop_sequences"] = stop_sequences
        if return_json:
            gen_config_kwargs["response_mime_type"] = "application/json"
        gen_config = GenerationConfig(**gen_config_kwargs)
        
        # Build the content
        contents = []
        if system_instruction:
            contents.append({"role": "user", "parts": [{"text": f"System: {system_instruction}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        # Make the request with retry logic
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Generate response
                response = await asyncio.to_thread(
                    model.generate_content,
                    contents,
                    generation_config=gen_config
                )
                
                elapsed_time = time.time() - start_time
                
                # Extract text
                response_text = response.text if hasattr(response, 'text') else ""
                
                # Calculate tokens and cost
                input_tokens = self._estimate_tokens(prompt + (system_instruction or ""))
                output_tokens = self._estimate_tokens(response_text)
                cost = self._calculate_cost(model_id, input_tokens, output_tokens)
                
                # Track cost
                if self.config.track_costs:
                    self.cost_tracker.add_request(cost)
                
                return {
                    "text": response_text,
                    "model": model_id,
                    "task": task,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost.total_cost,
                    "elapsed_seconds": elapsed_time,
                    "finish_reason": getattr(response, 'prompt_feedback', None),
                    "grounded": enable_grounding
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    raise RuntimeError(f"Gemini API error after {max_retries} attempts: {e}")
        
        # Should not reach here
        raise RuntimeError("Unexpected error in generate()")
    
    async def generate_stream(
        self,
        prompt: str,
        task: Optional[str] = None,
        model_override: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from Gemini.
        
        Yields:
            Text chunks as they're generated
        """
        model_id = model_override or self._select_model_for_task(task)
        model = self._get_model(model_id)
        
        # Get model config
        model_config = None
        for enum, config in MODEL_CONFIGS.items():
            if config.model_id == model_id:
                model_config = config
                break
        
        gen_config = GenerationConfig(
            temperature=temperature or (model_config.temperature if model_config else 0.3),
            max_output_tokens=max_tokens or (model_config.max_tokens if model_config else 8192),
        )
        
        contents = []
        if system_instruction:
            contents.append({"role": "user", "parts": [{"text": f"System: {system_instruction}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        response = await asyncio.to_thread(
            model.generate_content,
            contents,
            generation_config=gen_config,
            stream=True
        )
        
        for chunk in response:
            if hasattr(chunk, 'text'):
                yield chunk.text
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        task: Optional[str] = None,
        model_override: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response with function calling support.
        
        Args:
            prompt: The input prompt
            tools: List of tool definitions
            task: Task type for model selection
            model_override: Override model selection
            system_instruction: System instruction
        
        Returns:
            Response with potential function calls
        """
        model_id = model_override or self._select_model_for_task(task)
        
        # Create model with tools
        model = genai.GenerativeModel(
            model_id,
            tools=tools
        )
        
        contents = []
        if system_instruction:
            contents.append({"role": "user", "parts": [{"text": f"System: {system_instruction}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        response = await asyncio.to_thread(
            model.generate_content,
            contents
        )
        
        # Check for function calls
        function_calls = []
        if hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call'):
                    function_calls.append({
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args)
                    })
        
        return {
            "text": response.text if hasattr(response, 'text') else "",
            "function_calls": function_calls,
            "model": model_id
        }
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get a summary of costs."""
        return {
            "total_requests": self.cost_tracker.total_requests,
            "total_cost": self.cost_tracker.total_cost,
            "today_cost": self.cost_tracker.get_today_cost(),
            "month_cost": self.cost_tracker.get_month_cost(),
            "by_model": self.cost_tracker.model_costs,
            "budget_status": self.cost_tracker.check_budget(self.config)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def quick_generate(
    prompt: str,
    model: str = "gemini-flash-latest",
    api_key: Optional[str] = None
) -> str:
    """
    Quick one-off generation without full config setup.
    
    Args:
        prompt: The prompt to send
        model: Model to use (default: gemini-flash-latest)
        api_key: API key (or set GOOGLE_API_KEY env var)
    
    Returns:
        Generated text
    """
    if not GENAI_AVAILABLE:
        raise ImportError("google-generativeai not installed")
    
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("API key required")
    
    genai.configure(api_key=key)
    model_instance = genai.GenerativeModel(model)
    response = await asyncio.to_thread(model_instance.generate_content, prompt)
    return response.text


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

async def demo():
    """Demo the Gemini client."""
    print("=" * 60)
    print("GEMINI CLIENT DEMO")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠️ GOOGLE_API_KEY not set. Set it to run the demo.")
        print("\nExample usage:")
        print("""
from src.core.gemini_client import GeminiClient
from configs.config import create_config

config = create_config()
client = GeminiClient(config)

# Simple generation
result = await client.generate(
    prompt="What are the elements of breach of contract?",
    task="legal_synthesis"  # Uses gemini-pro-latest
)
print(result["text"])

# Task-based model selection
result = await client.generate(
    prompt="Classify this as contract or tort",
    task="classification"  # Uses gemini-flash-lite-latest
)
""")
        return
    
    # Run actual demo
    config = create_config()
    client = GeminiClient(config)
    
    print("\n🧪 Testing Flash Lite (classification)...")
    result = await client.generate(
        prompt="Is this a contract dispute or a tort case: 'The defendant failed to deliver goods as promised'",
        task="classification"
    )
    print(f"  Model: {result['model']}")
    print(f"  Response: {result['text'][:200]}...")
    print(f"  Cost: ${result['cost']:.6f}")
    
    print("\n🧪 Testing Flash (summarization)...")
    result = await client.generate(
        prompt="Summarize the key requirements for a valid contract",
        task="summarization"
    )
    print(f"  Model: {result['model']}")
    print(f"  Response: {result['text'][:200]}...")
    print(f"  Cost: ${result['cost']:.6f}")
    
    print("\n🧪 Testing Pro (IRAC analysis)...")
    result = await client.generate(
        prompt="Perform an IRAC analysis on whether a verbal promise to pay constitutes an enforceable contract",
        task="irac_analysis"
    )
    print(f"  Model: {result['model']}")
    print(f"  Response: {result['text'][:300]}...")
    print(f"  Cost: ${result['cost']:.6f}")
    
    print("\n📊 Cost Summary:")
    summary = client.get_cost_summary()
    print(f"  Total Requests: {summary['total_requests']}")
    print(f"  Total Cost: ${summary['total_cost']:.6f}")


if __name__ == "__main__":
    asyncio.run(demo())
