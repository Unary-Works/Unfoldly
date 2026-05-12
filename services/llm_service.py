import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, AsyncGenerator, Generator
from openai import OpenAI, AsyncOpenAI
import httpx
from config import settings
from utils.logger import get_logger
logger = get_logger()

_NO_SYSTEM_ROLE_MODELS = {"gemma", "gemma2", "gemma3", "gemma-3"}

_model_system_role_cache = {}


def _model_supports_system_role() -> bool:
    try:
        from services.preference_manager import PreferenceManager
        pref_manager = PreferenceManager()
        current_model_id = pref_manager.get_selected_model_id()
        
        if current_model_id:
            if current_model_id in _model_system_role_cache:
                return _model_system_role_cache[current_model_id]
            
            model_id_lower = current_model_id.lower()
            for pattern in _NO_SYSTEM_ROLE_MODELS:
                if pattern in model_id_lower:
                    _model_system_role_cache[current_model_id] = False
                    return False
            
            _model_system_role_cache[current_model_id] = True
            return True
    except Exception as e:
        logger.error(f"_model_supports_system_role error: {e}")
    return True


class LLMService:
    
    def __init__(self):
        self.service_type = "llm"
    
    def generate(self, message: str, history: List[Dict] = None) -> str:
        raise NotImplementedError
    
    def generate_stream(self, message: str, history: List[Dict] = None) -> Generator[str, None, None]:
        raise NotImplementedError


class RemoteLLMService(LLMService):
    
    def __init__(self, api_key: str = None, model: str = None, system_prompt: str = None):
        super().__init__()
        self.service_type = "remote_llm"
        self.api_key = api_key or settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL
        self.system_prompt = system_prompt
        
        if not self.api_key or self.api_key == "your_llm_api_key":
            raise ValueError(
                "❌ LLM_API_KEY 未配置。仅启用远程 LLM 模式时才需要通过环境变量设置。"
            )
        
        http_client = httpx.Client(proxy=None, timeout=60.0)
        async_http_client = httpx.AsyncClient(proxy=None, timeout=60.0)
        
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            http_client=http_client
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            http_client=async_http_client
        )
    
    def generate(self, message: str, history: List[Dict] = None, system_prompt: str = None) -> str:
        messages = self._build_messages(message, history, system_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return ""
    
    def generate_stream(self, message: str, history: List[Dict] = None, system_prompt: str = None) -> Generator[str, None, None]:
        messages = self._build_messages(message, history, system_prompt)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM Stream Error: {e}")
            yield f"[Error: {e}]"
    
    async def generate_async(self, message: str, history: List[Dict] = None, system_prompt: str = None) -> str:
        messages = self._build_messages(message, history, system_prompt)
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Async Error: {e}")
            return ""
    
    async def generate_stream_async(self, message: str, history: List[Dict] = None, system_prompt: str = None) -> AsyncGenerator[str, None]:
        messages = self._build_messages(message, history, system_prompt)
        
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM Async Stream Error: {e}")
            yield f"[Error: {e}]"
    
    def _build_messages(self, message: str, history: List[Dict] = None, system_prompt: str = None) -> List[Dict]:
        messages = []
        prompt = system_prompt or self.system_prompt
        supports_system = _model_supports_system_role()
        
        valid_pairs = []
        if history:
            _MAX_HIST_A_CHARS = 800
            i = 0
            hist = history[-10:]
            while i < len(hist):
                item = hist[i]
                role = item.get("role", "user")
                content = (item.get("content") or "").strip()
                
                if role == "user" and content:
                    user_content = content
                    assistant_content = ""
                    if i + 1 < len(hist):
                        next_item = hist[i + 1]
                        if next_item.get("role") == "assistant":
                            assistant_content = (next_item.get("content") or "").strip()
                            if len(assistant_content) > _MAX_HIST_A_CHARS:
                                assistant_content = assistant_content[:_MAX_HIST_A_CHARS] + "...[truncated]"
                            i += 1
                    if user_content and assistant_content:
                        valid_pairs.append((user_content, assistant_content))
                i += 1
        
        if supports_system:
            if prompt:
                messages.append({"role": "system", "content": prompt})
            
            for user_content, assistant_content in valid_pairs:
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": assistant_content})
            
            messages.append({"role": "user", "content": message})
        else:
            sys_prefix = f"[Instructions]\n{prompt}\n\n[User Question]\n" if prompt else ""
            
            if valid_pairs:
                first_user, first_assistant = valid_pairs[0]
                messages.append({"role": "user", "content": sys_prefix + first_user})
                messages.append({"role": "assistant", "content": first_assistant})
                
                for user_content, assistant_content in valid_pairs[1:]:
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": assistant_content})
                
                messages.append({"role": "user", "content": message})
            else:
                messages.append({"role": "user", "content": sys_prefix + message})
        
        return messages
