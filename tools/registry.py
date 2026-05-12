
from typing import Callable, Dict, List, Optional, Any
from functools import wraps


class ToolRegistry:
    
    _tools: Dict[str, Dict[str, Any]] = {}
    _langchain_tools: Dict[str, Any] = {}
    
    @classmethod
    def register(
        cls, 
        name: str, 
        func: Callable, 
        description: str = "",
        category: str = "general"
    ) -> None:
        cls._tools[name] = {
            "func": func,
            "description": description,
            "category": category,
            "name": name,
        }
        print(f"[ToolRegistry] 注册工具: {name}")
    
    @classmethod
    def register_langchain_tool(cls, name: str, tool: Any) -> None:
        cls._langchain_tools[name] = tool
        print(f"[ToolRegistry] 注册 LangChain 工具: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        tool_info = cls._tools.get(name)
        if tool_info:
            return tool_info["func"]
        return None
    
    @classmethod
    def get_langchain_tool(cls, name: str) -> Optional[Any]:
        return cls._langchain_tools.get(name)
    
    @classmethod
    def get_info(cls, name: str) -> Optional[Dict]:
        return cls._tools.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, Dict[str, Any]]:
        return cls._tools.copy()
    
    @classmethod
    def get_all_langchain_tools(cls) -> List[Any]:
        return list(cls._langchain_tools.values())
    
    @classmethod
    def get_by_category(cls, category: str) -> Dict[str, Dict[str, Any]]:
        return {
            name: info 
            for name, info in cls._tools.items() 
            if info.get("category") == category
        }
    
    @classmethod
    def list_tools(cls) -> List[str]:
        return list(cls._tools.keys())
    
    @classmethod
    def clear(cls) -> None:
        cls._tools.clear()
        cls._langchain_tools.clear()


def register_tool(name: str, description: str = "", category: str = "general"):
    def decorator(func: Callable) -> Callable:
        ToolRegistry.register(name, func, description, category)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_all_tools() -> List[Any]:
    return ToolRegistry.get_all_langchain_tools()


def get_tool(name: str) -> Optional[Any]:
    return ToolRegistry.get_langchain_tool(name)

