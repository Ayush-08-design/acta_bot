import importlib
from pathlib import Path

from langchain_core.tools import BaseTool


class ToolRegistry:
    _instance = None

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if getattr(self , '_initalized' , False):
            return
        
        self._tools : dict[str , BaseTool]= {}
        self.tools_root = Path(__file__).parent.resolve()
        self._discover_builtin_tools()
        
        self._initalized = True
        
    def _discover_builtin_tools(self):
        '''Automatically discovers and import all tools from builtin package'''
        builtin_dir = self.tools_root / 'builtin'
        if not builtin_dir.exists():
            return
        for file in builtin_dir.glob("*.py"):
            if file.name == '__init__.py':
                continue
            module_name = f".builtin.{file.stem}"
            try:
                module = importlib.import_module(module_name , package='src.app.tools')
                for attr_name in dir(module):
                    attr = getattr(module , attr_name)
                    if isinstance(attr , BaseTool):
                        self._tools[attr.name] = attr
            except Exception as e:
                print(f'Error loading builtin module {module_name} : {e}')
                
    def register_mcp_tools(self):
        pass

    def get_tool(self , name : str) -> BaseTool:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        
    def get_all_tools(self) -> list[BaseTool]:
        return list(self._tools.values())
    
    def get_tools_by_names(self , names : list[str]) -> list[BaseTool]:
        return [self._tools[name] for name in names if name in self._tools]
    