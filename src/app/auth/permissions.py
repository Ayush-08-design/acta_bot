from enum import Enum
from typing import Annotated

from fastapi import Depends, HTTPException, status

from src.app.api.deps import get_current_user


class RoleChecker:
    def __init__(self , allowed_roles : list[str]):
        self.allowed_roles = allowed_roles
        self.allowed_roles = [
            r.value if isinstance(r , Enum) else (r) for r in allowed_roles
        ]
        
    def __call__(
        self,
        current_user: Annotated[dict, Depends(get_current_user)],
    ):
        if current_user['role'] not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='You dont have access to this resource'
            )
        return current_user
    
    