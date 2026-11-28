import datetime
import uuid
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

from src.app.api.deps import CurrentUser, DbSession
from src.app.api.middleware import limiter
from src.app.auth.jwt_handler import PasswordManager
from src.app.core.config import get_settings
from src.app.db.repositories.user import (
    create_user,
    get_user_by_username,
    update_username,
)
from src.app.models.schemas.auth import TokenRefresh, UserAuth, UserUpdate

router = APIRouter(prefix="/auth", tags=["auth"])

settings = get_settings()

# AUTH configuration
SECRET_KEY = settings.AUTH.SECRET_KEY.get_secret_value()
ALGORITHM = settings.AUTH.ALGORITHM.get_secret_value()
ACCESS_TOKEN_EXPIRE_MINUTES = settings.AUTH.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = settings.AUTH.REFRESH_TOKEN_EXPIRE_DAYS

# Password manager
pwd_context = PasswordManager()


# Auth endpoints with rate limiting
@router.post("/register")
@limiter.limit("10/minute")  # Limits registrations to 10 per minute per ip
async def register(request: Request, user: UserAuth, db: DbSession):
    result = await get_user_by_username(db, user.username)
    if result:
        raise HTTPException(status_code=400, detail="user already exist")
    user_id = uuid.uuid4()
    hashed_pass = pwd_context.hash(user.password)
    try:
        result = await create_user(db, user.username, hashed_pass, user_id)
        if result:
            return {
                "message": f"Registration successful {result.username}",
                "tokens": create_access_token(user_id),
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create the account, try again later.",
        ) from e


def create_access_token(user_id: uuid.UUID):
    # Generate access token (15 mins)
    access_expire = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    access_payload = {"sub": str(user_id), "exp": access_expire, "type": "access"}
    access_token = jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)

    # Generate a refresh token (7 days)
    refresh_expire = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
        days=REFRESH_TOKEN_EXPIRE_DAYS
    )
    refresh_payload = {"sub": str(user_id), "exp": refresh_expire, "type": "refresh"}
    refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.post("/login")
@limiter.limit("10/minute")
async def login(
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DbSession,
):
    db_user = await get_user_by_username(db, form_data.username)

    if not db_user or not pwd_context.verify(form_data.password, db_user.hash_pass):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
        )

    # Generate tokens [access and refresh]
    return create_access_token(db_user.user_id)


@router.post("/refresh")
async def refresh_token(token_request: TokenRefresh):
    try:
        payload = jwt.decode(
            token_request.refresh_token, SECRET_KEY, algorithms=[ALGORITHM]
        )
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload.get("sub")

        # Generate a new access token
        access_expire = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
        new_access_payload = {"sub": user_id, "exp": access_expire, "type": "access"}
        new_access_token = jwt.encode(
            new_access_payload, SECRET_KEY, algorithm=ALGORITHM
        )

        return {"access_token": new_access_token, "token_type": "bearer"}

    except ExpiredSignatureError:
        raise HTTPException(
            status_code=401, detail="refresh token expired, please log in again"
        )
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="invalid refresh token")


@router.get("/profile")
async def show_profile(user: CurrentUser):
    return {"message": "welcome to your profile", "username": user.username}


@router.patch("/username")
async def change_username(
    payload: UserUpdate,
    db: DbSession,
    user: CurrentUser,
):
    changed = await update_username(db, user.username, payload.new_username)
    if changed:
        return {"username": payload.new_username}
    return {"message", f"someone already has this username [{payload.new_username}]"}
