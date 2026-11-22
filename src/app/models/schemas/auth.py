from pydantic import BaseModel, Field, field_validator


class UserAuth(BaseModel):
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="Alphanumeric username (3-50 chars; letters, numbers, _, ., -).",
        examples=["rahul_sharma"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,  # Prevents bcrypt/Argon2 DoS attacks with huge payloads
        description="Password between 8 and 128 characters.",
        examples=["StrongPass123!"],
    )

    @field_validator("username")
    @classmethod
    def normalize_username(cls, v: str) -> str:
        return v.strip().lower()


class TokenRefresh(BaseModel):
    refresh_token: str = Field(
        ...,
        min_length=20,
        description="Valid refresh token issued during login.",
    )

    @field_validator("refresh_token")
    @classmethod
    def strip_token(cls, v: str) -> str:
        token = v.strip()
        if not token:
            raise ValueError("Refresh token cannot be empty")
        return token


class TokenResponse(BaseModel):
    access_token: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Access token for the client.",
    )
    refresh_token: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Refresh token for the client.",
    )
    token_type: str = Field(default="bearer")
