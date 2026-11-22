from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

# Reusable Types

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ThreadIdStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_-]+$",  # Prevents path traversal / SQL/NoSQL injection in IDs
    ),
]


# Incoming Request Schemas


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ] = Field(..., description="User prompt or question.")
    thread_id: ThreadIdStr = Field(
        ..., description="Unique LangGraph checkpoint thread ID."
    )
    model_choice: Literal["fast-cheap", "reasoning", "coder", "auto"] = Field(
        default="fast-cheap",
        description="Routing profile for LLM selection.",
    )

    thinking: bool = False
    web_search: bool = False


class HumanApprovalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: ThreadIdStr
    approval: bool
    feedback: (
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def require_feedback_on_rejection(self) -> "HumanApprovalPayload":
        # This Ensures LangGraph interrupt resume nodes get a reason when an action is rejected
        if not self.approval and not self.feedback:
            raise ValueError(
                "Feedback is required when rejecting an action (approval=False)."
            )
        return self


class UpdateTitlePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: ThreadIdStr
    new_title: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
    ]


class FeedbackPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: ThreadIdStr | None = None  # Useful for tracing
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]
    response: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)
    ]
    feedback: bool = Field(
        ..., description="True for thumbs up, False for thumbs down."
    )
    comment: (
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] | None
    ) = None


# --- 2. Outgoing Response Schemas ---


class ChatThreadMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SidebarThreadsResponse(BaseModel):
    items: list[ChatThreadMetadata] = Field(default_factory=list)
    next_page: int | None = Field(default=None, ge=1)
    has_more: bool

    @model_validator(mode="after")
    def check_pagination_consistency(self) -> "SidebarThreadsResponse":
        if not self.has_more:
            self.next_page = None
        return self
