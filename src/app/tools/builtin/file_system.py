from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.app.files.download import (
    ALLOWED_EXTENSIONS,
    URL_TTL_SECONDS,
    WORKSPACE_DIR,
    create_download_url,
)


class WriteFileInput(BaseModel):
    filename: str = Field(
        description="Relative file path inside workspace (e.g., 'report.md')."
    )
    content: str = Field(description="Text content to write to the file.")


@tool("write_file", args_schema=WriteFileInput, response_format="content_and_artifact")
def write_file(filename: str, content: str, config: RunnableConfig) -> tuple[str, dict]:
    """Write content to a user-scoped file and generate a user-bound time-limited download URL."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: Missing user context.", {}

    user_dir = (WORKSPACE_DIR / str(user_id)).resolve()
    user_dir.mkdir(parents=True, exist_ok=True)
    target_path = (user_dir / filename).resolve()

    if not target_path.is_relative_to(user_dir):
        return "Error: Invalid file path.", {}

    if target_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return f"Error: Extension '{target_path.suffix}' not allowed.", {}

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")

    # Store path relative to WORKSPACE_DIR (e.g., "42/report.md")
    rel_path = target_path.relative_to(WORKSPACE_DIR).as_posix()
    download_url = create_download_url(rel_path, user_id=user_id)

    artifact = {
        "filename": target_path.name,
        "download_url": download_url,
        "expires_in_seconds": URL_TTL_SECONDS,
    }
    llm_text = f"File '{target_path.name}' saved. Download link (expires in 15 minutes): {download_url}"
    return llm_text, artifact


@tool
async def read_file(filename: str, config: RunnableConfig) -> str:
    """reads and returns the content of a file"""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: Missing user context."

    user_dir = (WORKSPACE_DIR / str(user_id)).resolve()
    target_path = (user_dir / filename).resolve()

    if not target_path.is_file():
        return f"{target_path} does not exists"
    try:
        file_content = target_path.read_text(encoding="utf-8")
        file_content = (
            file_content
            if len(list(file_content)) < 1000
            else file_content + "[Truncated Text...]"
        )
        return file_content
    except OSError as e:
        return f"Some error occured {e}"


@tool
async def alter_file(
    filename: str, config: RunnableConfig, content: str, line_number=0
) -> str:
    """change file path"""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: Missing user context."

    user_dir = (WORKSPACE_DIR / str(user_id)).resolve()
    target_path = (user_dir / filename).resolve()

    if not target_path.is_file():
        return f"{target_path} does not exists"
    try:
        lines = target_path.read_text(encoding="utf-8").splitlines()
        idx = max(0, min(line_number - 1, len(lines)))
        lines.insert(idx, content)

        target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"Successfully inserted the given text in the file {filename} at line number {line_number}"
    except OSError as e:
        return f"Error : {e}"
