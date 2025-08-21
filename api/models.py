from pydantic import BaseModel
from typing import Optional

class Comment(BaseModel):
    country_key: str
    comment_text: str
    user: str
    source: Optional[str] = None
    additional_notes: Optional[str] = None
