from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class FeedbackInput(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    message: str = Field(min_length=5, max_length=3000)
    category: str = 'other'
    channel: str = 'web'
    product: str = 'workspace'
    language: str = 'en'
    status: str = 'open'
    affected_users: int = Field(default=1, ge=1, le=100000)
    attachment_count: int = Field(default=0, ge=0, le=50)
    priority: Literal['low','medium','high'] = 'medium'

class SearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=3, ge=1, le=10)
