# backend/index/models.py
from __future__ import annotations
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
from datetime import datetime


class Offer(BaseModel):
    id: str
    title: str
    url: HttpUrl
    description: Optional[str]
    commission: Optional[str]
    start_date: Optional[datetime]
    end_date: Optional[datetime]


class DorkingFinding(BaseModel):
    query: str
    url: HttpUrl
    snippet: Optional[str]
    discovered_at: datetime = Field(default_factory=datetime.utcnow)


class ProgramIndexEntry(BaseModel):
    """The canonical record stored in the Master Index."""
    id: str                                      # internal uuid
    name: str
    domain: Optional[str]
    merchant_id: Optional[str]                   # e.g. Skimlinks or other network ID
    network: Optional[str]                       # 'skimlinks', 'shareasale', 'awin', ...
    country: Optional[str]
    commission_rate: Optional[str]
    tags: List[str] = []
    offers: List[Offer] = []
    dorking_findings: List[DorkingFinding] = []
    source: str                                  # 'aggregator', 'dorking', ...
    last_updated: datetime = Field(default_factory=datetime.utcnow)
