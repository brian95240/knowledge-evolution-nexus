# backend/index/ingest.py
import uuid
from datetime import datetime
from typing import Dict, List, Any
from .models import ProgramIndexEntry, Offer, DorkingFinding
from .store import MasterIndexStore


class MasterIndexIngester:
    """Transforms raw data from external sources and upserts it into the index."""

    def __init__(self, store: MasterIndexStore):
        self.store = store

    # ---------- Aggregator helpers ---------- #
    def ingest_skimlinks_merchants(self, merchants: List[Dict[str, Any]]) -> None:
        """Upsert merchants returned by the Skimlinks API."""
        for m in merchants:
            entry = self._merchant_to_entry(m)
            self.store.upsert(entry)

    def ingest_skimlinks_offers(self, merchant_id: str, offers: List[Dict[str, Any]]) -> None:
        """Attach offers to an existing merchant entry (or create if missing)."""
        offer_models = [self._offer_from_skimlinks(o, merchant_id) for o in offers]
        self.store.append_offers(merchant_id, offer_models, network='skimlinks')

    # ---------- Google‑dorking helpers ---------- #
    def ingest_dorking_results(self, domain: str, findings: List[Dict[str, Any]]) -> None:
        finding_models = [DorkingFinding(**f) for f in findings]
        self.store.append_dorking(domain, finding_models)

    # ---------- Internal adapters ---------- #
    @staticmethod
    def _merchant_to_entry(m: Dict[str, Any]) -> ProgramIndexEntry:
        return ProgramIndexEntry(
            id=str(uuid.uuid4()),
            name=m.get('name') or m.get('merchant_name'),
            domain=m.get('domain'),
            merchant_id=str(m.get('id')),
            network='skimlinks',
            country=m.get('country'),
            commission_rate=m.get('commission_rate'),
            tags=m.get('tags', []),
            source='aggregator',
            last_updated=datetime.utcnow()
        )

    @staticmethod
    def _offer_from_skimlinks(o: Dict[str, Any], merchant_id: str) -> Offer:
        return Offer(
            id=str(o.get('id')),
            title=o.get('title'),
            url=o.get('url'),
            description=o.get('description'),
            commission=o.get('commission'),
            start_date=o.get('start_date'),
            end_date=o.get('end_date')
        )
