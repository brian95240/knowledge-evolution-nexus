# backend/index/store.py
from typing import Dict, List, Optional
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from .models import ProgramIndexEntry, Offer, DorkingFinding


class MasterIndexStore:
    """In‑memory store; swap with DB layer later without changing callers."""

    def __init__(self):
        # key = (network, merchant_id)  OR fallback to domain
        self._store: Dict[str, ProgramIndexEntry] = {}

    # ------------ Low‑level helpers ------------ #
    @staticmethod
    def _make_key(network: Optional[str], merchant_id: Optional[str], domain: Optional[str]) -> str:
        if network and merchant_id:
            return f"{network}:{merchant_id}"
        return f"domain:{domain.lower()}" if domain else str(id)

    # ------------ CRUD‑like API --------------- #
    def upsert(self, entry: ProgramIndexEntry) -> None:
        key = self._make_key(entry.network, entry.merchant_id, entry.domain)
        existing = self._store.get(key)
        if existing:
            # naive merge – prefer freshest data
            entry.offers = existing.offers or entry.offers
            entry.dorking_findings = existing.dorking_findings or entry.dorking_findings
            entry.id = existing.id                         # keep stable uuid
        self._store[key] = entry.copy(update={'last_updated': datetime.utcnow()})

    def append_offers(self, merchant_id: str, offers: List[Offer], network: str) -> None:
        key = self._make_key(network, merchant_id, None)
        entry = self._store.get(key)
        if not entry:
            # create minimal entry if not present
            entry = ProgramIndexEntry(id=key, name='', merchant_id=merchant_id, network=network)
        # de‑duplicate by offer id
        existing_ids = {o.id for o in entry.offers}
        entry.offers.extend([o for o in offers if o.id not in existing_ids])
        entry.last_updated = datetime.utcnow()
        self._store[key] = entry

    def append_dorking(self, domain: str, findings: List[DorkingFinding]) -> None:
        key = self._make_key(None, None, domain)
        entry = self._store.get(key)
        if not entry:
            entry = ProgramIndexEntry(id=key, name='', domain=domain, network=None, merchant_id=None)
        entry.dorking_findings.extend(findings)
        entry.last_updated = datetime.utcnow()
        self._store[key] = entry

    # ------------ Query Helpers --------------- #
    def get_by_id(self, entry_id: str) -> Optional[ProgramIndexEntry]:
        for v in self._store.values():
            if v.id == entry_id:
                return v
        return None

    def list(self,
             network: Optional[str] = None,
             country: Optional[str] = None,
             tag: Optional[str] = None) -> List[ProgramIndexEntry]:
        results = list(self._store.values())
        if network:
            results = [e for e in results if e.network == network]
        if country:
            results = [e for e in results if e.country == country]
        if tag:
            results = [e for e in results if tag in e.tags]
        return results

    # Convenience JSON‑ready output
    def dump(self) -> List[dict]:
        return [jsonable_encoder(e) for e in self._store.values()]
