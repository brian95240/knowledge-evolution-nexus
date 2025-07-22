# backend/core/key_management.py
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from .vault_stub import VaultStub # Assuming vault_stub.py exists

logger = logging.getLogger(__name__)

class KeyManager:
    def __init__(self, vault: VaultStub):
        self.vault = vault
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(minutes=30)

    def get_credential(self, service_name: str, credential_type: str) -> Optional[str]:
        """
        Fetch credential from vault with caching

        Args:
            service_name: Name of service (e.g. 'skimlinks', 'shareasale')
            credential_type: Type of credential (e.g. 'api_key', 'secret')

        Returns:
            Credential string if found, None if not found
        """
        cache_key = f"{service_name}_{credential_type}"

        # Check cache first
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now() < cached['expires']:
                return cached['value']
            else:
                del self._cache[cache_key]

        try:
            credential = self.vault.get_secret(
                path=f"affiliate/{service_name}/{credential_type}"
            )
            if credential:
                self._cache[cache_key] = {
                    'value': credential,
                    'expires': datetime.now() + self._cache_ttl
                }
            return credential

        except Exception as e:
            logger.error(f"Error fetching credential: {str(e)}")
            return None

    def store_credential(self, service_name: str, credential_type: str, value: str) -> bool:
        """
        Store credential in vault

        Args:
            service_name: Name of service
            credential_type: Type of credential
            value: Credential value to store

        Returns:
            bool indicating success/failure
        """
        try:
            self.vault.set_secret(
                path=f"affiliate/{service_name}/{credential_type}",
                value=value
            )
            # Invalidate cache
            cache_key = f"{service_name}_{credential_type}"
            if cache_key in self._cache:
                del self._cache[cache_key]
            return True

        except Exception as e:
            logger.error(f"Error storing credential: {str(e)}")
            return False

    def delete_credential(self, service_name: str, credential_type: str) -> bool:
        """
        Delete credential from vault

        Args:
            service_name: Name of service
            credential_type: Type of credential

        Returns:
            bool indicating success/failure
        """
        try:
            self.vault.delete_secret(
                path=f"affiliate/{service_name}/{credential_type}"
            )
            # Invalidate cache
            cache_key = f"{service_name}_{credential_type}"
            if cache_key in self._cache:
                del self._cache[cache_key]
            return True

        except Exception as e:
            logger.error(f"Error deleting credential: {str(e)}")
            return False
