"""
Feature flag configuration for the Affiliate Matrix backend.

This module provides a centralized configuration for feature flags,
allowing features to be enabled or disabled at runtime.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from pydantic import BaseSettings, Field

# Configure logger
logger = logging.getLogger(__name__)

class FeatureFlags(BaseSettings):
    """
    Feature flag configuration.
    
    This class defines all available feature flags and their default values.
    Feature flags can be overridden via environment variables or a JSON string.
    """
    
    # Delta endpoints for tracking changes
    enable_delta_endpoints: bool = Field(False, description="Enable delta endpoints for tracking changes")
    
    # Vault integration for secure credential storage
    enable_vault_integration: bool = Field(False, description="Enable HashiCorp Vault integration for secure credential storage")
    
    # Advanced filtering options
    enable_advanced_filtering: bool = Field(True, description="Enable advanced filtering options")
    
    # Telemetry and monitoring
    enable_telemetry: bool = Field(True, description="Enable telemetry and monitoring")
    
    # Background tasks for long-running operations
    enable_background_tasks: bool = Field(True, description="Enable background tasks for long-running operations")
    
    # Caching for improved performance
    enable_caching: bool = Field(True, description="Enable caching for improved performance")
    
    # Rate limiting for API endpoints
    enable_rate_limiting: bool = Field(False, description="Enable rate limiting for API endpoints")
    
    # Experimental features
    enable_experimental_features: bool = Field(False, description="Enable experimental features")
    
    class Config:
        env_prefix = "FEATURE_FLAG_"


class Settings(BaseSettings):
    """
    Application settings.
    
    This class defines all application settings with their default values.
    Settings can be overridden via environment variables.
    """
    
    # Environment
    ENVIRONMENT: str = Field("development", description="Application environment (development, staging, production)")
    DEBUG: bool = Field(False, description="Debug mode")
    
    # API settings
    API_PREFIX: str = Field("/api/v1", description="API prefix")
    API_TITLE: str = Field("Affiliate Matrix API", description="API title")
    API_DESCRIPTION: str = Field("API for the Affiliate Matrix application", description="API description")
    API_VERSION: str = Field("1.0.0", description="API version")
    
    # Database settings
    DATABASE_URL: str = Field("sqlite:///./affiliate_matrix.db", description="Database URL")
    
    # Pagination settings
    DEFAULT_PAGE_SIZE: int = Field(10, description="Default page size for pagination")
    MAX_PAGE_SIZE: int = Field(100, description="Maximum page size for pagination")
    
    # Logging settings
    LOG_LEVEL: str = Field("INFO", description="Log level")
    
    # CORS settings
    CORS_ORIGINS: list = Field(["*"], description="CORS allowed origins")
    
    # Feature flags
    ENABLE_FEATURE_FLAGS: bool = Field(True, description="Enable feature flags")
    FEATURE_FLAGS: Dict[str, Any] = Field(
        default_factory=dict,
        description="Feature flag overrides as a dictionary"
    )
    
    # Vault settings
    VAULT_URL: str = Field("http://vault:8200", description="HashiCorp Vault URL")
    VAULT_TOKEN: str = Field("affiliate_matrix_token", description="HashiCorp Vault token")
    
    # Telemetry settings
    TELEMETRY_ENABLED: bool = Field(True, description="Enable telemetry")
    TELEMETRY_PROVIDER: str = Field("console", description="Telemetry provider (console, prometheus, datadog)")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_feature_flags(self) -> FeatureFlags:
        """
        Get feature flag configuration.
        
        This method combines the default feature flags with any overrides
        specified in the FEATURE_FLAGS environment variable.
        
        Returns:
            FeatureFlags: Feature flag configuration
        """
        # Start with default feature flags
        feature_flags = FeatureFlags()
        
        # If feature flags are disabled, return defaults with all flags disabled
        if not self.ENABLE_FEATURE_FLAGS:
            logger.info("Feature flags are disabled")
            for field in feature_flags.__fields__:
                setattr(feature_flags, field, False)
            return feature_flags
        
        # Apply overrides from environment variable
        if self.FEATURE_FLAGS:
            logger.info(f"Applying feature flag overrides: {self.FEATURE_FLAGS}")
            for key, value in self.FEATURE_FLAGS.items():
                if hasattr(feature_flags, key):
                    setattr(feature_flags, key, value)
                else:
                    logger.warning(f"Unknown feature flag: {key}")
        
        logger.info(f"Feature flags: {feature_flags.dict()}")
        return feature_flags


# Create settings instance
settings = Settings()

# Create feature flags instance
FEATURE_FLAGS = settings.get_feature_flags()
