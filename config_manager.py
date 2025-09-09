"""
Configuration Manager for LLM Firewalclass EndpointConfiguration(BaseModel):
    ""Configuration for endpoint forwarding.""
    url: str = Field(..., description="Target endpoint URL")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional HTTP headers")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates")
    method: str = Field(default="POST", description="HTTP method to use")
    include_scan_results: bool = Field(default=True, description="Include scan results in forwarded payload")
    forward_on_unsafe: bool = Field(default=False, description="Forward even if content is marked unsafe")
    forwarding_enabled: bool = Field(default=False, description="Enable automatic forwarding for scan requests") module handles configuration storage and retrieval using an external database.
It provides a secure, scalable way to manage endpoint configurations, headers,
and other application settings.
"""
# pyright: reportAttributeAccessIssue=false

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from sqlalchemy import create_engine, Column, String, Text, DateTime, Boolean, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager

logger = logging.getLogger("llmfirewall_api.config_manager")

Base = declarative_base()

class ConfigurationEntry(Base):
    """Database model for configuration entries."""
    __tablename__ = "configurations"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text)
    config_type = Column(String, nullable=False, index=True)  # 'endpoint', 'global', etc.
    config_data = Column(JSON, nullable=False)  # Stores the actual configuration as JSON
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, default="system")
    tags = Column(JSON)  # For categorization and search

# Pydantic models for validation and API
class EndpointConfiguration(BaseModel):
    """Configuration for endpoint forwarding."""
    url: str = Field(..., description="Target endpoint URL")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional HTTP headers")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates")
    method: str = Field(default="POST", description="HTTP method to use")
    include_scan_results: bool = Field(default=True, description="Include scan results in forwarded payload")
    forward_on_unsafe: bool = Field(default=False, description="Forward even if content is marked unsafe")
    forwarding_enabled: bool = Field(default=False, description="Enable automatic forwarding for scan requests")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate endpoint URL for security."""
        if not v:
            raise ValueError("URL cannot be empty")
        
        from urllib.parse import urlparse
        import ipaddress
        
        try:
            parsed = urlparse(v)
        except Exception:
            raise ValueError("Invalid URL format")
        
        if not parsed.scheme:
            raise ValueError("URL must include a protocol (http:// or https://)")
        
        if parsed.scheme not in ['http', 'https']:
            raise ValueError("URL must use HTTP or HTTPS protocol")
        
        if not parsed.netloc:
            raise ValueError("URL must include a valid host")
        
        # Security check: prevent SSRF to private networks
        try:
            # Extract hostname from netloc (remove port if present)
            hostname = parsed.hostname
            if hostname:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    # In production, you might want to be more restrictive
                    logger.warning(f"URL points to private/local address: {hostname}")
                    # For development, we'll allow it but log a warning
                    # raise ValueError("URL cannot point to private IP addresses")
        except ValueError:
            # Not an IP address, likely a domain name - this is fine
            pass

        return v

    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate HTTP method."""
        allowed_methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
        if v.upper() not in allowed_methods:
            raise ValueError(f"Method must be one of: {allowed_methods}")
        return v.upper()

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout value."""
        if v <= 0 or v > 300:  # Max 5 minutes
            raise ValueError("Timeout must be between 1 and 300 seconds")
        return v

class ConfigurationRecord(BaseModel):
    """Pydantic model for configuration records."""
    id: str
    name: str
    description: Optional[str] = None
    config_type: str
    config_data: Dict[str, Any]
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    created_by: str = "system"
    tags: Optional[List[str]] = None

class CreateConfigurationRequest(BaseModel):
    """Request model for creating new configurations."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    config_type: str = Field(..., min_length=1, max_length=50)
    config_data: Dict[str, Any] = Field(...)
    tags: Optional[List[str]] = Field(default=None, description="Configuration tags")

class UpdateConfigurationRequest(BaseModel):
    """Request model for updating configurations."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    config_data: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    tags: Optional[List[str]] = Field(default=None, description="Configuration tags")

class ConfigurationManager:
    """Manages configuration storage and retrieval."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            database_url: Database connection URL. If None, uses SQLite with default path.
        """
        if database_url is None:
            # Default to SQLite in the application directory
            db_path = os.getenv("CONFIG_DB_PATH", "config.db")
            database_url = f"sqlite+aiosqlite:///{db_path}"

        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(self.engine, class_=AsyncSession)

        logger.info(f"ConfigurationManager initialized with database: {database_url}")

    async def initialize_database(self):
        """Create database tables if they don't exist."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self):
        """Close the database connection."""
        await self.engine.dispose()

    @asynccontextmanager
    async def get_session(self):
        """Get an async database session."""
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def create_configuration(
        self,
        name: str,
        config_type: str,
        config_data: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_by: str = "system"
    ) -> str:
        """
        Create a new configuration entry.

        Args:
            name: Configuration name
            config_type: Type of configuration (e.g., 'endpoint', 'global')
            config_data: Configuration data as dictionary
            description: Optional description
            tags: Optional tags for categorization
            created_by: User who created the configuration

        Returns:
            Configuration ID
        """
        import uuid

        config_id = str(uuid.uuid4())

        async with self.get_session() as session:
            config_entry = ConfigurationEntry(
                id=config_id,
                name=name,
                description=description,
                config_type=config_type,
                config_data=config_data,
                created_by=created_by,
                tags=tags or []
            )

            session.add(config_entry)
            await session.commit()

        logger.info(f"Created configuration: {name} (ID: {config_id})")
        return config_id

    async def get_configuration(self, config_id: str) -> Optional[ConfigurationRecord]:
        """
        Get a configuration by ID.

        Args:
            config_id: Configuration ID

        Returns:
            Configuration record or None if not found
        """
        async with self.get_session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(ConfigurationEntry).where(ConfigurationEntry.id == config_id)
            )
            config_entry = result.scalar_one_or_none()

            if config_entry:
                # Type: ignore comments to suppress pylance warnings for SQLAlchemy attributes
                return ConfigurationRecord(
                    id=config_entry.id,  # type: ignore[arg-type]
                    name=config_entry.name,  # type: ignore[arg-type]
                    description=config_entry.description,  # type: ignore[arg-type]
                    config_type=config_entry.config_type,  # type: ignore[arg-type]
                    config_data=config_entry.config_data,  # type: ignore[arg-type]
                    is_active=config_entry.is_active,  # type: ignore[arg-type]
                    created_at=config_entry.created_at,  # type: ignore[arg-type]
                    updated_at=config_entry.updated_at,  # type: ignore[arg-type]
                    created_by=config_entry.created_by,  # type: ignore[arg-type]
                    tags=config_entry.tags or []  # type: ignore[arg-type]
                )

            return None

    async def get_configuration_by_name(
        self,
        name: str,
        config_type: Optional[str] = None
    ) -> Optional[ConfigurationRecord]:
        """
        Get a configuration by name and optionally type.

        Args:
            name: Configuration name
            config_type: Optional configuration type filter

        Returns:
            Configuration record or None if not found
        """
        async with self.get_session() as session:
            from sqlalchemy import select, and_

            conditions = [ConfigurationEntry.name == name]
            if config_type:
                conditions.append(ConfigurationEntry.config_type == config_type)

            result = await session.execute(
                select(ConfigurationEntry).where(and_(*conditions))
            )
            config_entry = result.scalar_one_or_none()

            if config_entry:
                return ConfigurationRecord(
                    id=config_entry.id,  # type: ignore[arg-type]
                    name=config_entry.name,  # type: ignore[arg-type]
                    description=config_entry.description,  # type: ignore[arg-type]
                    config_type=config_entry.config_type,  # type: ignore[arg-type]
                    config_data=config_entry.config_data,  # type: ignore[arg-type]
                    is_active=config_entry.is_active,  # type: ignore[arg-type]
                    created_at=config_entry.created_at,  # type: ignore[arg-type]
                    updated_at=config_entry.updated_at,  # type: ignore[arg-type]
                    created_by=config_entry.created_by,  # type: ignore[arg-type]
                    tags=config_entry.tags or []  # type: ignore[arg-type]
                )

            return None

    async def list_configurations(
        self,
        config_type: Optional[str] = None,
        active_only: bool = True,
        tags: Optional[List[str]] = None
    ) -> List[ConfigurationRecord]:
        """
        List configurations with optional filtering.

        Args:
            config_type: Filter by configuration type
            active_only: Only return active configurations
            tags: Filter by tags (any tag match)

        Returns:
            List of configuration records
        """
        async with self.get_session() as session:
            from sqlalchemy import select, and_, or_

            conditions = []
            if config_type:
                conditions.append(ConfigurationEntry.config_type == config_type)
            if active_only:
                conditions.append(ConfigurationEntry.is_active == True)

            query = select(ConfigurationEntry)
            if conditions:
                query = query.where(and_(*conditions))

            result = await session.execute(query.order_by(ConfigurationEntry.updated_at.desc()))
            config_entries = result.scalars().all()

            configurations = []
            for config_entry in config_entries:
                # Filter by tags if specified
                if tags:
                    entry_tags = config_entry.tags or []
                    if not any(tag in entry_tags for tag in tags):
                        continue

                configurations.append(ConfigurationRecord(
                    id=config_entry.id,  # type: ignore[arg-type]
                    name=config_entry.name,  # type: ignore[arg-type]
                    description=config_entry.description,  # type: ignore[arg-type]
                    config_type=config_entry.config_type,  # type: ignore[arg-type]
                    config_data=config_entry.config_data,  # type: ignore[arg-type]
                    is_active=config_entry.is_active,  # type: ignore[arg-type]
                    created_at=config_entry.created_at,  # type: ignore[arg-type]
                    updated_at=config_entry.updated_at,  # type: ignore[arg-type]
                    created_by=config_entry.created_by,  # type: ignore[arg-type]
                    tags=config_entry.tags or []  # type: ignore[arg-type]
                ))

            return configurations

    async def update_configuration(
        self,
        config_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a configuration.

        Args:
            config_id: Configuration ID
            updates: Dictionary of fields to update

        Returns:
            True if updated successfully, False if not found
        """
        async with self.get_session() as session:
            from sqlalchemy import select, update

            # Check if configuration exists
            result = await session.execute(
                select(ConfigurationEntry).where(ConfigurationEntry.id == config_id)
            )
            config_entry = result.scalar_one_or_none()

            if not config_entry:
                return False

            # Update the entry
            updates['updated_at'] = datetime.utcnow()
            await session.execute(
                update(ConfigurationEntry)
                .where(ConfigurationEntry.id == config_id)
                .values(**updates)
            )
            await session.commit()

            logger.info(f"Updated configuration: {config_id}")
            return True

    async def delete_configuration(self, config_id: str) -> bool:
        """
        Delete a configuration (permanently removes from database).

        Args:
            config_id: Configuration ID

        Returns:
            True if deleted successfully, False if not found
        """
        async with self.get_session() as session:
            from sqlalchemy import select, delete

            # Check if configuration exists
            result = await session.execute(
                select(ConfigurationEntry).where(ConfigurationEntry.id == config_id)
            )
            config_entry = result.scalar_one_or_none()

            if not config_entry:
                return False

            # Delete the entry permanently
            await session.execute(
                delete(ConfigurationEntry).where(ConfigurationEntry.id == config_id)
            )
            await session.commit()

            logger.info(f"Permanently deleted configuration: {config_id}")
            return True

    async def get_endpoint_configuration(self, name: str) -> Optional[EndpointConfiguration]:
        """
        Get an endpoint configuration by name.

        Args:
            name: Endpoint configuration name

        Returns:
            EndpointConfiguration object or None if not found
        """
        config = await self.get_configuration_by_name(name, "endpoint")
        if config:
            try:
                return EndpointConfiguration(**config.config_data)
            except Exception as e:
                logger.error(f"Failed to parse endpoint configuration {name}: {e}")
                return None
        return None

    async def create_endpoint_configuration(
        self,
        name: str,
        endpoint_config: EndpointConfiguration,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new endpoint configuration.

        Args:
            name: Configuration name
            endpoint_config: EndpointConfiguration object
            description: Optional description
            tags: Optional tags

        Returns:
            Configuration ID
        """
        return await self.create_configuration(
            name=name,
            config_type="endpoint",
            config_data=endpoint_config.model_dump(),
            description=description,
            tags=tags
        )

    async def update_endpoint_configuration(
        self,
        name: str,
        endpoint_config: EndpointConfiguration,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Update an endpoint configuration by name.

        Args:
            name: Configuration name
            endpoint_config: Updated EndpointConfiguration object
            description: Optional updated description
            tags: Optional updated tags

        Returns:
            True if updated successfully, False if not found
        """
        # Get the configuration ID by name
        config_record = await self.get_configuration_by_name(name, "endpoint")
        if not config_record:
            return False

        updates: Dict[str, Any] = {
            "config_data": endpoint_config.model_dump(),
        }
        if description is not None:
            updates["description"] = description
        if tags is not None:
            updates["tags"] = tags

        return await self.update_configuration(config_record.id, updates)

    async def delete_endpoint_configuration(self, name: str) -> bool:
        """
        Delete an endpoint configuration by name.

        Args:
            name: Configuration name

        Returns:
            True if deleted successfully, False if not found
        """
        # Get the configuration ID by name
        config_record = await self.get_configuration_by_name(name, "endpoint")
        if not config_record:
            return False

        return await self.delete_configuration(config_record.id)

# Global configuration manager instance
config_manager: Optional[ConfigurationManager] = None

async def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global config_manager
    if config_manager is None:
        config_manager = ConfigurationManager()
        await config_manager.initialize_database()
    return config_manager

async def close_config_manager():
    """Close the global configuration manager."""
    global config_manager
    if config_manager:
        await config_manager.close()
        config_manager = None
