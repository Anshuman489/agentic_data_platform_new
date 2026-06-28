"""
agents/catalog_manager.py — Discovers all profiled tables from the cache directory.

No separate catalog file is maintained. Every cache/*.json file IS the catalog.
CatalogManager scans the cache directory, deserializes each file as a
DatasetProfile, and returns them for use by TableRouterAgent.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import settings
from models.dataset import DatasetProfile

logger = logging.getLogger(__name__)


class CatalogManager:
    """
    Read-only view of all cached DatasetProfiles.

    Usage:
        catalog = CatalogManager()
        profiles = catalog.list()   # all valid, non-stale profiles
        n = catalog.count
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or settings.cache_dir

    def list(self) -> list[DatasetProfile]:
        """Return all valid, non-stale DatasetProfiles found in the cache directory."""
        cache_path = Path(self._cache_dir)
        if not cache_path.exists():
            return []

        profiles = []
        for path in sorted(cache_path.glob("*.json")):
            profile = self._load(path)
            if profile is not None:
                profiles.append(profile)

        logger.info("Catalog: %d table(s) available in cache", len(profiles))
        return profiles

    @property
    def count(self) -> int:
        return len(self.list())

    def _load(self, path: Path) -> DatasetProfile | None:
        try:
            profile = DatasetProfile.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            logger.warning("Skipping unreadable cache file: %s", path.name)
            return None

        if settings.cache_max_age_hours > 0:
            age = datetime.now(timezone.utc) - profile.profiled_at
            if age > timedelta(hours=settings.cache_max_age_hours):
                logger.info("Skipping stale cache: %s (age=%s)", path.name, age)
                return None

        return profile
