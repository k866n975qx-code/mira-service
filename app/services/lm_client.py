from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from app.infra.settings import settings
from app.services.cache_utils import load_ttl_cache, store_ttl_cache


class LunchMoneyClient:
    def __init__(self) -> None:
        self.base_url = settings.lunchmoney_base_url.rstrip("/")
        self.token = settings.lunchmoney_api_key
        self.cache_ttl = self._lm_cache_ttl_seconds()

        if not self.token:
            raise RuntimeError("LUNCHMONEY_API_KEY is not set")

        self._client = httpx.Client(timeout=30.0)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self._client.get(url, headers=self._headers(), params=params or {})
        resp.raise_for_status()
        return resp.json()

    def _lm_cache_ttl_seconds(self) -> int:
        try:
            return int(settings.lm_cache_ttl_seconds or 3600)  # type: ignore[attr-defined]
        except Exception:
            try:
                import os

                return int(os.getenv("MIRA_LM_CACHE_TTL_SECONDS", "3600"))
            except Exception:
                return 3600

    def _cache_key(self, label: str) -> str:
        return f"{label}:{self.base_url}"

    # ---------- existing methods ----------

    def get_assets(self) -> List[Dict[str, Any]]:
        cache_key = self._cache_key("assets")
        cached = load_ttl_cache("lunchmoney", cache_key)
        if isinstance(cached, list):
            return cached
        data = self._get("/assets")
        assets = data.get("assets", [])
        try:
            store_ttl_cache("lunchmoney", cache_key, assets, self.cache_ttl)
        except Exception:
            pass
        return assets

    def get_plaid_accounts(self) -> List[Dict[str, Any]]:
        cache_key = self._cache_key("plaid_accounts")
        cached = load_ttl_cache("lunchmoney", cache_key)
        if isinstance(cached, list):
            return cached
        data = self._get("/plaid_accounts")
        # /plaid_accounts returns a raw list
        result = data if isinstance(data, list) else data.get("plaid_accounts", data)
        try:
            store_ttl_cache("lunchmoney", cache_key, result, self.cache_ttl)
        except Exception:
            pass
        return result

    # ---------- NEW: transactions ----------

    def get_transactions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        plaid_account_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Wrapper for GET /transactions.

        Docs: https://github.com/lunch-money/developers
        """
        params: Dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if plaid_account_id:
            params["plaid_account_id"] = plaid_account_id

        data = self._get("/transactions", params=params)

        # /transactions returns {"transactions": [...], ...}
        return data.get("transactions", [])
