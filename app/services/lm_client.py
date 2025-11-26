from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from app.infra.settings import settings


class LunchMoneyClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        # Use env settings by default
        self.api_key = api_key or settings.lunchmoney_api_key
        # Strip trailing slash so we don't get // in the final URL
        self.base_url = (base_url or settings.lunchmoney_base_url).rstrip("/")

        if not self.api_key:
            raise RuntimeError("LUNCHMONEY_API_KEY not configured")

        # httpx client with base_url; paths will be relative
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    # ---------- low-level helper ----------

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Simple GET wrapper. `path` can be '/foo' or 'foo'.
        """
        resp = self._client.get(path.lstrip("/"), params=params or {})
        resp.raise_for_status()
        return resp.json()

    # ---------- public API methods ----------

    def get_assets(self) -> List[Dict[str, Any]]:
        """Return Lunch Money 'assets' (accounts)."""
        data = self._get("/assets")
        # LM usually wraps as {"assets": [...]}
        return data.get("assets", data)

    def get_plaid_accounts(self) -> List[Dict[str, Any]]:
        """
        Return Plaid-linked accounts.

        Endpoint: GET /plaid_accounts
        Docs: https://github.com/lunch-money/developers
        """
        data = self._get("/plaid_accounts")
        # Typically {"plaid_accounts": [...]}
        return data.get("plaid_accounts", data)

    def get_transactions(
        self,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return transactions.

        You can pass filters in `params` later (start_date, end_date, etc.).
        Endpoint: GET /transactions
        """
        data = self._get("/transactions", params=params or {})
        # Typically {"transactions": [...]}
        return data.get("transactions", data)