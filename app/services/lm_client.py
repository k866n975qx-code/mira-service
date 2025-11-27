from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from app.infra.settings import settings


class LunchMoneyClient:
    def __init__(self) -> None:
        self.base_url = settings.lunchmoney_base_url.rstrip("/")
        self.token = settings.lunchmoney_api_key

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

    # ---------- existing methods ----------

    def get_assets(self) -> List[Dict[str, Any]]:
        data = self._get("/assets")
        # /assets returns {"assets": [...]}
        return data.get("assets", [])

    def get_plaid_accounts(self) -> List[Dict[str, Any]]:
        data = self._get("/plaid_accounts")
        # /plaid_accounts returns a raw list
        if isinstance(data, list):
            return data
        return data.get("plaid_accounts", data)

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