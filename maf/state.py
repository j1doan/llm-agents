"""Simulated shared state with an effect log that supports undo.

Every mutation is an Effect tagged with the stage run that caused it. Reversible effects (key/value
writes, orders, ledger updates) can be undone; irreversible ones (emails, fund transfers) cannot,
and recovery reports them instead of pretending recomputation fixed them.
"""
from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

_MISSING = object()


@dataclass
class Effect:
    seq: int
    run_id: str
    kind: str          # kv | order | ledger | email | transfer
    data: dict
    reversible: bool
    prior: object = None
    undone: bool = False


class SimState:
    """ledger="sqlite" (D37) keeps only orders and the ledger in a per-episode SQLite file in a temp folder;
    `orders` and `ledger` then read the DB. Everything else stays in memory. Call close() to delete the file."""

    def __init__(self, skus, ledger="memory"):
        self.kv: dict = {}
        self.kv_writer: dict = {}     # key -> run_id of the last writer
        self.emails: list = []
        self.transfers: list = []
        self.io_s = 0.0               # seconds spent in SQLite calls
        self.backend = ledger
        self._db = self._tmp = None
        if ledger == "sqlite":
            self._tmp = tempfile.mkdtemp(prefix="maf_ledger_")
            self._db = sqlite3.connect(Path(self._tmp) / "ledger.db")
            self._sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, supplier TEXT, sku TEXT, qty INTEGER)")
            self._sql("CREATE TABLE ledger (sku TEXT PRIMARY KEY, qty INTEGER NOT NULL)")
            for s in skus:
                self._sql("INSERT INTO ledger VALUES (?, 0)", (s,))
        else:
            self._orders: list = []
            self._ledger: dict = {s: 0 for s in skus}
        self.effects: list = []
        self.reads: list = []         # (run_id, key, writer_run_id)
        self._order_id = 0

    def _sql(self, query, args=(), fetch=False):
        """One committed statement; its time counts as ledger I/O."""
        t0 = time.perf_counter()
        with self._db:
            cur = self._db.execute(query, args)
            rows = cur.fetchall() if fetch else None
        self.io_s += time.perf_counter() - t0
        return rows

    @property
    def orders(self) -> list:
        if self._db is None:
            return self._orders
        rows = self._sql("SELECT id, supplier, sku, qty FROM orders ORDER BY id", fetch=True)
        return [dict(zip(("id", "supplier", "sku", "qty"), r)) for r in rows]

    @property
    def ledger(self) -> dict:
        if self._db is None:
            return self._ledger
        return dict(self._sql("SELECT sku, qty FROM ledger", fetch=True))

    def close(self):
        """Close and delete the SQLite file (safe to call twice)."""
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def _add(self, run_id, kind, data, reversible, prior=None) -> Effect:
        eff = Effect(len(self.effects), run_id, kind, data, reversible, prior)
        self.effects.append(eff)
        return eff

    # ---- shared key/value state ----
    def read_kv(self, run_id, key):
        writer = self.kv_writer.get(key)
        self.reads.append((run_id, key, writer))
        return self.kv.get(key), writer

    def set_kv(self, run_id, key, value) -> Effect:
        eff = self._add(run_id, "kv", {"key": key, "value": value}, True,
                        (self.kv.get(key, _MISSING), self.kv_writer.get(key)))
        self.kv[key] = value
        self.kv_writer[key] = run_id
        return eff

    # ---- reversible actions ----
    def place_order(self, run_id, supplier, sku, qty) -> Effect:
        self._order_id += 1
        order = {"id": self._order_id, "supplier": supplier, "sku": sku, "qty": qty}
        if self._db is None:
            self._orders.append(order)
        else:
            self._sql("INSERT INTO orders VALUES (:id, :supplier, :sku, :qty)", order)
        return self._add(run_id, "order", dict(order), True)

    def update_ledger(self, run_id, sku, delta) -> Effect:
        self._add_ledger(sku, delta)
        return self._add(run_id, "ledger", {"sku": sku, "delta": delta}, True)

    def _add_ledger(self, sku, delta):
        if self._db is None:
            self._ledger[sku] = self._ledger.get(sku, 0) + delta
        else:
            self._sql("INSERT INTO ledger VALUES (?, ?) ON CONFLICT(sku) DO UPDATE SET qty = qty + excluded.qty",
                      (sku, delta))

    # ---- irreversible actions ----
    def send_email(self, run_id, to, subject, body) -> Effect:
        rec = {"to": to, "subject": subject, "body": body}
        self.emails.append(rec)
        return self._add(run_id, "email", rec, False)

    def transfer(self, run_id, account, amount) -> Effect:
        rec = {"account": account, "amount": amount}
        self.transfers.append(rec)
        return self._add(run_id, "transfer", rec, False)

    def undo(self, eff: Effect) -> bool:
        """Undo one effect. Returns False if it is irreversible (and leaves it in place)."""
        if eff.undone:
            return True
        if not eff.reversible:
            return False
        if eff.kind == "kv":
            key = eff.data["key"]
            prior_val, prior_writer = eff.prior
            if prior_val is _MISSING:
                self.kv.pop(key, None)
                self.kv_writer.pop(key, None)
            else:
                self.kv[key] = prior_val
                if prior_writer is None:
                    self.kv_writer.pop(key, None)
                else:
                    self.kv_writer[key] = prior_writer
        elif eff.kind == "order":
            if self._db is None:
                self._orders = [o for o in self._orders if o["id"] != eff.data["id"]]
            else:
                self._sql("DELETE FROM orders WHERE id = ?", (eff.data["id"],))
        elif eff.kind == "ledger":
            if self._db is None:
                self._ledger[eff.data["sku"]] -= eff.data["delta"]
            else:
                self._sql("UPDATE ledger SET qty = qty - ? WHERE sku = ?", (eff.data["delta"], eff.data["sku"]))
        eff.undone = True
        return True
