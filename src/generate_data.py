"""Generate synthetic buyer data for the COD Trust Score demo.

Creates data/users.csv with a mix of normal buyers, new buyers,
occasional refusers, and planted abuse rings (accounts sharing devices).
All data is fake and generated from assumptions, not real Shopee data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 1000
OUT = Path(__file__).resolve().parent.parent / "data" / "users.csv"

BUYER_FAULT_REASONS = [
    "ไม่อยู่บ้าน ติดต่อไม่ได้",
    "เปลี่ยนใจ ไม่ต้องการแล้ว",
    "สั่งผิด ไม่ได้ตั้งใจสั่ง",
    "ไม่มีเงินจ่ายตอนของมาส่ง",
    "เจอที่อื่นถูกกว่า",
]
SELLER_FAULT_REASONS = [
    "ของไม่ตรงปก สีไม่ตรงกับที่สั่ง",
    "ร้านส่งช้ากว่ากำหนด 10 วัน",
    "กล่องบุบ สินค้าเสียหาย",
    "ได้รับสินค้าผิดรุ่น",
]


def _segment(rng: np.random.Generator) -> str:
    return rng.choice(
        ["loyal", "normal", "new", "occasional_refuser", "abuser"],
        p=[0.25, 0.40, 0.15, 0.12, 0.08],
    )


def generate(n_users: int = N_USERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    # Planted abuse rings: groups of accounts sharing one device
    ring_ids = [f"DEV-RING-{i:02d}" for i in range(8)]

    for i in range(n_users):
        seg = _segment(rng)
        if seg == "loyal":
            age = int(rng.integers(365, 2500))
            cod = int(rng.integers(15, 80))
            refused = int(rng.binomial(cod, 0.01))
            kyc = rng.random() < 0.85
            shared = 0
        elif seg == "normal":
            age = int(rng.integers(90, 1500))
            cod = int(rng.integers(3, 30))
            refused = int(rng.binomial(cod, 0.04))
            kyc = rng.random() < 0.6
            shared = int(rng.choice([0, 0, 0, 1]))
        elif seg == "new":
            age = int(rng.integers(1, 30))
            cod = int(rng.integers(0, 3))
            refused = int(rng.binomial(cod, 0.1))
            kyc = rng.random() < 0.3
            shared = int(rng.choice([0, 0, 1]))
        elif seg == "occasional_refuser":
            age = int(rng.integers(60, 1200))
            cod = int(rng.integers(5, 25))
            refused = int(rng.integers(1, 4))
            kyc = rng.random() < 0.5
            shared = int(rng.choice([0, 1]))
        else:  # abuser
            age = int(rng.integers(3, 120))
            cod = int(rng.integers(4, 20))
            refused = int(rng.integers(max(2, cod // 3), max(3, cod // 3 * 2 + 1)))
            kyc = rng.random() < 0.1
            shared = int(rng.integers(3, 9))

        refused = min(refused, cod)
        success = cod - refused
        high_value_ratio = round(
            float(rng.uniform(2.5, 6.0) if seg == "abuser" else rng.uniform(0.6, 1.8)), 2
        )

        reason = ""
        if refused > 0:
            buyer_fault = rng.random() < (0.9 if seg == "abuser" else 0.55)
            reason = rng.choice(BUYER_FAULT_REASONS if buyer_fault else SELLER_FAULT_REASONS)

        rows.append(
            {
                "user_id": f"U{i + 1:05d}",
                "segment_truth": seg,  # hidden ground truth, for evaluation only
                "account_age_days": age,
                "kyc_verified": bool(kyc),
                "shopeepay_linked": bool(rng.random() < (0.15 if seg == "abuser" else 0.55)),
                "cod_orders": cod,
                "success_count": success,
                "refused_count": refused,
                "refused_90d": min(refused, int(rng.integers(0, refused + 1))),
                "shared_device_accounts": shared,
                "device_id": rng.choice(ring_ids) if seg == "abuser" else f"DEV-{i:05d}",
                "address_changes_90d": int(
                    rng.integers(2, 7) if seg == "abuser" else rng.integers(0, 2)
                ),
                "avg_order_value": int(rng.integers(150, 2500)),
                "high_value_ratio": high_value_ratio,
                "last_refusal_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} users to {OUT}")
    print(df["segment_truth"].value_counts())


if __name__ == "__main__":
    main()
