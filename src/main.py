from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
RANDOM_STATE = 42


def generate_interactions(n_users: int = 450, n_items: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = [f"U{i:04d}" for i in range(1, n_users + 1)]
    item_ids = [f"I{i:04d}" for i in range(1, n_items + 1)]

    n_categories = 12
    item_category = rng.integers(0, n_categories, size=n_items)
    user_pref_category = rng.integers(0, n_categories, size=n_users)

    user_f = rng.normal(0, 1, size=(n_users, 8))
    item_f = rng.normal(0, 1, size=(n_items, 8))

    rows = []
    start = pd.Timestamp("2025-09-01")

    for u_idx, user in enumerate(user_ids):
        affinity = user_f[u_idx] @ item_f.T
        probs = np.exp(affinity - affinity.max())
        probs = probs / probs.sum()
        n_events = int(rng.integers(20, 85))
        event_days = rng.integers(0, 120, size=n_events)

        pref_cat = user_pref_category[u_idx]
        pref_idx = np.where(item_category == pref_cat)[0]
        non_pref_idx = np.where(item_category != pref_cat)[0]
        favorite_size = min(12, len(pref_idx))
        favorite_idx = rng.choice(pref_idx, size=favorite_size, replace=False)
        pref_other_idx = np.setdiff1d(pref_idx, favorite_idx)

        pref_probs = probs[pref_idx]
        pref_probs = pref_probs / pref_probs.sum()
        fav_probs = probs[favorite_idx]
        fav_probs = fav_probs / fav_probs.sum()
        pref_other_probs = probs[pref_other_idx] if len(pref_other_idx) > 0 else np.array([])
        if len(pref_other_probs) > 0:
            pref_other_probs = pref_other_probs / pref_other_probs.sum()
        non_pref_probs = probs[non_pref_idx]
        non_pref_probs = non_pref_probs / non_pref_probs.sum()

        for day in event_days:
            r = rng.uniform()
            if r < 0.85:
                i_idx = int(rng.choice(favorite_idx, p=fav_probs))
                purchase_p, cart_p = 0.32, 0.34
            elif r < 0.95 and len(pref_other_idx) > 0:
                i_idx = int(rng.choice(pref_other_idx, p=pref_other_probs))
                purchase_p, cart_p = 0.15, 0.30
            else:
                i_idx = int(rng.choice(non_pref_idx, p=non_pref_probs))
                purchase_p, cart_p = 0.03, 0.17
            view_p = max(0.01, 1.0 - purchase_p - cart_p)
            event_type = rng.choice(["view", "cart", "purchase"], p=[view_p, cart_p, purchase_p])
            weight = {"view": 1.0, "cart": 3.0, "purchase": 7.0}[event_type]
            rows.append(
                {
                    "user_id": user,
                    "item_id": item_ids[i_idx],
                    "event_type": event_type,
                    "event_weight": weight,
                    "timestamp": start + pd.Timedelta(days=int(day)),
                }
            )

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def cosine_similarity_matrix(m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    m_norm = m / norm
    sim = m_norm @ m_norm.T
    np.fill_diagonal(sim, 0.0)
    return sim


def recommend_for_user(user_vector: np.ndarray, sim: np.ndarray, consumed_mask: np.ndarray, top_k: int = 10) -> np.ndarray:
    scores = user_vector @ sim
    scores[consumed_mask] = -np.inf
    top_idx = np.argsort(-scores)[:top_k]
    return top_idx


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    interactions = generate_interactions(seed=RANDOM_STATE)

    purchases = interactions[interactions["event_type"] == "purchase"].copy()
    purchases = purchases.sort_values(["user_id", "timestamp"])

    holdout_rows = purchases.groupby("user_id").tail(1).copy()
    holdout_ts = holdout_rows.set_index("user_id")["timestamp"]

    # Temporal split: keep only events strictly before each user's holdout purchase.
    train_interactions = interactions.merge(
        holdout_ts.rename("holdout_ts"),
        how="left",
        left_on="user_id",
        right_index=True,
    )
    train_interactions = train_interactions[
        train_interactions["holdout_ts"].isna() | (train_interactions["timestamp"] < train_interactions["holdout_ts"])
    ].drop(columns=["holdout_ts"])

    users = sorted(interactions["user_id"].unique())
    items = sorted(interactions["item_id"].unique())
    u_to_i = {u: i for i, u in enumerate(users)}
    it_to_i = {it: i for i, it in enumerate(items)}

    mat = np.zeros((len(users), len(items)), dtype=float)
    purchase_mat = np.zeros((len(users), len(items)), dtype=float)
    for row in train_interactions.itertuples(index=False):
        mat[u_to_i[row.user_id], it_to_i[row.item_id]] += float(row.event_weight)
        if row.event_type == "purchase":
            purchase_mat[u_to_i[row.user_id], it_to_i[row.item_id]] += 1.0

    item_user = mat.T
    sim = cosine_similarity_matrix(item_user)

    rec_rows = []
    hits = 0
    eval_users = 0
    holdout_map = dict(zip(holdout_rows["user_id"], holdout_rows["item_id"]))

    for user in users:
        u_idx = u_to_i[user]
        rec_idx = recommend_for_user(
            user_vector=mat[u_idx],
            sim=sim,
            consumed_mask=(purchase_mat[u_idx] > 0),
            top_k=10,
        )
        rec_items = [items[i] for i in rec_idx]

        for rank, item_id in enumerate(rec_items, start=1):
            rec_rows.append({"user_id": user, "item_id": item_id, "rank": rank})

        if user in holdout_map:
            eval_users += 1
            if holdout_map[user] in rec_items:
                hits += 1

    recommendations = pd.DataFrame(rec_rows)
    hit_rate_at_10 = hits / max(1, eval_users)

    interactions.to_csv(DATA_DIR / "interactions_synthetic.csv", index=False)
    recommendations.to_csv(DATA_DIR / "recommendations_top10.csv", index=False)

    model_info = {
        "algorithm": "item_item_cosine_implicit",
        "n_users": len(users),
        "n_items": len(items),
        "evaluation_users": eval_users,
        "hit_rate_at_10": round(hit_rate_at_10, 4),
    }
    with (MODELS_DIR / "model_info.json").open("w", encoding="utf-8") as fp:
        json.dump(model_info, fp, indent=2)

    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump({"hit_rate_at_10": round(hit_rate_at_10, 4)}, fp, indent=2)

    notes = (
        "# Recomendacao Ecommerce - Analysis Notes\n\n"
        f"- HitRate@10: {round(hit_rate_at_10, 4)}\n"
        "- Treino com feedback implicito (view/cart/purchase).\n"
    )
    (NOTEBOOKS_DIR / "analysis_notes.md").write_text(notes, encoding="utf-8")

    print(f"HitRate@10: {round(hit_rate_at_10, 4)}")


if __name__ == "__main__":
    main()
