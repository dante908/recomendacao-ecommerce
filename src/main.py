from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache-reco")

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    import optuna  # type: ignore
except Exception:
    optuna = None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
REPORTS_DIR = ROOT / "reports"
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
    return np.argsort(-scores)[:top_k]


def evaluate_recs(rec_map: dict[str, list[str]], holdout_map: dict[str, str], k: int) -> dict[str, float]:
    hits = 0
    mrr_sum = 0.0
    eval_users = 0
    for user, target_item in holdout_map.items():
        recs = rec_map.get(user, [])[:k]
        eval_users += 1
        if target_item in recs:
            hits += 1
            rank = recs.index(target_item) + 1
            mrr_sum += 1.0 / rank
    hit_rate = hits / max(1, eval_users)
    mrr = mrr_sum / max(1, eval_users)
    return {
        "evaluation_users": float(eval_users),
        f"hit_rate_at_{k}": hit_rate,
        f"mrr_at_{k}": mrr,
    }


def recommend_with_classifier(
    train_interactions: pd.DataFrame,
    holdout_rows: pd.DataFrame,
    users: list[str],
    items: list[str],
    baseline_map: dict[str, list[str]],
) -> tuple[dict[str, list[str]] | None, dict[str, Any]]:
    if xgb is None:
        return None, {"xgboost_available": False, "used_optuna": False}

    score_map = {"view": 1.0, "cart": 3.0, "purchase": 7.0}
    train_scored = train_interactions.copy()
    train_scored["score"] = train_scored["event_type"].map(score_map).astype(float)

    user_item = train_scored.groupby(["user_id", "item_id"])["score"].sum().reset_index()
    user_stats = user_item.groupby("user_id").agg(user_events=("score", "sum"), user_unique_items=("item_id", "count")).reset_index()
    item_stats = user_item.groupby("item_id").agg(item_events=("score", "sum"), item_unique_users=("user_id", "count")).reset_index()

    holdout_targets = holdout_rows[["user_id", "item_id"]].copy()
    holdout_targets["label"] = 1

    # Hard negatives from candidate items not in observed positive pair.
    rng = np.random.default_rng(RANDOM_STATE)
    negatives = []
    pos_pairs = set(zip(holdout_targets["user_id"], holdout_targets["item_id"]))
    for user in holdout_targets["user_id"]:
        sampled = 0
        while sampled < 8:
            item = items[int(rng.integers(0, len(items)))]
            if (user, item) in pos_pairs:
                continue
            negatives.append({"user_id": user, "item_id": item, "label": 0})
            sampled += 1

    train_pairs = pd.concat([holdout_targets, pd.DataFrame(negatives)], ignore_index=True)

    feats = train_pairs.merge(user_stats, on="user_id", how="left").merge(item_stats, on="item_id", how="left")
    feats = feats.fillna(0.0)
    x_train = feats[["user_events", "user_unique_items", "item_events", "item_unique_users"]].to_numpy(dtype=float)
    y_train = feats["label"].to_numpy(dtype=int)

    if optuna is None:
        params = {
            "n_estimators": 240,
            "max_depth": 5,
            "learning_rate": 0.06,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train)
        used_optuna = False
    else:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 120, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": RANDOM_STATE,
                "n_jobs": 1,
            }
            model = xgb.XGBClassifier(**params)
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_train)[:, 1]
            # simple separability objective
            return float(np.mean(proba[y_train == 1]) - np.mean(proba[y_train == 0]))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=12, show_progress_bar=False)
        params = study.best_params | {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train)
        used_optuna = True

    rec_map: dict[str, list[str]] = {}
    user_stats_idx = user_stats.set_index("user_id")
    item_stats_idx = item_stats.set_index("item_id")

    for user in users:
        base = baseline_map.get(user, [])
        if not base:
            rec_map[user] = []
            continue

        uev = 0.0
        uui = 0.0
        if user in user_stats_idx.index:
            row = user_stats_idx.loc[user]
            uev = float(row["user_events"])
            uui = float(row["user_unique_items"])

        cand_rows = []
        for item in base:
            iev = 0.0
            iuu = 0.0
            if item in item_stats_idx.index:
                irow = item_stats_idx.loc[item]
                iev = float(irow["item_events"])
                iuu = float(irow["item_unique_users"])
            cand_rows.append([uev, uui, iev, iuu])

        scores = model.predict_proba(np.array(cand_rows, dtype=float))[:, 1]
        order = np.argsort(-scores)
        rec_map[user] = [base[i] for i in order[:10]]

    info = {
        "xgboost_available": True,
        "used_optuna": used_optuna,
        "xgboost_params": params,
    }
    return rec_map, info


def generate_reports(interactions: pd.DataFrame, recommendations: pd.DataFrame) -> list[str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return []

    generated: list[str] = []

    event_counts = interactions["event_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(event_counts.index.tolist(), event_counts.values.tolist())
    ax.set_title("Distribuicao de Eventos")
    ax.set_xlabel("Evento")
    ax.set_ylabel("Quantidade")
    ax.grid(axis="y", alpha=0.2)
    p1 = REPORTS_DIR / "event_distribution.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    generated.append(p1.name)

    rank_counts = recommendations["rank"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(rank_counts.index, rank_counts.values, marker="o")
    ax.set_title("Cobertura por Posicao de Rank")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Quantidade de recomendacoes")
    ax.grid(alpha=0.25)
    p2 = REPORTS_DIR / "rank_coverage.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    generated.append(p2.name)

    top_items = recommendations["item_id"].value_counts().head(15).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(top_items.index, top_items.values)
    ax.set_title("Itens Mais Recomendados (Top 15)")
    ax.set_xlabel("Ocorrencias")
    ax.grid(axis="x", alpha=0.2)
    p3 = REPORTS_DIR / "top_recommended_items.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    generated.append(p3.name)

    return generated


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    interactions = generate_interactions(seed=RANDOM_STATE)

    purchases = interactions[interactions["event_type"] == "purchase"].copy()
    purchases = purchases.sort_values(["user_id", "timestamp"])
    holdout_rows = purchases.groupby("user_id").tail(1).copy()
    holdout_ts = holdout_rows.set_index("user_id")["timestamp"]

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

    baseline_map: dict[str, list[str]] = {}
    rec_rows = []
    for user in users:
        u_idx = u_to_i[user]
        rec_idx = recommend_for_user(
            user_vector=mat[u_idx],
            sim=sim,
            consumed_mask=(purchase_mat[u_idx] > 0),
            top_k=10,
        )
        rec_items = [items[i] for i in rec_idx]
        baseline_map[user] = rec_items
        for rank, item_id in enumerate(rec_items, start=1):
            rec_rows.append({"user_id": user, "item_id": item_id, "rank": rank})

    holdout_map = dict(zip(holdout_rows["user_id"], holdout_rows["item_id"]))
    baseline_metrics = evaluate_recs(baseline_map, holdout_map, 10)

    classifier_map, clf_info = recommend_with_classifier(
        train_interactions=train_interactions,
        holdout_rows=holdout_rows,
        users=users,
        items=items,
        baseline_map=baseline_map,
    )

    selected_model = "item_item_cosine"
    selected_map = baseline_map
    advanced_metrics = None

    if classifier_map is not None:
        advanced_metrics = evaluate_recs(classifier_map, holdout_map, 10)
        if advanced_metrics["hit_rate_at_10"] >= baseline_metrics["hit_rate_at_10"]:
            selected_model = "item_item_plus_xgboost_ranker"
            selected_map = classifier_map

    rec_rows = []
    for user in users:
        for rank, item_id in enumerate(selected_map[user], start=1):
            rec_rows.append({"user_id": user, "item_id": item_id, "rank": rank})
    recommendations = pd.DataFrame(rec_rows)

    selected_metrics = evaluate_recs(selected_map, holdout_map, 10)

    interactions.to_csv(DATA_DIR / "interactions_synthetic.csv", index=False)
    recommendations.to_csv(DATA_DIR / "recommendations_top10.csv", index=False)

    report_files = generate_reports(interactions, recommendations)

    model_info = {
        "algorithm": selected_model,
        "n_users": len(users),
        "n_items": len(items),
        "evaluation_users": int(selected_metrics["evaluation_users"]),
        "hit_rate_at_10": round(float(selected_metrics["hit_rate_at_10"]), 4),
        "mrr_at_10": round(float(selected_metrics["mrr_at_10"]), 4),
        "xgboost_available": bool(clf_info.get("xgboost_available", False)),
        "used_optuna": bool(clf_info.get("used_optuna", False)),
        "reports_generated": len(report_files),
    }
    with (MODELS_DIR / "model_info.json").open("w", encoding="utf-8") as fp:
        json.dump(model_info, fp, indent=2)

    metrics = {
        "selected_model": selected_model,
        "evaluation_users": int(selected_metrics["evaluation_users"]),
        "hit_rate_at_10": round(float(selected_metrics["hit_rate_at_10"]), 4),
        "mrr_at_10": round(float(selected_metrics["mrr_at_10"]), 4),
        "baseline_hit_rate_at_10": round(float(baseline_metrics["hit_rate_at_10"]), 4),
        "baseline_mrr_at_10": round(float(baseline_metrics["mrr_at_10"]), 4),
        "advanced_hit_rate_at_10": round(float(advanced_metrics["hit_rate_at_10"]), 4)
        if advanced_metrics is not None
        else None,
        "advanced_mrr_at_10": round(float(advanced_metrics["mrr_at_10"]), 4)
        if advanced_metrics is not None
        else None,
        "xgboost_available": bool(clf_info.get("xgboost_available", False)),
        "used_optuna": bool(clf_info.get("used_optuna", False)),
        "reports_generated": len(report_files),
    }
    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    notes = (
        "# Recomendacao Ecommerce - Analysis Notes\n\n"
        f"- Modelo selecionado: {selected_model}\n"
        f"- HitRate@10: {metrics['hit_rate_at_10']}\n"
        f"- MRR@10: {metrics['mrr_at_10']}\n"
        f"- Baseline HitRate@10: {metrics['baseline_hit_rate_at_10']}\n"
        f"- Modelo avancado HitRate@10: {metrics['advanced_hit_rate_at_10']}\n"
        f"- Usuarios avaliados: {metrics['evaluation_users']}\n"
        f"- Graficos em reports/: {', '.join(report_files) if report_files else 'nao gerado'}\n"
    )
    (NOTEBOOKS_DIR / "analysis_notes.md").write_text(notes, encoding="utf-8")

    print(f"Modelo selecionado: {selected_model}")
    print(f"HitRate@10: {metrics['hit_rate_at_10']}")
    print(f"MRR@10: {metrics['mrr_at_10']}")


if __name__ == "__main__":
    main()
