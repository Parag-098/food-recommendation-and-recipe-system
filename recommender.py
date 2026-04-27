import ast
import math
import os
import re
from collections import Counter
from difflib import get_close_matches
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd


class FoodRecommender:
    # ---------------------------------------------------------------------
    # Section 1: Load dataset
    # ---------------------------------------------------------------------
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df = self._load_data(dataset_path)

        # Cached stats for quick goal and explanation logic.
        self._calorie_q30 = float(self.df["calories"].quantile(0.30))
        self._calorie_q35 = float(self.df["calories"].quantile(0.35))
        self._calorie_q70 = float(self.df["calories"].quantile(0.70))
        self._calorie_median = float(self.df["calories"].median())

        # Name-query similarity cache for speed on repeated calls.
        self._name_similarity_cache: Dict[int, List[float]] = {}

        self._build_features()

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = str(text).lower().strip()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _token_set(text: str) -> Set[str]:
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        return {t for t in tokens if len(t) > 2}

    @staticmethod
    def _tokenize_terms(text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 2]

    def _load_data(self, dataset_path: str) -> pd.DataFrame:
        required_cols = [
            "name",
            "ingredients",
            "tags",
            "calories",
            "protein",
            "fat",
            "diet",
            "taste",
            "cost",
            "prep_time",
        ]

        df_raw = pd.read_csv(dataset_path)
        missing_cols = [c for c in required_cols if c not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

        selected_cols = required_cols.copy()
        if "steps" in df_raw.columns:
            selected_cols.append("steps")
        df = df_raw[selected_cols].copy()

        for col in ["name", "ingredients", "tags", "diet", "taste", "cost", "prep_time", "steps"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str).str.strip().str.lower()

        for col in ["calories", "protein", "fat"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["calories"] = df["calories"].fillna(df["calories"].median()).clip(lower=0)
        df["protein"] = df["protein"].fillna(df["protein"].median())
        df["fat"] = df["fat"].fillna(df["fat"].median())

        df = df[df["name"].str.len() > 0].reset_index(drop=True)
        df["name_norm"] = df["name"].apply(self._normalize_text)
        df["text_features"] = (df["ingredients"] + " " + df["tags"]).str.strip()
        df["ingredient_tokens"] = df["ingredients"].apply(self._token_set)

        # Enrich steps from RAW_recipes.csv when steps are missing in final dataset.
        if (df["steps"].str.len() == 0).mean() > 0.50:
            self._attach_optional_steps_from_raw(df, dataset_path)

        return df

    @staticmethod
    def _parse_list_like_text(value: str) -> List[str]:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [text] if text else []

    def _attach_optional_steps_from_raw(self, df: pd.DataFrame, dataset_path: str) -> None:
        raw_path = os.path.join(os.path.dirname(dataset_path), "RAW_recipes.csv")
        if not os.path.exists(raw_path):
            return

        target_names = set(df["name_norm"].tolist())
        if not target_names:
            return

        steps_map: Dict[str, str] = {}
        try:
            for chunk in pd.read_csv(raw_path, usecols=["name", "steps"], chunksize=50000):
                chunk["name"] = chunk["name"].fillna("").astype(str).str.strip().str.lower()
                chunk["name_norm"] = chunk["name"].apply(self._normalize_text)
                chunk = chunk[chunk["name_norm"].isin(target_names)]
                if chunk.empty:
                    continue

                for _, row in chunk.iterrows():
                    key = row["name_norm"]
                    if key in steps_map:
                        continue
                    steps_list = self._parse_list_like_text(row.get("steps", ""))
                    if steps_list:
                        steps_map[key] = "\n".join(steps_list)

                if len(steps_map) >= len(target_names):
                    break
        except Exception:
            return

        missing_mask = df["steps"].str.len() == 0
        if missing_mask.any():
            df.loc[missing_mask, "steps"] = df.loc[missing_mask, "name_norm"].map(steps_map).fillna("")

    @staticmethod
    def _health_label(calories: float) -> str:
        if calories < 200:
            return "Low Calorie"
        if calories > 500:
            return "High Energy"
        return "Balanced Meal"

    @staticmethod
    def compute_health_score(calories: float, protein: float = 0.0, fat: float = 0.0) -> int:
        # Lightweight post-processing metric for UI display (does not affect ranking).
        score = 5

        if 150 <= calories <= 450:
            score += 2
        elif calories < 150:
            score += 1

        if protein >= 15:
            score += 2
        elif protein >= 8:
            score += 1

        if fat <= 20:
            score += 1
        elif fat > 35:
            score -= 1

        return int(max(1, min(10, score)))

    # ---------------------------------------------------------------------
    # Section 2: Feature engineering (TF-IDF + numeric + categorical)
    # ---------------------------------------------------------------------
    def _build_features(self) -> None:
        self.cat_cols = ["diet", "taste", "cost", "prep_time"]

        self._calorie_min = float(self.df["calories"].min())
        self._calorie_max = float(self.df["calories"].max())
        self._calorie_range = max(1e-9, self._calorie_max - self._calorie_min)

        doc_tokens = [self._tokenize_terms(text) for text in self.df["text_features"].tolist()]
        doc_freq: Counter[str] = Counter()
        for tokens in doc_tokens:
            doc_freq.update(set(tokens))

        total_docs = max(1, len(doc_tokens))
        self._idf: Dict[str, float] = {}
        for term, df_count in doc_freq.items():
            self._idf[term] = math.log((1.0 + total_docs) / (1.0 + float(df_count))) + 1.0

        self._doc_vectors: List[Dict[str, float]] = []
        self._doc_norms: List[float] = []
        for tokens in doc_tokens:
            tf_counts = Counter(tokens)
            total_terms = max(1, len(tokens))
            vector: Dict[str, float] = {}
            for term, count in tf_counts.items():
                idf = self._idf.get(term)
                if idf is None:
                    continue
                vector[term] = (float(count) / float(total_terms)) * idf
            norm = math.sqrt(sum(value * value for value in vector.values()))
            self._doc_vectors.append(vector)
            self._doc_norms.append(norm)

        # Final feature weighting requirement.
        self.text_weight = 0.75
        self.calorie_weight = 0.20
        self.cat_weight = 0.05

        self.name_to_indices: Dict[str, List[int]] = {}
        for idx, nm in enumerate(self.df["name_norm"].tolist()):
            self.name_to_indices.setdefault(nm, []).append(idx)

        self.protein_var = float(self.df["protein"].var())
        self.fat_var = float(self.df["fat"].var())

    def _vectorize_text(self, text: str) -> Tuple[Dict[str, float], float]:
        tokens = self._tokenize_terms(text)
        if not tokens:
            return {}, 0.0

        tf_counts = Counter(tokens)
        total_terms = max(1, len(tokens))
        vector: Dict[str, float] = {}
        for term, count in tf_counts.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            vector[term] = (float(count) / float(total_terms)) * idf

        norm = math.sqrt(sum(value * value for value in vector.values()))
        return vector, norm

    @staticmethod
    def _cosine_similarity_dicts(
        left_vector: Dict[str, float],
        left_norm: float,
        right_vector: Dict[str, float],
        right_norm: float,
    ) -> float:
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0

        if len(left_vector) > len(right_vector):
            left_vector, right_vector = right_vector, left_vector
            left_norm, right_norm = right_norm, left_norm

        dot = 0.0
        for term, value in left_vector.items():
            dot += value * right_vector.get(term, 0.0)

        if dot <= 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _calorie_similarity(base_calories: float, candidate_calories: float, calorie_range: float) -> float:
        if calorie_range <= 0.0:
            return 1.0
        similarity = 1.0 - (abs(float(base_calories) - float(candidate_calories)) / calorie_range)
        return max(0.0, min(1.0, similarity))

    def _categorical_similarity(
        self,
        candidate_idx: int,
        anchor_idx: Optional[int] = None,
        user_context: Optional[Dict[str, str]] = None,
    ) -> float:
        if anchor_idx is not None:
            anchor_row = self.df.loc[anchor_idx]
            matches = 0
            for col in self.cat_cols:
                if str(self.df.at[candidate_idx, col]).strip() == str(anchor_row[col]).strip():
                    matches += 1
            return matches / float(len(self.cat_cols))

        if user_context:
            provided = [(col, value) for col, value in user_context.items() if value]
            if not provided:
                return 0.0
            matches = 0
            for col, value in provided:
                if str(self.df.at[candidate_idx, col]).strip() == str(value).strip():
                    matches += 1
            return matches / float(len(provided))

        return 0.0

    def _combined_similarity_for_index(
        self,
        candidate_idx: int,
        source_vector: Dict[str, float],
        source_norm: float,
        source_calories: float,
        anchor_idx: Optional[int] = None,
        user_context: Optional[Dict[str, str]] = None,
        use_calories: bool = True,
        use_categories: bool = True,
    ) -> float:
        text_score = self._cosine_similarity_dicts(source_vector, source_norm, self._doc_vectors[candidate_idx], self._doc_norms[candidate_idx])
        score = self.text_weight * text_score

        if use_calories:
            score += self.calorie_weight * self._calorie_similarity(source_calories, float(self.df.at[candidate_idx, "calories"]), self._calorie_range)

        if use_categories:
            score += self.cat_weight * self._categorical_similarity(candidate_idx, anchor_idx=anchor_idx, user_context=user_context)

        return score

    def _full_similarity_scores(
        self,
        source_idx: int,
        user_context: Optional[Dict[str, str]] = None,
        use_calories: bool = True,
        use_categories: bool = True,
    ) -> List[float]:
        if (
            source_idx in self._name_similarity_cache
            and user_context is None
            and use_calories
            and use_categories
        ):
            return self._name_similarity_cache[source_idx][:]

        source_vector = self._doc_vectors[source_idx]
        source_norm = self._doc_norms[source_idx]
        source_calories = float(self.df.at[source_idx, "calories"])

        scores = [
            self._combined_similarity_for_index(
                candidate_idx=i,
                source_vector=source_vector,
                source_norm=source_norm,
                source_calories=source_calories,
                anchor_idx=source_idx if use_categories else None,
                user_context=user_context,
                use_calories=use_calories,
                use_categories=use_categories,
            )
            for i in range(len(self.df))
        ]

        if user_context is None and use_calories and use_categories:
            self._name_similarity_cache[source_idx] = scores[:]
        return scores

    # ---------------------------------------------------------------------
    # Section 3: Model (cosine similarity)
    # ---------------------------------------------------------------------
    def _resolve_food_index(self, food_name: str) -> int:
        q = self._normalize_text(food_name)
        if q in self.name_to_indices:
            return self.name_to_indices[q][0]

        contains = self.df[self.df["name_norm"].str.contains(re.escape(q), na=False)]
        if not contains.empty:
            return int(contains.index[0])

        raise ValueError(f"Food name not found: {food_name}")

    def _base_similarity_for_index(self, idx: int) -> List[float]:
        return self._full_similarity_scores(idx)

    def _build_mask(
        self,
        diet: Optional[str] = None,
        prep_time: Optional[str] = None,
        cost: Optional[str] = None,
        taste: Optional[str] = None,
    ) -> List[bool]:
        mask = [True] * len(self.df)
        if diet is not None:
            values = self.df["diet"].eq(str(diet).lower().strip()).tolist()
            mask = [left and right for left, right in zip(mask, values)]
        if prep_time is not None:
            values = self.df["prep_time"].eq(str(prep_time).lower().strip()).tolist()
            mask = [left and right for left, right in zip(mask, values)]
        if cost is not None:
            values = self.df["cost"].eq(str(cost).lower().strip()).tolist()
            mask = [left and right for left, right in zip(mask, values)]
        if taste is not None:
            values = self.df["taste"].eq(str(taste).lower().strip()).tolist()
            mask = [left and right for left, right in zip(mask, values)]
        return mask

    def _apply_ingredient_overlap_boost(
        self,
        anchor_idx: int,
        candidate_idxs: Sequence[int],
        base_sims: Sequence[float],
        max_boost: float = 0.08,
    ) -> List[float]:
        # Small boost for ingredient overlap as requested.
        anchor_tokens = self.df.at[int(anchor_idx), "ingredient_tokens"]
        if not anchor_tokens:
            return list(base_sims)

        boosted = list(base_sims)
        for i, cand_idx in enumerate(candidate_idxs):
            cand_tokens = self.df.at[int(cand_idx), "ingredient_tokens"]
            if not cand_tokens:
                continue
            overlap = len(anchor_tokens.intersection(cand_tokens))
            if overlap > 0:
                overlap_ratio = overlap / max(1, len(anchor_tokens))
                boosted[i] = boosted[i] + max_boost * min(1.0, overlap_ratio)
        return boosted

    def _rank_indices(
        self,
        candidate_idxs: Sequence[int],
        sims: Sequence[float],
        top_n: int,
        sort_by: Optional[str],
        ascending: bool,
    ) -> List[int]:
        if sort_by is None:
            return sorted(range(len(sims)), key=lambda pos: sims[pos], reverse=True)[:top_n]

        s = str(sort_by).lower().strip()
        if s not in {"calories", "protein", "fat"}:
            return sorted(range(len(sims)), key=lambda pos: sims[pos], reverse=True)[:top_n]
        if s == "protein" and self.protein_var < 1e-6:
            return sorted(range(len(sims)), key=lambda pos: sims[pos], reverse=True)[:top_n]
        if s == "fat" and self.fat_var < 1e-6:
            return sorted(range(len(sims)), key=lambda pos: sims[pos], reverse=True)[:top_n]

        sub = [
            (pos, int(candidate_idxs[pos]), float(self.df.at[int(candidate_idxs[pos]), s]), float(sims[pos]))
            for pos in range(len(candidate_idxs))
        ]
        sub.sort(key=lambda item: (item[2], item[3]), reverse=not ascending)
        return [pos for pos, _, _, _ in sub[:top_n]]

    def _name_keywords(self, name: str) -> Set[str]:
        tokens = self._token_set(name)
        # Ignore generic words so diversity focuses on meaningful name tokens.
        stop_words = {
            "recipe", "food", "style", "easy", "quick", "best", "homemade", "fresh", "simple",
            "low", "fat", "free", "healthy", "hot", "cold",
        }
        return {t for t in tokens if t not in stop_words}

    def _diversify_ranked(
        self,
        ranked_idxs: Sequence[int],
        ranked_sims: Optional[Sequence[float]],
        top_n: int,
    ) -> Tuple[List[int], Optional[List[float]]]:
        selected_positions: List[int] = []
        seen_keywords: Set[str] = set()

        for pos, idx in enumerate(ranked_idxs):
            name_words = self._name_keywords(str(self.df.at[int(idx), "name"]))

            # If too similar to already selected names, skip.
            if len(name_words.intersection(seen_keywords)) > 1:
                continue

            selected_positions.append(pos)
            seen_keywords.update(name_words)
            if len(selected_positions) >= top_n:
                break

        # Backfill if diversity filtering is too strict.
        if len(selected_positions) < top_n:
            for pos in range(len(ranked_idxs)):
                if pos not in selected_positions:
                    selected_positions.append(pos)
                    if len(selected_positions) >= top_n:
                        break

        selected_idxs = [int(ranked_idxs[pos]) for pos in selected_positions]
        if ranked_sims is None:
            return selected_idxs, None
        return selected_idxs, [float(ranked_sims[pos]) for pos in selected_positions]

    # ---------------------------------------------------------------------
    # Section 4: Recommendation functions
    # ---------------------------------------------------------------------
    def recommend_by_name(
        self,
        food_name: str,
        top_n: int = 10,
        diet: Optional[str] = None,
        prep_time: Optional[str] = None,
        cost: Optional[str] = None,
        sort_by: Optional[str] = None,
        ascending: bool = True,
    ) -> pd.DataFrame:
        idx = self._resolve_food_index(food_name)
        sims = self._base_similarity_for_index(idx)
        sims[idx] = -1.0

        mask = self._build_mask(diet=diet, prep_time=prep_time, cost=cost)
        candidate_idxs = [i for i, keep in enumerate(mask) if keep and i != idx]
        if len(candidate_idxs) == 0:
            return pd.DataFrame(columns=["name", "calories", "diet", "explanation"])

        cand_sims = [sims[i] for i in candidate_idxs]
        cand_sims = self._apply_ingredient_overlap_boost(idx, candidate_idxs, cand_sims)

        shortlist_n = min(len(candidate_idxs), max(top_n * 6, top_n))
        order = self._rank_indices(candidate_idxs, cand_sims, shortlist_n, sort_by, ascending)
        ranked_idxs = [candidate_idxs[pos] for pos in order]
        ranked_sims = [cand_sims[pos] for pos in order]
        top_idxs, top_sims = self._diversify_ranked(ranked_idxs, ranked_sims, top_n)

        return self._format_results(
            top_idxs,
            sims=top_sims,
            anchor_idx=idx,
            user_context={
                "diet": str(diet).lower() if diet else "",
                "cost": str(cost).lower() if cost else "",
                "prep_time": str(prep_time).lower() if prep_time else "",
            },
        )

    def recommend_by_goal(self, goal: str, top_n: int = 10) -> pd.DataFrame:
        goal = str(goal).lower().strip()

        # 1) Filter FIRST by goal nutrition constraints.
        if goal == "weight_loss":
            subset = self.df[self.df["calories"] < 300].copy()
            # Avoid drinks/junk suggestions for weight loss.
            bad_terms = r"dessert|cake|cookie|brownie|ice cream|milkshake|soda|cola|soft drink|chips|fries|burger|pizza"
            bad_mask = subset["text_features"].str.contains(bad_terms, regex=True, na=False)
            subset = subset[~bad_mask]
            calorie_sort_asc = True
        elif goal == "weight_gain":
            subset = self.df[self.df["calories"] > 500].copy()
            calorie_sort_asc = False
        elif goal == "muscle_gain":
            subset = self.df[self.df["calories"].between(250, 600)].copy()
            protein_terms = r"chicken|egg|milk|paneer|beans|lentil|meat|protein"
            subset["muscle_priority"] = subset["ingredients"].str.contains(protein_terms, regex=True, na=False).astype(int)
            # Avoid desserts for muscle gain.
            dessert_mask = subset["text_features"].str.contains(r"dessert|cake|cookie|brownie|ice cream|pastry", regex=True, na=False)
            subset = subset[~dessert_mask]
            calorie_sort_asc = False
        elif goal == "maintain":
            subset = self.df[self.df["calories"].between(200, 500)].copy()
            calorie_sort_asc = True
        else:
            raise ValueError("Unsupported goal. Use: weight_loss, weight_gain, muscle_gain, maintain")

        # Graceful fallback if strict filtering becomes empty.
        if subset.empty:
            if goal == "weight_loss":
                subset = self.df[self.df["calories"] < 300].copy()
                calorie_sort_asc = True
            elif goal == "weight_gain":
                subset = self.df[self.df["calories"] > 500].copy()
                calorie_sort_asc = False
            elif goal == "muscle_gain":
                subset = self.df[self.df["calories"].between(250, 600)].copy()
                subset["muscle_priority"] = subset["ingredients"].str.contains(
                    r"chicken|egg|milk|paneer|beans|lentil|meat|protein", regex=True, na=False
                ).astype(int)
                calorie_sort_asc = False
            else:
                subset = self.df[self.df["calories"].between(200, 500)].copy()
                calorie_sort_asc = True

        # 2) If filtered set is too small, fallback to calorie-based ranking (no early TF-IDF).
        minimum_pool = max(top_n, 8)
        if len(subset) < minimum_pool:
            if goal == "muscle_gain" and "muscle_priority" in subset.columns:
                subset = subset.sort_values(by=["muscle_priority", "calories"], ascending=[False, False])
            elif goal == "maintain":
                target = 350.0
                subset = subset.assign(cal_gap=(subset["calories"] - target).abs()).sort_values(by="cal_gap", ascending=True)
            else:
                subset = subset.sort_values(by="calories", ascending=calorie_sort_asc)

            shortlist = subset.head(max(top_n * 6, top_n))
            top_idxs, _ = self._diversify_ranked(shortlist.index.tolist(), None, top_n)
            return self._format_results(top_idxs, sims=None, anchor_idx=None, goal=goal)

        # 3) Then apply similarity inside filtered set.
        if goal == "weight_loss":
            anchor_idx = int(subset["calories"].idxmin())
        elif goal == "weight_gain":
            anchor_idx = int(subset["calories"].idxmax())
        elif goal == "muscle_gain":
            prioritized = subset[subset.get("muscle_priority", 0) == 1]
            anchor_idx = int(prioritized["calories"].idxmax()) if not prioritized.empty else int(subset["calories"].idxmax())
        else:  # maintain
            target = 350.0
            anchor_idx = int((subset["calories"] - target).abs().idxmin())

        source_vector = self._doc_vectors[anchor_idx]
        source_norm = self._doc_norms[anchor_idx]
        source_calories = float(self.df.at[anchor_idx, "calories"])
        candidate_idxs = [int(i) for i in subset.index.tolist() if int(i) != anchor_idx]
        sims = [
            self._combined_similarity_for_index(
                candidate_idx=candidate_idx,
                source_vector=source_vector,
                source_norm=source_norm,
                source_calories=source_calories,
                anchor_idx=anchor_idx,
                use_calories=True,
                use_categories=True,
            )
            for candidate_idx in candidate_idxs
        ]
        sims = self._apply_ingredient_overlap_boost(anchor_idx, candidate_idxs, sims)

        if goal == "muscle_gain":
            priority_flags = self.df.loc[candidate_idxs, "ingredients"].str.contains(
                r"chicken|egg|milk|paneer|beans|lentil|meat|protein", regex=True, na=False
            ).astype(int).tolist()
            sims = [score + (flag * 0.08) for score, flag in zip(sims, priority_flags)]

        shortlist_n = min(len(candidate_idxs), max(top_n * 6, top_n))
        order = sorted(range(len(sims)), key=lambda pos: sims[pos], reverse=True)[:shortlist_n]
        ranked_idxs = [candidate_idxs[pos] for pos in order]
        ranked_sims = [sims[pos] for pos in order]
        top_idxs, top_sims = self._diversify_ranked(ranked_idxs, ranked_sims, top_n)

        return self._format_results(top_idxs, sims=top_sims, anchor_idx=anchor_idx, goal=goal)

    def recommend_custom(
        self,
        calories: Optional[float] = None,
        diet: Optional[str] = None,
        taste: Optional[str] = None,
        cost: Optional[str] = None,
        prep_time: Optional[str] = None,
        top_n: int = 10,
        sort_by: Optional[str] = None,
        ascending: bool = True,
    ) -> pd.DataFrame:
        mask = self._build_mask(diet=diet, prep_time=prep_time, cost=cost, taste=taste)
        subset = self.df[mask]

        if calories is not None:
            subset = subset[subset["calories"].between(float(calories) - 150, float(calories) + 150)]

        if subset.empty:
            return pd.DataFrame(columns=["name", "calories", "diet", "explanation"])

        if calories is not None:
            anchor_idx = int((subset["calories"] - float(calories)).abs().idxmin())
        else:
            anchor_idx = int(subset.index[0])

        source_vector = self._doc_vectors[anchor_idx]
        source_norm = self._doc_norms[anchor_idx]
        source_calories = float(self.df.at[anchor_idx, "calories"])
        candidate_idxs = [int(i) for i in subset.index.tolist() if int(i) != anchor_idx]
        sims = [
            self._combined_similarity_for_index(
                candidate_idx=candidate_idx,
                source_vector=source_vector,
                source_norm=source_norm,
                source_calories=source_calories,
                anchor_idx=anchor_idx,
                user_context={
                    "diet": str(diet).lower() if diet else "",
                    "cost": str(cost).lower() if cost else "",
                    "prep_time": str(prep_time).lower() if prep_time else "",
                },
                use_calories=True,
                use_categories=True,
            )
            for candidate_idx in candidate_idxs
        ]
        sims = self._apply_ingredient_overlap_boost(anchor_idx, candidate_idxs, sims)

        shortlist_n = min(len(candidate_idxs), max(top_n * 6, top_n))
        order = self._rank_indices(candidate_idxs, sims, shortlist_n, sort_by, ascending)
        ranked_idxs = [candidate_idxs[pos] for pos in order]
        ranked_sims = [sims[pos] for pos in order]
        top_idxs, top_sims = self._diversify_ranked(ranked_idxs, ranked_sims, top_n)

        return self._format_results(
            top_idxs,
            sims=top_sims,
            anchor_idx=anchor_idx,
            user_context={
                "diet": str(diet).lower() if diet else "",
                "cost": str(cost).lower() if cost else "",
                "prep_time": str(prep_time).lower() if prep_time else "",
            },
        )

    def recommend_by_ingredients(
        self,
        ingredient_list: List[str],
        top_n: int = 10,
        diet: Optional[str] = None,
        prep_time: Optional[str] = None,
        cost: Optional[str] = None,
    ) -> pd.DataFrame:
        query_text = " ".join(str(x).strip().lower() for x in ingredient_list if str(x).strip())
        if not query_text:
            return pd.DataFrame(columns=["name", "calories", "diet", "explanation"])

        query_vector, query_norm = self._vectorize_text(query_text)

        mask = self._build_mask(diet=diet, prep_time=prep_time, cost=cost)
        candidate_idxs = [i for i, keep in enumerate(mask) if keep]
        if len(candidate_idxs) == 0:
            return pd.DataFrame(columns=["name", "calories", "diet", "explanation"])

        cand_sims = [
            self.text_weight * self._cosine_similarity_dicts(query_vector, query_norm, self._doc_vectors[candidate_idx], self._doc_norms[candidate_idx])
            for candidate_idx in candidate_idxs
        ]

        query_tokens = self._token_set(query_text)
        if query_tokens:
            for i, cand_idx in enumerate(candidate_idxs):
                cand_tokens = self.df.at[int(cand_idx), "ingredient_tokens"]
                overlap = len(query_tokens.intersection(cand_tokens))
                if overlap > 0:
                    cand_sims[i] = cand_sims[i] + 0.08 * min(1.0, overlap / max(1, len(query_tokens)))

        shortlist_n = min(len(candidate_idxs), max(top_n * 6, top_n))
        order = sorted(range(len(cand_sims)), key=lambda pos: cand_sims[pos], reverse=True)[:shortlist_n]
        ranked_idxs = [candidate_idxs[pos] for pos in order]
        ranked_sims = [cand_sims[pos] for pos in order]
        top_idxs, top_sims = self._diversify_ranked(ranked_idxs, ranked_sims, top_n)

        return self._format_results(
            top_idxs,
            sims=top_sims,
            query_ingredients=[str(x).lower().strip() for x in ingredient_list],
            user_context={
                "diet": str(diet).lower() if diet else "",
                "cost": str(cost).lower() if cost else "",
                "prep_time": str(prep_time).lower() if prep_time else "",
            },
        )

    def chatbot_recommend(self, query: str, top_n: int = 10) -> pd.DataFrame:
        q = str(query).lower()

        kwargs: Dict[str, Optional[str]] = {
            "diet": None,
            "cost": None,
            "prep_time": None,
            "taste": None,
        }

        if "cheap" in q or "budget" in q or "low cost" in q:
            kwargs["cost"] = "low"
        if "veg" in q or "vegetarian" in q:
            kwargs["diet"] = "veg"
        if "quick" in q or "fast" in q:
            kwargs["prep_time"] = "quick"
        if "sweet" in q:
            kwargs["taste"] = "sweet"
        if "spicy" in q:
            kwargs["taste"] = "spicy"

        if "weight loss" in q or "healthy" in q or "low calorie" in q:
            return self.recommend_by_goal("weight_loss", top_n=top_n)
        if "weight gain" in q or "bulk" in q:
            return self.recommend_by_goal("weight_gain", top_n=top_n)
        if "muscle" in q or "protein" in q or "gym" in q:
            return self.recommend_by_goal("muscle_gain", top_n=top_n)

        ing_tokens = [w for w in re.findall(r"[a-zA-Z]+", q) if len(w) > 3]
        if ing_tokens:
            return self.recommend_by_ingredients(
                ingredient_list=ing_tokens[:6],
                top_n=top_n,
                diet=kwargs["diet"],
                prep_time=kwargs["prep_time"],
                cost=kwargs["cost"],
            )

        return self.recommend_custom(
            calories=None,
            diet=kwargs["diet"],
            taste=kwargs["taste"],
            cost=kwargs["cost"],
            prep_time=kwargs["prep_time"],
            top_n=top_n,
        )

    def demo_mode(self) -> None:
        print("\n=== DEMO: gym_user (muscle_gain) ===")
        print(self.recommend_by_goal("muscle_gain", top_n=5).to_string(index=False))

        print("\n=== DEMO: sick_user (light, low-fat proxy) ===")
        sick = self.recommend_custom(calories=float(self.df["calories"].quantile(0.25)), prep_time="quick", top_n=5)
        print(sick.to_string(index=False))

        print("\n=== DEMO: normal_user (maintain) ===")
        print(self.recommend_by_goal("maintain", top_n=5).to_string(index=False))

    def generate_meal_plan(self, goal: str) -> Dict[str, Dict[str, object]]:
        recs = self.recommend_by_goal(goal, top_n=9)
        if recs is None or recs.empty:
            return {"breakfast": {}, "lunch": {}, "dinner": {}}

        picks = recs.head(3).to_dict(orient="records")
        while len(picks) < 3:
            picks.append({})

        return {
            "breakfast": picks[0],
            "lunch": picks[1],
            "dinner": picks[2],
        }

    # ---------------------------------------------------------------------
    # Section 5: Explanation logic
    # ---------------------------------------------------------------------
    def _format_results(
        self,
        idxs: Sequence[int],
        sims: Optional[List[float]] = None,
        anchor_idx: Optional[int] = None,
        goal: Optional[str] = None,
        user_context: Optional[Dict[str, str]] = None,
        query_ingredients: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        out = self.df.loc[list(idxs), ["name", "calories", "protein", "fat", "diet", "ingredients", "steps"]].copy()

        out["health_label"] = out["calories"].apply(lambda x: self._health_label(float(x)))
        out["health_score"] = out.apply(
            lambda row: self.compute_health_score(
                float(row.get("calories", 0.0)),
                float(row.get("protein", 0.0)),
                float(row.get("fat", 0.0)),
            ),
            axis=1,
        )
        out["health_score_text"] = out["health_score"].apply(lambda x: f"{int(x)}/10")

        out["ingredients_items"] = out["ingredients"].fillna("").astype(str).apply(
            lambda s: [p.strip() for p in s.split(",") if p.strip()]
        )

        out["steps_items"] = out["steps"].fillna("").astype(str).apply(
            lambda s: [p.strip() for p in s.split("\n") if p.strip()]
        )
        out["steps_available"] = out["steps_items"].apply(lambda lst: len(lst) > 0)

        explanations: List[str] = []
        anchor_row = self.df.loc[anchor_idx] if anchor_idx is not None else None
        for i, idx in enumerate(idxs):
            sim = float(sims[i]) if sims is not None and i < len(sims) else None
            rec_row = self.df.loc[idx]
            explanations.append(self._explain(anchor_row, rec_row, goal, user_context, query_ingredients, sim))

        out["explanation"] = explanations
        return out[
            [
                "name",
                "calories",
                "diet",
                "health_label",
                "health_score",
                "health_score_text",
                "explanation",
                "ingredients_items",
                "steps_items",
                "steps_available",
            ]
        ].reset_index(drop=True)

    def _explain(
        self,
        anchor_row: Optional[pd.Series],
        rec_row: pd.Series,
        goal: Optional[str],
        user_context: Optional[Dict[str, str]],
        query_ingredients: Optional[List[str]],
        sim: Optional[float],
    ) -> str:
        reasons: List[str] = []

        goal_phrase = {
            "weight_loss": "weight loss",
            "weight_gain": "weight gain",
            "muscle_gain": "muscle gain",
            "maintain": "maintenance",
        }.get(goal)
        if goal_phrase:
            reasons.append(f"This dish matches your {goal_phrase} goal")

        if anchor_row is not None:
            overlap = sorted(list(anchor_row["ingredient_tokens"].intersection(rec_row["ingredient_tokens"])))
            if overlap:
                reasons.append(f"Contains similar ingredients like {', '.join(overlap[:3])}")

        if query_ingredients:
            q_set = {x.lower().strip() for x in query_ingredients if str(x).strip()}
            overlap = sorted(list(q_set.intersection(rec_row["ingredient_tokens"])))
            if overlap:
                reasons.append(f"Matches requested ingredients such as {', '.join(overlap[:3])}")

        cal = float(rec_row["calories"])
        label = self._health_label(cal)
        if label == "Low Calorie":
            reasons.append("It is low calorie")
        elif label == "High Energy":
            reasons.append("It is high energy")
        else:
            reasons.append("It is a balanced meal")

        if user_context and user_context.get("diet"):
            if rec_row["diet"] == user_context["diet"]:
                reasons.append("Matches your diet preference")
            else:
                reasons.append("Diet differs from your selected preference")

        explanation = ". ".join(reasons).strip()
        if explanation and not explanation.endswith("."):
            explanation += "."

        if sim is not None:
            explanation += f" Similarity score: {sim:.3f}."

        return explanation


_DEFAULT_RECOMMENDER: Optional[FoodRecommender] = None


def _get_default_recommender() -> FoodRecommender:
    global _DEFAULT_RECOMMENDER
    if _DEFAULT_RECOMMENDER is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        primary = os.path.join(base_dir, "final_food_dataset_final.csv")
        fallback = os.path.join(base_dir, "final_food_dataset.csv")
        dataset_path = primary if os.path.exists(primary) else fallback
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("Could not find final_food_dataset_final.csv or final_food_dataset.csv")
        _DEFAULT_RECOMMENDER = FoodRecommender(dataset_path)
    return _DEFAULT_RECOMMENDER


def suggest_food_names(query: str, limit: int = 5) -> List[str]:
    rec = _get_default_recommender()
    q = rec._normalize_text(query)
    if not q:
        return []

    exact = rec.df[rec.df["name_norm"].str.contains(re.escape(q), na=False)]["name"].head(limit).tolist()
    if exact:
        return exact

    return get_close_matches(q, rec.df["name_norm"].tolist(), n=limit, cutoff=0.45)


def recommend_by_name(food_name: str, top_n: int = 10, **kwargs) -> pd.DataFrame:
    return _get_default_recommender().recommend_by_name(food_name, top_n=top_n, **kwargs)


def recommend_by_goal(goal: str, top_n: int = 10) -> pd.DataFrame:
    return _get_default_recommender().recommend_by_goal(goal, top_n=top_n)


def recommend_custom(
    calories: Optional[float] = None,
    diet: Optional[str] = None,
    taste: Optional[str] = None,
    cost: Optional[str] = None,
    prep_time: Optional[str] = None,
    top_n: int = 10,
    **kwargs,
) -> pd.DataFrame:
    return _get_default_recommender().recommend_custom(
        calories=calories,
        diet=diet,
        taste=taste,
        cost=cost,
        prep_time=prep_time,
        top_n=top_n,
        **kwargs,
    )


def recommend_by_ingredients(ingredient_list: List[str], top_n: int = 10, **kwargs) -> pd.DataFrame:
    return _get_default_recommender().recommend_by_ingredients(ingredient_list, top_n=top_n, **kwargs)


def chatbot_recommend(query: str, top_n: int = 10) -> pd.DataFrame:
    return _get_default_recommender().chatbot_recommend(query, top_n=top_n)


def compute_health_score(calories: float, protein: float = 0.0, fat: float = 0.0) -> int:
    return FoodRecommender.compute_health_score(calories, protein, fat)


def generate_meal_plan(goal: str) -> Dict[str, Dict[str, object]]:
    return _get_default_recommender().generate_meal_plan(goal)


def demo_mode() -> None:
    _get_default_recommender().demo_mode()


def print_recommendation_card(row: pd.Series) -> None:
    print(f"Food: {row['name']}")
    print(f"Calories: {row['calories']:.1f}")
    print(f"Diet: {row['diet']}")
    print(f"Reason: {row['explanation']}")
    print("-" * 60)


def print_recommendation_results(results: pd.DataFrame, empty_message: str = "No results found.") -> None:
    if results is None or results.empty:
        print(empty_message)
        return

    for _, row in results.iterrows():
        print_recommendation_card(row)


def prompt_int_choice(prompt: str, valid_choices: Set[int]) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if value in valid_choices:
                return value
        except ValueError:
            pass
        print(f"Please enter one of these options: {sorted(valid_choices)}")


def prompt_optional_text(prompt: str) -> Optional[str]:
    raw = input(prompt).strip()
    return raw if raw else None


def cli_recommend_by_name(rec: FoodRecommender) -> None:
    food_name = input("Enter food name: ").strip()
    if not food_name:
        print("No food name entered.")
        return
    try:
        results = rec.recommend_by_name(food_name, top_n=10)
        print_recommendation_results(results)
    except ValueError:
        print(f"Food name not found: {food_name}")
        suggestions = suggest_food_names(food_name)
        if suggestions:
            print("Did you mean:")
            for item in suggestions:
                print(f"- {item}")
        else:
            print("No close matches found.")


def cli_recommend_by_goal(rec: FoodRecommender) -> None:
    goal = input("Enter goal (weight_loss, weight_gain, muscle_gain, maintain): ").strip().lower()
    try:
        results = rec.recommend_by_goal(goal, top_n=10)
        print_recommendation_results(results)
    except ValueError as exc:
        print(str(exc))


def cli_recommend_custom(rec: FoodRecommender) -> None:
    calories_raw = prompt_optional_text("Calories target (press Enter to skip): ")
    diet = prompt_optional_text("Diet (veg/non-veg, press Enter to skip): ")
    taste = prompt_optional_text("Taste (sweet/spicy, press Enter to skip): ")
    cost = prompt_optional_text("Cost (low/medium/high, press Enter to skip): ")
    prep_time = prompt_optional_text("Prep time (quick/medium/long, press Enter to skip): ")

    calories = float(calories_raw) if calories_raw else None
    results = rec.recommend_custom(
        calories=calories,
        diet=diet,
        taste=taste,
        cost=cost,
        prep_time=prep_time,
        top_n=10,
    )
    print_recommendation_results(results)


def cli_recommend_by_ingredients(rec: FoodRecommender) -> None:
    raw = input("Enter ingredients separated by commas: ").strip()
    ingredient_list = [x.strip() for x in raw.split(",") if x.strip()]
    if not ingredient_list:
        print("No ingredients entered.")
        return
    results = rec.recommend_by_ingredients(ingredient_list, top_n=10)
    print_recommendation_results(results)


def cli_chatbot(rec: FoodRecommender) -> None:
    query = input("Enter your query: ").strip()
    if not query:
        print("No query entered.")
        return
    results = rec.chatbot_recommend(query, top_n=10)
    print_recommendation_results(results)


# -------------------------------------------------------------------------
# Section 6: main() demo
# -------------------------------------------------------------------------
def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    primary = os.path.join(base_dir, "final_food_dataset_final.csv")
    fallback = os.path.join(base_dir, "final_food_dataset.csv")

    dataset_path = primary if os.path.exists(primary) else fallback
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Could not find final_food_dataset_final.csv or final_food_dataset.csv")

    rec = FoodRecommender(dataset_path)
    print("Smart Food Recommendation System")
    print(f"Loaded rows: {len(rec.df)} from {os.path.basename(dataset_path)}")
    print(
        "Feature weights -> "
        f"TF-IDF: {rec.text_weight}, Calories: {rec.calorie_weight}, Categorical: {rec.cat_weight}"
    )

    while True:
        print("\nChoose a mode:")
        print("1 -> recommend by food name")
        print("2 -> goal-based")
        print("3 -> custom input")
        print("4 -> ingredient-based")
        print("5 -> chatbot query")
        print("6 -> demo mode")
        print("0 -> exit")

        choice = prompt_int_choice("Enter choice: ", {0, 1, 2, 3, 4, 5, 6})

        if choice == 0:
            print("Goodbye.")
            break
        if choice == 1:
            cli_recommend_by_name(rec)
        elif choice == 2:
            cli_recommend_by_goal(rec)
        elif choice == 3:
            cli_recommend_custom(rec)
        elif choice == 4:
            cli_recommend_by_ingredients(rec)
        elif choice == 5:
            cli_chatbot(rec)
        elif choice == 6:
            rec.demo_mode()


if __name__ == "__main__":
    main()
