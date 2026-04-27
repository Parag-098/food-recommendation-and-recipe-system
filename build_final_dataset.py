import ast
import csv
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_OUTPUT = os.path.join(ROOT_DIR, "final_food_dataset.csv")
RANDOM_SEED = 42
MAX_RECIPES = 20000
MATCH_THRESHOLD = 80


def detect_delimiter(file_path: str) -> str:
    """Detect CSV delimiter using a small file sample."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)

    if not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def inspect_csv_files(root_dir: str) -> List[Dict[str, object]]:
    """Inspect all CSV files and return filename + columns + delimiter metadata."""
    csv_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(".csv")])
    inspection: List[Dict[str, object]] = []

    for file_name in csv_files:
        full_path = os.path.join(root_dir, file_name)
        delimiter = detect_delimiter(full_path)

        try:
            header_df = pd.read_csv(full_path, sep=delimiter, nrows=0)
            columns = [str(c) for c in header_df.columns]
        except Exception:
            columns = []

        inspection.append(
            {
                "file": file_name,
                "path": full_path,
                "delimiter": delimiter,
                "columns": columns,
                "column_set": set(c.strip().lower() for c in columns),
            }
        )

    return inspection


def print_inspection(inspection: List[Dict[str, object]]) -> None:
    print("\n=== CSV INSPECTION: FILES AND COLUMNS ===")
    for item in inspection:
        print(f"\nFile: {item['file']}")
        print(f"Delimiter: {repr(item['delimiter'])}")
        print("Columns:")
        if item["columns"]:
            for col in item["columns"]:
                print(f"  - {col}")
        else:
            print("  - <unable to read columns>")


def _find_best_file(
    inspection: List[Dict[str, object]], required_cols: Set[str], preferred_cols: Set[str]
) -> Optional[Dict[str, object]]:
    """Find the file that best matches required/preferred schema columns."""
    candidates = []
    for item in inspection:
        col_set = item["column_set"]
        if required_cols.issubset(col_set):
            score = len(preferred_cols.intersection(col_set))
            candidates.append((score, item))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def auto_select_files(inspection: List[Dict[str, object]]) -> Dict[str, Optional[Dict[str, object]]]:
    selections = {
        "foodcom_recipes": _find_best_file(
            inspection,
            required_cols={"name", "ingredients", "tags"},
            preferred_cols={"minutes", "nutrition", "id", "n_ingredients"},
        ),
        "usda_food": _find_best_file(
            inspection,
            required_cols={"fdc_id", "description"},
            preferred_cols={"data_type", "food_category_id"},
        ),
        "usda_nutrient_lookup": _find_best_file(
            inspection,
            required_cols={"id", "name", "unit_name"},
            preferred_cols={"nutrient_nbr", "rank"},
        ),
        "usda_food_nutrient": _find_best_file(
            inspection,
            required_cols={"fdc_id", "nutrient_id", "amount"},
            preferred_cols={"id", "data_points", "derivation_id"},
        ),
    }
    return selections


def print_selected_files(selections: Dict[str, Optional[Dict[str, object]]]) -> None:
    print("\n=== AUTO-SELECTED RELEVANT FILES ===")
    for key, item in selections.items():
        if item is None:
            print(f"- {key}: <NOT FOUND>")
        else:
            print(f"- {key}: {item['file']}")


def safe_parse_list(value) -> List:
    """Parse list-like cell content from Food.com files safely."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except Exception:
        return [part.strip() for part in text.split(",") if part.strip()]


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_foodcom_base(path: str, limit: int = 1000) -> pd.DataFrame:
    required = ["name", "ingredients", "tags", "minutes", "nutrition", "id"]
    header = pd.read_csv(path, nrows=0)
    available_cols = [c for c in required if c in header.columns]

    df = pd.read_csv(path, usecols=available_cols, nrows=limit)

    if "name" not in df.columns:
        raise ValueError("Selected Food.com file does not contain a 'name' column.")

    if "ingredients" not in df.columns:
        df["ingredients"] = ""
    if "tags" not in df.columns:
        df["tags"] = "[]"
    if "minutes" not in df.columns:
        df["minutes"] = np.nan
    if "nutrition" not in df.columns:
        df["nutrition"] = "[]"

    df["ingredients_list"] = df["ingredients"].apply(safe_parse_list)
    df["tags_list"] = df["tags"].apply(safe_parse_list)
    df["nutrition_list"] = df["nutrition"].apply(safe_parse_list)

    def _pick_nutrition_val(nut_list: List, idx: int) -> Optional[float]:
        try:
            return float(nut_list[idx])
        except Exception:
            return np.nan

    # Food.com commonly stores calories at index 0 in nutrition array.
    df["foodcom_calories"] = df["nutrition_list"].apply(lambda x: _pick_nutrition_val(x, 0))

    df["name"] = df["name"].astype(str).str.strip()
    df = df[df["name"].str.len() > 0].copy()

    df["ingredients"] = df["ingredients_list"].apply(
        lambda xs: ", ".join(str(x).strip() for x in xs if str(x).strip())
    )
    df["tags"] = df["tags_list"].apply(
        lambda xs: ", ".join(str(x).strip() for x in xs if str(x).strip())
    )

    df["name_norm"] = df["name"].apply(normalize_text)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")

    # Keep rows as-is so output can reach the configured maximum limit.
    df = df.reset_index(drop=True)
    return df


def detect_target_nutrient_ids(nutrient_lookup_path: str) -> Dict[str, int]:
    ndf = pd.read_csv(nutrient_lookup_path, usecols=["id", "name", "unit_name", "nutrient_nbr"])
    ndf["name_norm"] = ndf["name"].astype(str).str.lower()
    ndf["unit_name"] = ndf["unit_name"].astype(str).str.upper()

    ids: Dict[str, Optional[int]] = {"calories": None, "protein": None, "fat": None}

    # Canonical USDA mappings should win if they exist.
    available_ids = set(pd.to_numeric(ndf["id"], errors="coerce").dropna().astype(int).tolist())
    if 1008 in available_ids:
        ids["calories"] = 1008
    if 1003 in available_ids:
        ids["protein"] = 1003
    if 1004 in available_ids:
        ids["fat"] = 1004

    # Fill any remaining gaps using name/unit heuristics.
    for _, row in ndf.iterrows():
        nid = int(row["id"])
        name = row["name_norm"]
        unit = row["unit_name"]

        if ids["calories"] is None and (nid == 1008 or ("energy" in name and unit == "KCAL")):
            ids["calories"] = nid
        if ids["protein"] is None and (nid == 1003 or name == "protein"):
            ids["protein"] = nid
        if ids["fat"] is None and (nid == 1004 or "total lipid" in name or name == "fat"):
            ids["fat"] = nid

    missing = [k for k, v in ids.items() if v is None]
    if missing:
        raise ValueError(f"Could not detect nutrient IDs for: {missing}")

    return {k: int(v) for k, v in ids.items()}  # type: ignore[arg-type]


def build_usda_nutrition_table(
    food_path: str,
    food_nutrient_path: str,
    nutrient_ids: Dict[str, int],
    chunksize: int = 300000,
) -> pd.DataFrame:
    target_ids = {nutrient_ids["calories"], nutrient_ids["protein"], nutrient_ids["fat"]}

    parts: List[pd.DataFrame] = []
    usecols = ["fdc_id", "nutrient_id", "amount"]

    for chunk in pd.read_csv(food_nutrient_path, usecols=usecols, chunksize=chunksize):
        chunk["nutrient_id"] = pd.to_numeric(chunk["nutrient_id"], errors="coerce")
        chunk = chunk[chunk["nutrient_id"].isin(target_ids)].copy()
        if not chunk.empty:
            parts.append(chunk)

    if not parts:
        raise ValueError("No USDA nutrient records found for calories/protein/fat IDs.")

    nut = pd.concat(parts, ignore_index=True)

    pivot = (
        nut.pivot_table(index="fdc_id", columns="nutrient_id", values="amount", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    col_map = {
        nutrient_ids["calories"]: "calories",
        nutrient_ids["protein"]: "protein",
        nutrient_ids["fat"]: "fat",
    }
    pivot = pivot.rename(columns=col_map)

    for col in ["calories", "protein", "fat"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    food = pd.read_csv(food_path, usecols=["fdc_id", "description"])
    food = food.rename(columns={"description": "food_name"})
    food["food_name"] = food["food_name"].astype(str)
    food["food_name_norm"] = food["food_name"].apply(normalize_text)

    usda = food.merge(pivot[["fdc_id", "calories", "protein", "fat"]], on="fdc_id", how="left")

    usda = usda.drop_duplicates(subset=["food_name_norm"]).reset_index(drop=True)
    return usda


def build_token_index(names: List[str]) -> Tuple[Dict[str, List[int]], List[Set[str]]]:
    token_to_indices: Dict[str, List[int]] = defaultdict(list)
    token_sets: List[Set[str]] = []

    for idx, name in enumerate(names):
        tokens = {t for t in name.split() if len(t) > 2}
        token_sets.append(tokens)
        for tok in tokens:
            token_to_indices[tok].append(idx)

    return token_to_indices, token_sets


def fast_token_overlap_match(
    query: str,
    token_index: Dict[str, List[int]],
    token_sets: List[Set[str]],
) -> Tuple[Optional[int], float]:
    """Approximate matching by token overlap for better scalability on 20k rows."""
    q_tokens = {t for t in query.split() if len(t) > 2}
    if not q_tokens:
        return None, 0.0

    counts = Counter()
    for tok in q_tokens:
        for idx in token_index.get(tok, []):
            counts[idx] += 1

    if not counts:
        return None, 0.0

    # Keep top overlap candidates only to control runtime.
    candidate_indices = [idx for idx, _ in counts.most_common(120)]

    best_idx = None
    best_score = 0.0
    for idx in candidate_indices:
        overlap = len(q_tokens.intersection(token_sets[idx]))
        score = (overlap / max(1, len(q_tokens))) * 100.0
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx, best_score


def match_recipes_to_usda(
    recipes: pd.DataFrame,
    usda: pd.DataFrame,
    threshold: int = 80,
) -> pd.DataFrame:
    usda = usda.copy()
    usda = usda[usda["food_name_norm"].str.len() > 0].reset_index(drop=True)

    all_names = usda["food_name_norm"].tolist()
    token_index, token_sets = build_token_index(all_names)
    exact_index = {name: idx for idx, name in enumerate(all_names)}

    matched_rows = []

    for _, row in recipes.iterrows():
        query = row["name_norm"]
        best_idx = None
        score = 0.0

        if query in exact_index:
            best_idx = exact_index[query]
            score = 100.0
        else:
            best_idx, score = fast_token_overlap_match(query, token_index, token_sets)

        if best_idx is not None and score >= threshold:
            matched = usda.iloc[int(best_idx)]
            calories = matched.get("calories", np.nan)
            protein = matched.get("protein", np.nan)
            fat = matched.get("fat", np.nan)
            matched_name = matched.get("food_name", "")
            matched_flag = True
        else:
            calories = np.nan
            protein = np.nan
            fat = np.nan
            matched_name = ""
            matched_flag = False

        matched_rows.append(
            {
                "name": row["name"],
                "ingredients": row["ingredients"],
                "tags": row["tags"],
                "foodcom_calories": row.get("foodcom_calories", np.nan),
                "minutes": row.get("minutes", np.nan),
                "usda_match_name": matched_name,
                "match_score": score,
                "matched": matched_flag,
                "calories": calories,
                "protein": protein,
                "fat": fat,
            }
        )

    return pd.DataFrame(matched_rows)


def derive_columns_and_fill(df: pd.DataFrame, rng_seed: int = 42) -> pd.DataFrame:
    out = df.copy()

    # Keep missing after merge first, then fallback fill where needed.
    out["calories"] = pd.to_numeric(out["calories"], errors="coerce")
    out["protein"] = pd.to_numeric(out["protein"], errors="coerce")
    out["fat"] = pd.to_numeric(out["fat"], errors="coerce")

    # Use Food.com calories where USDA calories missing, before statistical fallback.
    out["foodcom_calories"] = pd.to_numeric(out["foodcom_calories"], errors="coerce")
    out["calories"] = out["calories"].fillna(out["foodcom_calories"])

    before_missing = out[["calories", "protein", "fat"]].isna().sum().to_dict()

    # Realistic median fallback from available values.
    fallback_defaults = {"calories": 250.0, "protein": 10.0, "fat": 8.0}
    for col in ["calories", "protein", "fat"]:
        med = out[col].median(skipna=True)
        fill_val = float(med) if pd.notna(med) else fallback_defaults[col]
        out[col] = out[col].fillna(fill_val)

    after_missing = out[["calories", "protein", "fat"]].isna().sum().to_dict()

    tags_lower = out["tags"].fillna("").astype(str).str.lower()
    out["diet"] = np.where(tags_lower.str.contains("vegetarian"), "veg", "non-veg")
    out["taste"] = np.where(tags_lower.str.contains("dessert"), "sweet", "spicy")

    random.seed(rng_seed)
    np.random.seed(rng_seed)

    out["cost"] = np.random.choice(["low", "medium", "high"], size=len(out), p=[0.4, 0.4, 0.2])
    out["prep_time"] = np.random.choice(["quick", "medium", "long"], size=len(out), p=[0.4, 0.4, 0.2])

    # Final cleaning.
    out["name"] = out["name"].astype(str).str.strip()
    out["ingredients"] = out["ingredients"].fillna("").astype(str)
    out["tags"] = out["tags"].fillna("").astype(str)

    out = out[out["name"].str.len() > 0].copy()

    for col in ["calories", "protein", "fat"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0)

    print("\n=== MISSINGNESS SUMMARY (NUTRITION) ===")
    print("Before fallback fill:", before_missing)
    print("After fallback fill:", after_missing)

    return out


def build_final_dataset(root_dir: str) -> pd.DataFrame:
    inspection = inspect_csv_files(root_dir)
    print_inspection(inspection)

    selections = auto_select_files(inspection)
    print_selected_files(selections)

    missing_required = [k for k, v in selections.items() if v is None]
    if missing_required:
        raise RuntimeError(f"Could not auto-select required files: {missing_required}")

    foodcom_path = selections["foodcom_recipes"]["path"]  # type: ignore[index]
    usda_food_path = selections["usda_food"]["path"]  # type: ignore[index]
    usda_nutrient_lookup_path = selections["usda_nutrient_lookup"]["path"]  # type: ignore[index]
    usda_food_nutrient_path = selections["usda_food_nutrient"]["path"]  # type: ignore[index]

    recipes = read_foodcom_base(foodcom_path, limit=MAX_RECIPES)
    nutrient_ids = detect_target_nutrient_ids(usda_nutrient_lookup_path)

    print("\n=== DETECTED USDA NUTRIENT IDS ===")
    print(nutrient_ids)

    usda = build_usda_nutrition_table(
        usda_food_path,
        usda_food_nutrient_path,
        nutrient_ids=nutrient_ids,
    )

    merged = match_recipes_to_usda(recipes, usda, threshold=MATCH_THRESHOLD)

    print("\n=== MATCH SUMMARY ===")
    matched_count = int(merged["matched"].sum())
    print(f"Matched rows: {matched_count} / {len(merged)}")
    print(f"Unmatched rows: {len(merged) - matched_count}")

    final_df = derive_columns_and_fill(merged, rng_seed=RANDOM_SEED)

    final_df = final_df[
        [
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
    ].reset_index(drop=True)

    return final_df


def main() -> None:
    final_df = build_final_dataset(ROOT_DIR)

    print("\n=== FINAL DATASET PREVIEW (FIRST 5 ROWS) ===")
    print(final_df.head(5).to_string(index=False))

    print("\n=== FINAL COLUMN LIST ===")
    print(final_df.columns.tolist())

    try:
        final_df.to_csv(FINAL_OUTPUT, index=False)
        saved_path = FINAL_OUTPUT
    except PermissionError:
        fallback_output = os.path.join(ROOT_DIR, "final_food_dataset_latest.csv")
        final_df.to_csv(fallback_output, index=False)
        saved_path = fallback_output
        print(
            "\nWarning: Could not overwrite final_food_dataset.csv because it is locked. "
            "Saved to final_food_dataset_latest.csv instead."
        )

    print(f"\nSaved final dataset to: {saved_path}")
    print(f"Final row count: {len(final_df)}")


if __name__ == "__main__":
    main()
