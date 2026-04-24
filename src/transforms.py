# ---- Addendum for notebook orchestration (paste/append this) ----
# Provides: numeric_features, categorical_features, get_preprocessor,
#           get_feature_names_from_preprocessor, select_cols_top7, ColumnSelector
# These helpers are intentionally small and importable so joblib/pickle can find them.

from typing import List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator

def select_cols_indices(X, indices):
    """
    Portable selector function that returns X[:, indices].
    Used by FunctionTransformer with kwargs {'indices': indices}.
    """
    X = np.asarray(X)
    return X[:, indices]


def to_string_block(X):
    """
    Convert an array-like or DataFrame chunk to a DataFrame with stringified values
    (preserving NaN). This is top-level so FunctionTransformer can reference it.
    """
    if hasattr(X, "iloc"):
        df = X.copy()
    else:
        # If numpy array, wrap it in DataFrame with numeric column names
        df = pd.DataFrame(X)
    for col in df.columns:
        # preserve NaN, else convert to str
        df[col] = df[col].where(pd.notna(df[col]), np.nan).astype(object).apply(lambda v: str(v) if pd.notna(v) else v)
    return df

def fill_with_string(X, fill_value: str):
    """
    Fill NaNs with fill_value and return underlying numpy array.
    Keep it top-level and parameterize by creating a small wrapper if needed.
    Note: FunctionTransformer will call this with only X; to use different fill_value,
    create a tiny top-level wrapper function bound to that value (still top-level).
    """
    df = pd.DataFrame(X)
    return df.fillna(fill_value).values


# 1) canonical feature lists for the UCI heart dataset
# If your CSV uses slightly different column names, modify these lists accordingly.
numeric_features: List[str] = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_features: List[str] = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# 2) ColumnSelector helper class (picklable)
class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Select columns by name (if passed a DataFrame) or by integer indices (if passed numpy array).
    Picklable because it's a named top-level class in this module.
    """
    def __init__(self, columns):
        # columns: list of strings (column names) OR list of ints (indices)
        self.columns = list(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "iloc") and isinstance(self.columns[0], str):
            # DataFrame + names
            return X[self.columns].to_numpy()
        else:
            arr = np.asarray(X)
            # treat columns as indices
            return arr[:, self.columns]

# 3) select_cols_top7: portable selector used by FunctionTransformer or direct calls
def select_cols_top7(X, indices):
    """
    Takes array-like X (numpy array or pandas DataFrame) and returns X with selected column indices.
    indices: list-like of integer indices referring to columns in the transformed (preprocessor) array.
    This function is intentionally top-level so joblib can find it at import time.
    """
    if isinstance(X, pd.DataFrame):
        # If a DataFrame is passed, try to convert to numpy first (expected: preprocessor.transform output usually)
        return X.values[:, indices]
    else:
        arr = np.asarray(X)
        return arr[:, indices]

def canonicalize_categorical_df(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """
    Convert listed categorical columns to consistent strings while preserving NaN.
    Returns a COPY of df (does not mutate input).
    Robust for bool/numpy.bool_/pandas BooleanDtype, ints, and already-string values.
    """
    out = df.copy()
    for c in cat_cols:
        if c not in out.columns:
            continue
        s = out[c]
        # Replace pandas boolean extension NA with np.nan, then convert non-nulls to str
        # Use vectorized operations for speed and correctness
        mask_na = s.isna()
        # Convert values to string for non-missing entries (works for bool, numpy.bool_, ints, strings)
        # Use .astype(str) but avoid turning nan into 'nan' by applying mask
        converted = s.astype(str)
        converted = converted.where(~mask_na, np.nan)
        # Optional: normalize casing/trimming (choose one consistent convention)
        # converted = converted.str.strip().str.capitalize()   # example normalization
        out[c] = converted
    return out
    
    
# 4) get_preprocessor() - builds (but does NOT fit) a ColumnTransformer for our dataset
# ---------- UPDATED: flexible preprocessor builder ----------
def get_preprocessor(impute_num_strategy="median", impute_cat_strategy="most_frequent",
                     impute_cat_fill_value=None, onehot_handle_unknown="ignore"):
    """
    Returns an unfitted ColumnTransformer configured for the UCI heart dataset.
    Fit this on training data only.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_num_strategy)),
        ("scaler", StandardScaler())
    ])

    # For categorical imputation: use SimpleImputer (no nested functions)
    if impute_cat_fill_value is None:
        cat_imputer = SimpleImputer(strategy=impute_cat_strategy)
    else:
        # constant fill (will produce strings like 'missing' or whatever you pass)
        cat_imputer = SimpleImputer(strategy="constant", fill_value=impute_cat_fill_value)

    cat_pipeline = Pipeline([
        ("cast_to_str", FunctionTransformer(to_string_block, validate=False)),
        ("imputer", cat_imputer),
        ("onehot", OneHotEncoder(handle_unknown=onehot_handle_unknown, sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop",
        sparse_threshold=0
    )
    return preprocessor


# 5) get_feature_names_from_preprocessor(preprocessor, input_columns)
def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer, input_columns: List[str]):
    """
    Reconstruct transformed feature names from a fitted ColumnTransformer.
    - preprocessor must be fitted (so transformers_ exists).
    - input_columns: original column names list passed when fitting preprocessor.
    Returns a list of feature names in the same order as preprocessor.transform(X) columns.
    """
    names = []
    # Iterate through fitted transformers
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        # If transformer is a pipeline, get last step
        if hasattr(transformer, "named_steps"):
            last = list(transformer.named_steps.values())[-1]
            if hasattr(last, "get_feature_names_out"):
                try:
                    # Some sklearn objects require passing input feature names
                    out = last.get_feature_names_out(cols)
                except TypeError:
                    out = last.get_feature_names_out()
                names.extend([str(x) for x in out])
            else:
                names.extend(list(cols))
        else:
            if hasattr(transformer, "get_feature_names_out"):
                names.extend([str(x) for x in transformer.get_feature_names_out(cols)])
            else:
                names.extend(list(cols))
    return names



# ---------- NEW: missing-indicator helper ----------
def add_missing_indicators(df: pd.DataFrame, cols: List[str]):
    """
    Add binary columns named <col>_missing for every col in cols that exists in df.
    Returns a copy of df with added columns.
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c + "_missing"] = df[c].isna().astype(int)
    return df

def get_preprocessor_fixed(categories_list: List[List[str]]):
    """
    Build a ColumnTransformer like get_preprocessor(), but with OneHotEncoder(categories=categories_list).
    categories_list: list of category lists in the same order as categorical_features.
    Returns an unfitted ColumnTransformer (fit it on training data inside CV pipeline or before saving).
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, categories=categories_list))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0
    )
    return preprocessor

def compute_categories_list(X: pd.DataFrame, categorical_cols: List[str]):
    """Return categories_list for OneHotEncoder in the same order as categorical_cols.
       Converts boolean-like to strings to ensure homogeneous types per column.
    """
    cats = []
    for c in categorical_cols:
        col = X[c].dropna()
        # If dtype is bool or mixed bool/str, coerce to str for consistency
        if col.dtype == 'bool' or col.map(type).apply(lambda t: t is bool).any():
            col = col.astype(str)
        # ensure categories are sorted for reproducibility
        unique_vals = sorted(col.astype(str).unique().tolist())
        cats.append(unique_vals)
    return cats

# ----------------- end addendum -----------------
