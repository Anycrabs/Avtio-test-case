"""Обучение модели и формирование submission.csv.

Пример запуска из корня задания, где находятся data/ и sample_submission.csv:

    python solution/train.py --data-root .

Для локальной проверки по времени:

    python solution/train.py --data-root . --validate

Финальный score строится как смесь рангов двух CatBoost-моделей:
1) обычной модели на всей доступной обучающей истории;
2) модели с повышенным весом более свежих дней.

Такой вариант устойчивее одного случайного holdout и лучше соответствует ситуации,
когда test расположен строго позже train по времени.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from feature_engineering import CATEGORICAL_FEATURES, build_features
from feature_order import FEATURE_ORDER
from metric import precision_at_recall


RANDOM_STATE = 42
RECENCY_RANDOM_STATE = 17
RECENCY_HALF_LIFE_DAYS = 3.0
BASE_WEIGHT = 0.70
RECENCY_WEIGHT = 0.30

# Эти параметры оставлены умеренными специально: в этой задаче основной прирост
# даёт инженерия признаков и корректная временная проверка, а не глубокие деревья.
BASE_CATBOOST_PARAMS = dict(
    iterations=780,
    depth=6,
    learning_rate=0.045,
    l2_leaf_reg=7,
    random_strength=0.5,
    loss_function="Logloss",
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1,
)

RECENCY_CATBOOST_PARAMS = dict(
    iterations=650,
    depth=6,
    learning_rate=0.045,
    l2_leaf_reg=8,
    random_strength=0.5,
    loss_function="Logloss",
    random_seed=RECENCY_RANDOM_STATE,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1,
)

# Два последних временных окна используются как stress-test устойчивости.
# Одного последнего holdout недостаточно: метрика P@R>=0.70 заметно меняется
# от нескольких переставленных объектов около рабочего порога.
ROLLING_FOLDS = [
    (pd.Timestamp("2026-04-16"), pd.Timestamp("2026-04-19")),
    (pd.Timestamp("2026-04-17"), pd.Timestamp("2026-04-20")),
]


class RussianArgumentParser(argparse.ArgumentParser):
    """Небольшая локализация стандартной справки argparse."""

    def format_help(self) -> str:
        text = super().format_help()
        return (
            text.replace("usage:", "использование:")
            .replace("options:", "параметры:")
            .replace("show this help message and exit", "показать эту справку и выйти")
        )


def parse_args() -> argparse.Namespace:
    parser = RussianArgumentParser(description="Обучение модели для определения автоматизированного трафика")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Корневая папка задания с data/ и sample_submission.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Путь для сохранения предсказаний",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Перед финальным обучением выполнить rolling temporal validation",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Использовать только основную CatBoost-модель без recency-модели",
    )
    return parser.parse_args()


def load_data(root: Path):
    """Загрузить исходные файлы и преобразовать временные поля."""
    train = pd.read_csv(
        root / "data" / "train.csv",
        parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"],
    )
    test = pd.read_csv(
        root / "data" / "test.csv",
        parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"],
    )
    events = pd.read_csv(root / "data" / "events.csv.gz", parse_dates=["event_ts"])
    sample = pd.read_csv(root / "sample_submission.csv")
    return train, test, events, sample


def audit_data(train: pd.DataFrame, test: pd.DataFrame, events: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Базовые проверки данных и диагностика временной утечки."""
    print("=== ПРОВЕРКА ДАННЫХ ===")
    print(f"Размер train:  {train.shape}")
    print(f"Размер test:   {test.shape}")
    print(f"Размер events: {events.shape}")
    print(f"Доля положительного класса в train: {train['target'].mean():.4f}")
    print("Даты train:", train["window_start_ts"].min().date(), "..", train["window_start_ts"].max().date())
    print("Даты test: ", test["window_start_ts"].min().date(), "..", test["window_start_ts"].max().date())

    assert train["cookie_id"].is_unique
    assert test["cookie_id"].is_unique
    assert not set(train["cookie_id"]).intersection(test["cookie_id"])
    assert set(sample["cookie_id"]) == set(test["cookie_id"])

    all_meta = pd.concat(
        [
            train[["cookie_id", "window_start_ts", "window_end_ts"]],
            test[["cookie_id", "window_start_ts", "window_end_ts"]],
        ],
        ignore_index=True,
    )
    attached = events.merge(all_meta, on="cookie_id", how="left", validate="many_to_one")
    inside = (
        (attached["event_ts"] >= attached["window_start_ts"])
        & (attached["event_ts"] < attached["window_end_ts"])
    )
    print(f"События внутри допустимых окон: {inside.sum():,} / {len(attached):,}")
    print(f"События вне допустимых окон: {(~inside).sum():,}")
    print()


def normalized_rank(score: np.ndarray) -> np.ndarray:
    """Преобразовать предсказания в нормированные ранги от 0 до 1."""
    score = np.asarray(score, dtype=float)
    return rankdata(score, method="average") / (len(score) + 1.0)


def recency_weights(window_start_ts: pd.Series, reference_date: pd.Timestamp) -> np.ndarray:
    """Экспоненциально уменьшить вес старых наблюдений.

    Half-life = 3 дня означает, что строка трёхдневной давности получает примерно
    половину веса относительно строки, расположенной непосредственно перед
    reference_date. Вес зависит только от времени, target здесь не используется.
    """
    age_days = (reference_date.normalize() - window_start_ts.dt.normalize()).dt.days.to_numpy(dtype=float)
    age_days = np.maximum(age_days, 0.0)
    return np.exp(-np.log(2.0) * age_days / RECENCY_HALF_LIFE_DAYS)


def make_base_model() -> CatBoostClassifier:
    return CatBoostClassifier(**BASE_CATBOOST_PARAMS)


def make_recency_model() -> CatBoostClassifier:
    return CatBoostClassifier(**RECENCY_CATBOOST_PARAMS)


def fit_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    recency: bool,
) -> CatBoostClassifier:
    """Обучить одну CatBoost-модель на фиксированном наборе признаков."""
    cat_indices = [FEATURE_ORDER.index(c) for c in CATEGORICAL_FEATURES]
    model = make_recency_model() if recency else make_base_model()
    model.fit(
        X[FEATURE_ORDER],
        y,
        cat_features=cat_indices,
        sample_weight=sample_weight,
        verbose=False,
    )
    return model


def blend_scores(base_score: np.ndarray, recency_score: np.ndarray) -> np.ndarray:
    """Смешать ранги обычной и recency-модели.

    Основная метрика зависит от порядка объектов, поэтому смешиваются именно
    ранги. Это снижает влияние различий в калибровке вероятностей между моделями.
    """
    return BASE_WEIGHT * normalized_rank(base_score) + RECENCY_WEIGHT * normalized_rank(recency_score)


def metric_row(fold: str, model_name: str, y_true: np.ndarray, score: np.ndarray) -> dict:
    return {
        "Фолд": fold,
        "Модель": model_name,
        "P@R>=0.70": precision_at_recall(y_true, score),
        "PR-AUC": average_precision_score(y_true, score),
        "ROC-AUC": roc_auc_score(y_true, score),
    }


def run_validation(train: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Rolling temporal validation на двух последних временных отрезках."""
    y = train["target"].to_numpy()
    dates = train["window_start_ts"].dt.normalize()
    report: list[dict] = []

    print("=== ROLLING TEMPORAL VALIDATION ===")
    for valid_from, valid_to in ROLLING_FOLDS:
        is_train = dates.lt(valid_from).to_numpy()
        is_valid = dates.ge(valid_from).to_numpy() & dates.lt(valid_to).to_numpy()
        fold_name = f"{valid_from.date()}..{(valid_to - pd.Timedelta(days=1)).date()}"

        print(
            f"Фолд {fold_name}: train={is_train.sum():,}, valid={is_valid.sum():,}, "
            f"positive_valid={y[is_valid].sum():,}"
        )

        base_model = fit_catboost(
            X.loc[is_train],
            y[is_train],
            sample_weight=None,
            recency=False,
        )
        base_score = base_model.predict_proba(X.loc[is_valid, FEATURE_ORDER])[:, 1]
        report.append(metric_row(fold_name, "CatBoost", y[is_valid], base_score))

        weights = recency_weights(train.loc[is_train, "window_start_ts"], valid_from)
        recent_model = fit_catboost(
            X.loc[is_train],
            y[is_train],
            sample_weight=weights,
            recency=True,
        )
        recent_score = recent_model.predict_proba(X.loc[is_valid, FEATURE_ORDER])[:, 1]
        report.append(metric_row(fold_name, "CatBoost с весом свежих данных", y[is_valid], recent_score))

        blend = blend_scores(base_score, recent_score)
        report.append(metric_row(fold_name, "Смешивание рангов 70/30", y[is_valid], blend))

    result = pd.DataFrame(report)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nСреднее по фолдам:")
    print(
        result.groupby("Модель")[["P@R>=0.70", "PR-AUC", "ROC-AUC"]]
        .mean()
        .sort_values("P@R>=0.70", ascending=False)
        .to_string(float_format=lambda x: f"{x:.5f}")
    )
    print()
    return result


def fit_and_predict(
    train: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    base_only: bool = False,
) -> np.ndarray:
    """Обучить финальную модель на всём train и вернуть score для test."""
    y = train["target"].to_numpy()

    print("=== ФИНАЛЬНОЕ ОБУЧЕНИЕ ===")
    base_model = fit_catboost(X_train, y, sample_weight=None, recency=False)
    base_score = base_model.predict_proba(X_test[FEATURE_ORDER])[:, 1]

    if base_only:
        print("Используется только основная CatBoost-модель")
        return base_score

    # Test начинается сразу после последнего дня train, поэтому reference_date
    # ставится на следующий календарный день после максимальной даты train.
    reference_date = train["window_start_ts"].dt.normalize().max() + pd.Timedelta(days=1)
    weights = recency_weights(train["window_start_ts"], reference_date)
    recent_model = fit_catboost(X_train, y, sample_weight=weights, recency=True)
    recent_score = recent_model.predict_proba(X_test[FEATURE_ORDER])[:, 1]

    print(
        f"Смешивание рангов: обычный CatBoost={BASE_WEIGHT:.2f}, "
        f"CatBoost с recency-весами={RECENCY_WEIGHT:.2f}"
    )
    return blend_scores(base_score, recent_score)


def make_submission(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    score: np.ndarray,
    output: Path,
) -> pd.DataFrame:
    """Сохранить ровно одну строку на каждую cookie из test.csv."""
    pred = pd.DataFrame({"cookie_id": test["cookie_id"], "score": score})
    submission = sample[["cookie_id"]].merge(pred, on="cookie_id", how="left", validate="one_to_one")

    assert submission.columns.tolist() == ["cookie_id", "score"]
    assert len(submission) == len(test)
    assert submission["cookie_id"].is_unique
    assert submission["score"].notna().all()
    assert submission["score"].between(0.0, 1.0).all()
    assert set(submission["cookie_id"]) == set(test["cookie_id"])

    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Файл сохранён: {output}")
    return submission


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()

    train, test, events, sample = load_data(root)
    audit_data(train, test, events, sample)

    print("=== ПОСТРОЕНИЕ ПРИЗНАКОВ ===")
    X_train = build_features(events, train)
    X_test = build_features(events, test)
    assert X_train.columns.tolist() == X_test.columns.tolist()
    assert X_train["cookie_id"].tolist() == train["cookie_id"].tolist()
    assert X_test["cookie_id"].tolist() == test["cookie_id"].tolist()
    print(f"Количество признаков на cookie: {len(FEATURE_ORDER)}\n")

    if args.validate:
        report = run_validation(train, X_train)
        report_path = args.output.parent / "validation_report.csv"
        report.to_csv(report_path, index=False)
        print(f"Отчёт по валидации сохранён: {report_path}\n")

    score = fit_and_predict(
        train=train,
        X_train=X_train,
        X_test=X_test,
        base_only=args.base_only,
    )
    make_submission(test, sample, score, args.output)


if __name__ == "__main__":
    main()
