"""Построение признаков для задачи определения автоматизированного трафика.

Главное ограничение задачи: для каждой cookie можно использовать только события,
которые были доступны внутри её окна наблюдения:

    window_start_ts <= event_ts < window_end_ts.

В исходном файле есть события за пределами окна, поэтому временная фильтрация
выполняется до расчёта любых агрегатов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_order import FEATURE_ORDER


EVENT_NAMES = [
    "search_results_view",
    "item_view",
    "photo_swipe",
    "seller_page_view",
    "contact_phone_show",
    "contact_chat_open",
    "contact_message_sent",
    "favorite_add",
    "login",
]

PLATFORM_NORM = ["web", "desktop", "android", "ios"]

UA_FAMILIES = [
    "scrapy",
    "curl",
    "python_requests",
    "python_urllib3",
    "go_http",
    "node_fetch",
    "headless_chrome",
    "avito_android",
    "avito_ios",
    "yandex_browser",
    "firefox",
    "chrome_mobile",
    "chrome_desktop",
    "safari_ios",
    "other",
]

ITEM_CATEGORIES = [
    "avtomobili",
    "mebel",
    "bytovaya_tehnika",
    "hobbi",
    "elektronika",
    "kvartiry_prodazha",
    "uslugi",
    "odezhda",
    "kvartiry_arenda",
    "noutbuki",
    "rabota",
    "detskie_tovary",
    "telefony",
    "zhivotnye",
    "zapchasti",
]

SELLER_TYPES = ["private", "pro"]

CATEGORICAL_FEATURES = [
    "primary_user_agent",
    "primary_ua_family",
    "primary_ua_os",
    "primary_platform",
    "ua_platform_combo",
    "platform_pointer_state",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def normalize_platform(s: pd.Series) -> pd.Series:
    """Нормализовать написание платформ, не смешивая разные типы устройств."""
    x = s.astype("string").str.lower()
    # В этих данных значение ``iphone`` используется как вариант обозначения iOS.
    x = x.replace({"iphone": "ios"})
    return x.fillna("missing")


def ua_family_series(s: pd.Series) -> pd.Series:
    """Выделить укрупнённое семейство User-Agent без внешних парсеров.

    Исходная строка User-Agent тоже сохраняется как категориальный признак, а
    семейство уменьшает зависимость от конкретной версии клиента.
    """
    s = s.fillna("").astype(str)
    out = np.full(len(s), "other", dtype=object)

    # Сначала проверяем явные клиенты автоматизации, затем обычные браузеры.
    checks = [
        (s.str.contains("Scrapy/", case=False, regex=False), "scrapy"),
        (s.str.contains("curl/", case=False, regex=False), "curl"),
        (s.str.contains("python-requests", case=False, regex=False), "python_requests"),
        (s.str.contains("python-urllib3", case=False, regex=False), "python_urllib3"),
        (s.str.contains("Go-http-client", case=False, regex=False), "go_http"),
        (s.str.contains("node-fetch", case=False, regex=False), "node_fetch"),
        (s.str.contains("HeadlessChrome", case=False, regex=False), "headless_chrome"),
        (s.str.startswith("Avito/") & s.str.contains("Android", case=False), "avito_android"),
        (s.str.startswith("Avito/") & s.str.contains("iPhone", case=False), "avito_ios"),
        (s.str.contains("YaBrowser/", case=False, regex=False), "yandex_browser"),
        (s.str.contains("Firefox/", case=False, regex=False), "firefox"),
        (
            s.str.contains("Chrome/", case=False, regex=False)
            & s.str.contains("Mobile", case=False, regex=False),
            "chrome_mobile",
        ),
        (s.str.contains("Chrome/", case=False, regex=False), "chrome_desktop"),
        (
            s.str.contains("iPhone", case=False, regex=False)
            & s.str.contains("Safari/", case=False, regex=False),
            "safari_ios",
        ),
    ]

    for mask, label in checks:
        use = mask.to_numpy() & (out == "other")
        out[use] = label

    return pd.Series(out, index=s.index, dtype="object")


def ua_os_series(s: pd.Series) -> pd.Series:
    """Выделить укрупнённое семейство операционной системы из User-Agent."""
    s = s.fillna("").astype(str)
    out = np.full(len(s), "other", dtype=object)

    # iOS проверяем раньше macOS: User-Agent iPhone может содержать строку ``Mac OS X``.
    checks = [
        (s.str.contains("iPhone|iOS", case=False, regex=True), "ios"),
        (s.str.contains("Android", case=False, regex=False), "android"),
        (s.str.contains("Windows", case=False, regex=False), "windows"),
        (s.str.contains("Macintosh", case=False, regex=False), "macos"),
        (s.str.contains("Linux", case=False, regex=False), "linux"),
    ]

    for mask, label in checks:
        use = mask.to_numpy() & (out == "other")
        out[use] = label

    return pd.Series(out, index=s.index, dtype="object")


def filter_events_in_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Оставить только события, доступные к моменту формирования прогноза.

    Правая граница не включается: ``event_ts < window_end_ts``.
    """
    windows = meta[["cookie_id", "window_start_ts", "window_end_ts"]]
    ev = events.merge(windows, on="cookie_id", how="inner", validate="many_to_one")
    keep = (ev["event_ts"] >= ev["window_start_ts"]) & (ev["event_ts"] < ev["window_end_ts"])
    return ev.loc[keep].copy()


def _entropy_by_group(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Посчитать энтропию Шеннона категориального признака отдельно для каждой cookie."""
    c = (
        df.groupby(["cookie_id", value_col], dropna=False, observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    c["p"] = c["n"] / c.groupby("cookie_id")["n"].transform("sum")
    c["term"] = -(c["p"] * np.log2(c["p"]))
    return c.groupby("cookie_id")["term"].sum()


def _mode_by_cookie(df: pd.DataFrame, col: str) -> pd.Series:
    """Найти наиболее частое значение для каждой cookie с детерминированным разрешением ничьих."""
    c = (
        df.groupby(["cookie_id", col], dropna=False, observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    c["_value_as_text"] = c[col].astype(str)
    c = c.sort_values(
        ["cookie_id", "n", "_value_as_text"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return c.drop_duplicates("cookie_id").set_index("cookie_id")[col]


def _longest_true_run(s: pd.Series, threshold: float) -> int:
    """Длина самой длинной серии интервалов между событиями не больше ``threshold``."""
    a = (s.fillna(threshold + 1) <= threshold).to_numpy()
    best = current = 0
    for value in a:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


# ---------------------------------------------------------------------------
# Основная функция построения признаков
# ---------------------------------------------------------------------------

def build_features(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Построить одну строку признаков для каждой cookie из ``meta``.

    Функция не использует ``target``, поэтому одинаково применяется к train,
    validation и test без утечки целевой переменной.
    """
    ev = filter_events_in_window(events, meta)

    # Нормализованные представления, которые используются в нескольких группах признаков.
    ev["platform_norm"] = normalize_platform(ev["platform"])
    ev["ua_family"] = ua_family_series(ev["user_agent"])
    ev["ua_os"] = ua_os_series(ev["user_agent"])
    ev["ua_automation_marker"] = ev["ua_family"].isin(
        [
            "scrapy",
            "curl",
            "python_requests",
            "python_urllib3",
            "go_http",
            "node_fetch",
            "headless_chrome",
        ]
    ).astype("int8")

    # Исходные события не гарантированно упорядочены. Последовательностные и временные
    # признаки считаются только после явной стабильной сортировки.
    ev = ev.sort_values(["cookie_id", "event_ts", "eid"], kind="mergesort").reset_index(drop=True)
    g = ev.groupby("cookie_id", sort=False, observed=True)

    meta_idx = meta.set_index("cookie_id")
    index = meta_idx.index
    feat: dict[str, pd.Series | np.ndarray | float] = {}

    # ------------------------------------------------------------------
    # 1) Метаданные, доступные к окончанию окна наблюдения
    # ------------------------------------------------------------------
    cookie_age_days = (
        (meta_idx["window_start_ts"] - meta_idx["cookie_created_at"]).dt.total_seconds() / 86400.0
    )
    feat["cookie_age_days"] = cookie_age_days
    feat["log_cookie_age_days"] = np.log1p(cookie_age_days.clip(lower=0))
    feat["cookie_age_lt_1d"] = (cookie_age_days < 1).astype("int8")
    feat["cookie_age_lt_7d"] = (cookie_age_days < 7).astype("int8")
    feat["cookie_age_lt_30d"] = (cookie_age_days < 30).astype("int8")
    feat["window_hours"] = (
        (meta_idx["window_end_ts"] - meta_idx["window_start_ts"]).dt.total_seconds() / 3600.0
    )
    feat["cookie_created_hour"] = meta_idx["cookie_created_at"].dt.hour.astype(float)
    feat["cookie_created_dow"] = meta_idx["cookie_created_at"].dt.dayofweek.astype(float)
    feat["window_start_dow"] = meta_idx["window_start_ts"].dt.dayofweek.astype(float)

    # ------------------------------------------------------------------
    # 2) Общий объём активности и разнообразие
    # ------------------------------------------------------------------
    n_events = g.size()
    feat["n_events"] = n_events

    for col in [
        "event_ts",
        "event_name",
        "platform_norm",
        "user_agent",
        "ua_family",
        "ua_os",
        "item_id",
        "item_category",
        "item_location",
        "seller_type",
        "search_query",
    ]:
        feat[f"{col}_nunique"] = g[col].nunique(dropna=True)

    first_ts = g["event_ts"].min()
    last_ts = g["event_ts"].max()
    feat["first_event_offset_sec"] = (first_ts - meta_idx["window_start_ts"]).dt.total_seconds()
    feat["last_event_offset_sec"] = (last_ts - meta_idx["window_start_ts"]).dt.total_seconds()
    feat["seconds_after_last_event"] = (meta_idx["window_end_ts"] - last_ts).dt.total_seconds()
    activity_span = (last_ts - first_ts).dt.total_seconds()
    feat["activity_span_sec"] = activity_span
    feat["events_per_active_hour"] = n_events / (activity_span / 3600.0 + 1.0 / 60.0)
    feat["events_per_window_hour"] = n_events / pd.Series(feat["window_hours"], index=index).clip(lower=1e-6)

    # В исходных данных есть точные дубликаты строк. Не удаляем их автоматически:
    # количество дублей передаётся модели отдельным признаком, а общий объём событий
    # остаётся таким, каким он был в исходном журнале.
    original_cols = [
        "cookie_id",
        "event_ts",
        "eid",
        "event_name",
        "platform",
        "user_agent",
        "item_id",
        "item_category",
        "item_location",
        "seller_type",
        "search_query",
        "search_page",
        "pointer_x",
        "pointer_y",
    ]
    ev["_exact_dup_after_first"] = ev.duplicated(original_cols, keep="first").astype("int8")
    dup_excess = ev.groupby("cookie_id")["_exact_dup_after_first"].sum()
    feat["exact_duplicate_excess"] = dup_excess
    feat["exact_duplicate_ratio"] = dup_excess / n_events.clip(lower=1)

    # Пропуски могут быть информативны, потому что часть полей зависит от платформы и типа события.
    for col in [
        "item_id",
        "item_category",
        "item_location",
        "seller_type",
        "search_query",
        "search_page",
        "pointer_x",
        "pointer_y",
    ]:
        feat[f"{col}_missing_ratio"] = ev[col].isna().groupby(ev["cookie_id"]).mean()

    # ------------------------------------------------------------------
    # 3) Частоты типов событий и поведенческая воронка
    # ------------------------------------------------------------------
    event_counts = pd.crosstab(ev["cookie_id"], ev["event_name"])
    for name in EVENT_NAMES:
        count = event_counts[name] if name in event_counts.columns else pd.Series(0, index=index)
        feat[f"event__{name}__count"] = count
        feat[f"event__{name}__share"] = count / n_events.clip(lower=1)
        feat[f"event__{name}__present"] = (count > 0).astype("int8")

    feat["event_name_entropy"] = _entropy_by_group(ev, "event_name")
    feat["top_event_share"] = event_counts.max(axis=1) / event_counts.sum(axis=1)

    item = pd.Series(feat["event__item_view__count"], index=index)
    search = pd.Series(feat["event__search_results_view__count"], index=index)
    seller = pd.Series(feat["event__seller_page_view__count"], index=index)
    contacts = (
        pd.Series(feat["event__contact_phone_show__count"], index=index)
        + pd.Series(feat["event__contact_chat_open__count"], index=index)
        + pd.Series(feat["event__contact_message_sent__count"], index=index)
    )
    feat["item_per_search"] = item / (search + 1.0)
    feat["seller_per_item"] = seller / (item + 1.0)
    feat["contact_per_item"] = contacts / (item + 1.0)
    feat["favorite_per_item"] = pd.Series(feat["event__favorite_add__count"], index=index) / (item + 1.0)
    feat["photo_per_item"] = pd.Series(feat["event__photo_swipe__count"], index=index) / (item + 1.0)
    feat["contact_actions_count"] = contacts

    # ------------------------------------------------------------------
    # 4) Платформа и User-Agent
    # ------------------------------------------------------------------
    platform_counts = pd.crosstab(ev["cookie_id"], ev["platform_norm"])
    for value in PLATFORM_NORM:
        count = platform_counts[value] if value in platform_counts.columns else pd.Series(0, index=index)
        feat[f"platform__{value}__count"] = count
        feat[f"platform__{value}__share"] = count / n_events.clip(lower=1)
    feat["platform_entropy"] = _entropy_by_group(ev, "platform_norm")

    family_counts = pd.crosstab(ev["cookie_id"], ev["ua_family"])
    for value in UA_FAMILIES:
        count = family_counts[value] if value in family_counts.columns else pd.Series(0, index=index)
        feat[f"ua_family__{value}__count"] = count
        feat[f"ua_family__{value}__share"] = count / n_events.clip(lower=1)

    feat["ua_automation_marker_share"] = g["ua_automation_marker"].mean()
    feat["ua_automation_marker_any"] = g["ua_automation_marker"].max()
    feat["user_agent_length_mean"] = g["user_agent"].apply(lambda s: s.astype(str).str.len().mean())

    primary_user_agent = _mode_by_cookie(ev, "user_agent")
    primary_ua_family = _mode_by_cookie(ev, "ua_family")
    primary_ua_os = _mode_by_cookie(ev, "ua_os")
    primary_platform = _mode_by_cookie(ev, "platform_norm")
    feat["primary_user_agent"] = primary_user_agent
    feat["primary_ua_family"] = primary_ua_family
    feat["primary_ua_os"] = primary_ua_os
    feat["primary_platform"] = primary_platform

    # ------------------------------------------------------------------
    # 5) Разнообразие и концентрация контента и поиска
    # ------------------------------------------------------------------
    for col, prefix in [
        ("item_category", "category"),
        ("item_location", "location"),
        ("search_query", "query"),
    ]:
        sub = ev.loc[ev[col].notna(), ["cookie_id", col]]
        if not sub.empty:
            feat[f"{col}_entropy"] = _entropy_by_group(sub, col)
            counts = sub.groupby(["cookie_id", col]).size().rename("n").reset_index()
            total = counts.groupby("cookie_id")["n"].sum()
            feat[f"top_{prefix}_share"] = counts.groupby("cookie_id")["n"].max() / total
            feat[f"{prefix}_count_nonmissing"] = total

    feat["unique_item_ratio"] = g["item_id"].nunique() / (item + 1.0)
    feat["unique_query_ratio"] = g["search_query"].nunique() / (search + 1.0)
    feat["unique_category_ratio"] = g["item_category"].nunique() / (item + 1.0)
    feat["unique_location_ratio"] = g["item_location"].nunique() / (item + 1.0)
    feat["locations_per_category"] = g["item_location"].nunique() / (g["item_category"].nunique() + 1.0)
    feat["items_per_category"] = g["item_id"].nunique() / (g["item_category"].nunique() + 1.0)
    feat["items_per_location"] = g["item_id"].nunique() / (g["item_location"].nunique() + 1.0)

    cat_loc = ev.loc[ev["item_category"].notna() & ev["item_location"].notna(), ["cookie_id", "item_category", "item_location"]].copy()
    if not cat_loc.empty:
        cat_loc["category_location"] = cat_loc["item_category"].astype(str) + "|" + cat_loc["item_location"].astype(str)
        feat["category_location_nunique"] = cat_loc.groupby("cookie_id")["category_location"].nunique()

    category_counts = pd.crosstab(ev["cookie_id"], ev["item_category"])
    for value in ITEM_CATEGORIES:
        feat[f"category__{value}__count"] = (
            category_counts[value] if value in category_counts.columns else pd.Series(0, index=index)
        )

    seller_counts = pd.crosstab(ev["cookie_id"], ev["seller_type"])
    for value in SELLER_TYPES:
        count = seller_counts[value] if value in seller_counts.columns else pd.Series(0, index=index)
        feat[f"seller__{value}__count"] = count
        feat[f"seller__{value}__share"] = count / n_events.clip(lower=1)

    # Глубина поиска и переходы по страницам одного запроса.
    search_rows = ev.loc[ev["search_query"].notna() & ev["search_page"].notna()].copy()
    if not search_rows.empty:
        spg = search_rows.groupby("cookie_id")["search_page"]
        feat["search_page_mean"] = spg.mean()
        feat["search_page_std"] = spg.std()
        feat["search_page_max"] = spg.max()
        feat["search_page_nunique"] = spg.nunique()
        feat["search_page_gt1_share"] = (
            search_rows.assign(_v=(search_rows["search_page"] > 1).astype(float))
            .groupby("cookie_id")["_v"]
            .mean()
        )

        query_stats = (
            search_rows.groupby(["cookie_id", "search_query"])["search_page"]
            .agg(["count", "nunique", "max"])
            .reset_index()
        )
        feat["queries_multi_page_share"] = (
            query_stats.assign(_v=(query_stats["nunique"] > 1).astype(float))
            .groupby("cookie_id")["_v"]
            .mean()
        )
        feat["pages_per_query_mean"] = query_stats.groupby("cookie_id")["nunique"].mean()
        feat["max_page_per_query_mean"] = query_stats.groupby("cookie_id")["max"].mean()
        feat["max_page_per_query_max"] = query_stats.groupby("cookie_id")["max"].max()

        search_rows = search_rows.sort_values(["cookie_id", "event_ts", "eid"], kind="mergesort")
        search_rows["prev_query"] = search_rows.groupby("cookie_id")["search_query"].shift()
        search_rows["prev_page"] = search_rows.groupby("cookie_id")["search_page"].shift()
        search_rows["page_delta"] = search_rows["search_page"] - search_rows["prev_page"]
        same_query = search_rows["prev_query"].eq(search_rows["search_query"])
        sq = search_rows.loc[same_query & search_rows["page_delta"].notna()].copy()
        if not sq.empty:
            feat["same_query_page_plus1_share"] = (
                sq.assign(_v=(sq["page_delta"] == 1).astype(float)).groupby("cookie_id")["_v"].mean()
            )
            feat["same_query_page_nondecrease_share"] = (
                sq.assign(_v=(sq["page_delta"] >= 0).astype(float)).groupby("cookie_id")["_v"].mean()
            )
            feat["same_query_page_delta_abs_mean"] = (
                sq.assign(_v=sq["page_delta"].abs()).groupby("cookie_id")["_v"].mean()
            )

    # ------------------------------------------------------------------
    # 6) Интервалы между событиями, всплески активности и сессии
    # ------------------------------------------------------------------
    ev["gap_sec"] = g["event_ts"].diff().dt.total_seconds()
    gaps = ev.loc[ev["gap_sec"].notna(), ["cookie_id", "gap_sec"]]
    if not gaps.empty:
        gg = gaps.groupby("cookie_id")["gap_sec"]
        feat["gap_mean"] = gg.mean()
        feat["gap_std"] = gg.std()
        feat["gap_min"] = gg.min()
        feat["gap_max"] = gg.max()
        feat["gap_median"] = gg.median()

        for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
            feat[f"gap_q{int(q * 100):02d}"] = gg.quantile(q)

        for threshold in [0, 1, 2, 5, 10, 30, 60, 300, 900]:
            feat[f"gap_le_{threshold}s_share"] = (
                gaps.assign(_v=(gaps["gap_sec"] <= threshold).astype(float))
                .groupby("cookie_id")["_v"]
                .mean()
            )

        for lo, hi, name in [
            (2, 10, "2_10"),
            (3, 8, "3_8"),
            (4, 12, "4_12"),
            (5, 15, "5_15"),
            (10, 60, "10_60"),
            (60, 300, "60_300"),
            (300, 1800, "300_1800"),
        ]:
            feat[f"gap_range_{name}_share"] = (
                gaps.assign(_v=((gaps["gap_sec"] >= lo) & (gaps["gap_sec"] <= hi)).astype(float))
                .groupby("cookie_id")["_v"]
                .mean()
            )

        for threshold in [60, 300, 900, 1800, 3600]:
            feat[f"gap_gt_{threshold}s_share"] = (
                gaps.assign(_v=(gaps["gap_sec"] > threshold).astype(float))
                .groupby("cookie_id")["_v"]
                .mean()
            )

        gap_nunique = gg.nunique()
        gap_count = gg.size()
        feat["gap_repeat_ratio"] = 1.0 - gap_nunique / gap_count.clip(lower=1)
        feat["gap_cv"] = pd.Series(feat["gap_std"], index=index) / (pd.Series(feat["gap_mean"], index=index).abs() + 1e-6)

        gap_counts = gaps.groupby(["cookie_id", "gap_sec"]).size().rename("n").reset_index()
        feat["modal_gap_share"] = gap_counts.groupby("cookie_id")["n"].max() / gap_count

        short = gaps.loc[gaps["gap_sec"] <= 300]
        if not short.empty:
            sg = short.groupby("cookie_id")["gap_sec"]
            feat["short_gap_mean"] = sg.mean()
            feat["short_gap_std"] = sg.std()
            feat["short_gap_median"] = sg.median()
            feat["short_gap_q90"] = sg.quantile(0.90)
            feat["short_gap_cv"] = pd.Series(feat["short_gap_std"], index=index) / (
                pd.Series(feat["short_gap_mean"], index=index) + 1e-6
            )
            short_counts = short.groupby(["cookie_id", "gap_sec"]).size().rename("n").reset_index()
            short_counts["p"] = short_counts["n"] / short_counts.groupby("cookie_id")["n"].transform("sum")
            short_counts["term"] = -(short_counts["p"] * np.log2(short_counts["p"]))
            feat["short_gap_entropy"] = short_counts.groupby("cookie_id")["term"].sum()

        # Новая сессия начинается после достаточно длинного периода бездействия.
        for threshold in [300, 900, 1800]:
            new_session = (ev["gap_sec"].isna() | (ev["gap_sec"] > threshold)).astype("int8")
            feat[f"sessions_gap{threshold}s"] = new_session.groupby(ev["cookie_id"]).sum()

        for threshold in [60, 300]:
            feat[f"longest_run_gap_le_{threshold}s"] = ev.groupby("cookie_id")["gap_sec"].apply(
                lambda s, t=threshold: _longest_true_run(s, t)
            )

    # События в одну и ту же секунду учитываются отдельно от точных дублей строк.
    same_second = ev.groupby(["cookie_id", "event_ts"]).size().rename("n").reset_index()
    feat["max_events_same_second"] = same_second.groupby("cookie_id")["n"].max()
    same_timestamp_excess = same_second.assign(_excess=same_second["n"] - 1).groupby("cookie_id")["_excess"].sum()
    feat["same_timestamp_excess"] = same_timestamp_excess
    feat["same_timestamp_ratio"] = same_timestamp_excess / n_events.clip(lower=1)

    # Календарные признаки и плотность событий внутри суточного окна.
    ev["minute_bucket"] = ev["event_ts"].dt.floor("min")
    ev["bucket_5min"] = ev["event_ts"].dt.floor("5min")
    ev["hour"] = ev["event_ts"].dt.hour
    ev["second"] = ev["event_ts"].dt.second
    feat["active_minutes"] = g["minute_bucket"].nunique()
    feat["active_5min_buckets"] = g["bucket_5min"].nunique()
    feat["active_hours"] = g["hour"].nunique()
    feat["active_seconds_of_minute"] = g["second"].nunique()
    feat["events_per_active_minute"] = n_events / pd.Series(feat["active_minutes"], index=index).clip(lower=1)
    feat["events_per_active_5min"] = n_events / pd.Series(feat["active_5min_buckets"], index=index).clip(lower=1)
    feat["max_events_same_minute"] = ev.groupby(["cookie_id", "minute_bucket"]).size().groupby("cookie_id").max()
    feat["max_events_same_5min"] = ev.groupby(["cookie_id", "bucket_5min"]).size().groupby("cookie_id").max()
    feat["hour_entropy"] = _entropy_by_group(ev, "hour")
    feat["second_of_minute_entropy"] = _entropy_by_group(ev, "second")

    # ------------------------------------------------------------------
    # 7) Последовательности и переходы между типами событий
    # ------------------------------------------------------------------
    ev["prev_event"] = g["event_name"].shift(1)
    transitions = ev.loc[ev["prev_event"].notna(), ["cookie_id", "prev_event", "event_name"]].copy()
    if not transitions.empty:
        transitions["transition"] = transitions["prev_event"].astype(str) + "__to__" + transitions["event_name"].astype(str)
        transition_counts = pd.crosstab(transitions["cookie_id"], transitions["transition"])
        for a in EVENT_NAMES:
            for b in EVENT_NAMES:
                name = f"{a}__to__{b}"
                feat[f"trans__{name}__count"] = (
                    transition_counts[name]
                    if name in transition_counts.columns
                    else pd.Series(0, index=index)
                )
        feat["same_event_transition_share"] = (
            transitions.assign(_v=(transitions["prev_event"] == transitions["event_name"]).astype(float))
            .groupby("cookie_id")["_v"]
            .mean()
        )
        feat["transition_nunique"] = transitions.groupby("cookie_id")["transition"].nunique()
        feat["transition_entropy"] = _entropy_by_group(transitions, "transition")

    # ------------------------------------------------------------------
    # 8) Геометрия указателя для web/desktop-трафика
    # ------------------------------------------------------------------
    pointer = ev.loc[ev["pointer_x"].notna() & ev["pointer_y"].notna()].copy()
    pointer_count = pointer.groupby("cookie_id").size()
    feat["pointer_event_count"] = pointer_count
    feat["pointer_present_share"] = pointer_count / n_events.clip(lower=1)

    if not pointer.empty:
        pg = pointer.groupby("cookie_id")
        for col in ["pointer_x", "pointer_y"]:
            feat[f"{col}_mean"] = pg[col].mean()
            feat[f"{col}_std"] = pg[col].std()
            feat[f"{col}_nunique"] = pg[col].nunique()
            feat[f"{col}_min"] = pg[col].min()
            feat[f"{col}_max"] = pg[col].max()
            feat[f"{col}_range"] = pg[col].max() - pg[col].min()

        pointer["xy"] = pointer["pointer_x"].astype(str) + "|" + pointer["pointer_y"].astype(str)
        pair_nunique = pointer.groupby("cookie_id")["xy"].nunique()
        feat["pointer_pair_nunique"] = pair_nunique
        feat["pointer_pair_unique_ratio"] = pair_nunique / pointer_count.clip(lower=1)

        pointer = pointer.sort_values(["cookie_id", "event_ts"], kind="mergesort")
        pointer["dt"] = pointer.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()
        pointer["dx"] = pointer.groupby("cookie_id")["pointer_x"].diff()
        pointer["dy"] = pointer.groupby("cookie_id")["pointer_y"].diff()
        pointer["dist"] = np.sqrt(pointer["dx"] ** 2 + pointer["dy"] ** 2)
        dg = pointer.groupby("cookie_id")["dist"]
        feat["pointer_dist_mean"] = dg.mean()
        feat["pointer_dist_std"] = dg.std()
        feat["pointer_dist_median"] = dg.median()
        feat["pointer_dist_q25"] = dg.quantile(0.25)
        feat["pointer_dist_q75"] = dg.quantile(0.75)
        feat["pointer_dist_q90"] = dg.quantile(0.90)
        feat["pointer_dist_max"] = dg.max()
        feat["pointer_total_distance"] = dg.sum()
        feat["pointer_abs_dx_mean"] = pointer.assign(_v=pointer["dx"].abs()).groupby("cookie_id")["_v"].mean()
        feat["pointer_abs_dy_mean"] = pointer.assign(_v=pointer["dy"].abs()).groupby("cookie_id")["_v"].mean()
        feat["pointer_zero_move_share"] = (
            pointer.assign(_v=(pointer["dist"] == 0).astype(float)).groupby("cookie_id")["_v"].mean()
        )

        speed = pointer.loc[(pointer["dt"] > 0) & pointer["dist"].notna()].copy()
        if not speed.empty:
            speed["speed"] = speed["dist"] / speed["dt"]
            feat["pointer_speed_mean"] = speed.groupby("cookie_id")["speed"].mean()
            feat["pointer_speed_median"] = speed.groupby("cookie_id")["speed"].median()

    # Явные взаимодействия помогают неглубоким деревьям выделить паттерн
    # «браузер без координат указателя».
    pointer_any = pointer_count.reindex(index).fillna(0).gt(0)
    feat["is_web_no_pointer"] = ((primary_platform.reindex(index) == "web") & (~pointer_any)).astype("int8")
    feat["is_desktop_no_pointer"] = ((primary_platform.reindex(index) == "desktop") & (~pointer_any)).astype("int8")
    browser_families = {
        "yandex_browser",
        "firefox",
        "chrome_mobile",
        "chrome_desktop",
        "safari_ios",
        "headless_chrome",
    }
    feat["is_browser_no_pointer"] = (
        primary_ua_family.reindex(index).isin(browser_families) & (~pointer_any)
    ).astype("int8")
    feat["ua_platform_combo"] = (
        primary_ua_family.reindex(index).astype(str) + "|" + primary_platform.reindex(index).astype(str)
    )
    feat["platform_pointer_state"] = (
        primary_platform.reindex(index).astype(str)
        + "|"
        + pointer_any.map({True: "ptr", False: "noptr"}).astype(str)
    )

    # ------------------------------------------------------------------
    # Собираем итоговую таблицу одним действием, чтобы не фрагментировать DataFrame.
    # ------------------------------------------------------------------
    X = pd.DataFrame(feat, index=index)
    X.index.name = "cookie_id"

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna("missing").astype(str)

    numeric = [c for c in X.columns if c not in CATEGORICAL_FEATURES]
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Фиксируем порядок признаков. Это делает обучение с заданным random_seed
    # воспроизводимым и защищает от случайного изменения порядка колонок.
    missing = [c for c in FEATURE_ORDER if c not in X.columns]
    if missing:
        raise RuntimeError(f"Не удалось построить ожидаемые признаки: {missing[:10]}")
    X = X[FEATURE_ORDER]
    return X.reset_index()
