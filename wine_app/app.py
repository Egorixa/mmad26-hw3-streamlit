import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
CONFIG_PATH = APP_DIR / "config.json"
TEST_PATH = APP_DIR / "test.csv"

FEATURE_INPUTS = {
    "fixed_acidity": ("Фиксированная кислотность, г/л", 0.1, "%.1f"),
    "volatile_acidity": ("Летучая кислотность, г/л", 0.01, "%.2f"),
    "citric_acid": ("Лимонная кислота, г/л", 0.01, "%.2f"),
    "residual_sugar": ("Остаточный сахар, г/л", 0.1, "%.1f"),
    "chlorides": ("Хлориды, г/л", 0.001, "%.3f"),
    "free_sulfur_dioxide": ("Свободный SO₂, мг/л", 1.0, "%.0f"),
    "total_sulfur_dioxide": ("Общий SO₂, мг/л", 1.0, "%.0f"),
    "density": ("Плотность, г/см³", 0.0001, "%.5f"),
    "pH": ("pH", 0.01, "%.2f"),
    "sulphates": ("Сульфаты, г/л", 0.01, "%.2f"),
    "alcohol": ("Алкоголь, % об.", 0.1, "%.1f"),
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_config():
    return json.loads(CONFIG_PATH.read_text())


@st.cache_data
def load_test():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def test_proba(_model, X):  # _model: модель не хешируем, она и так одна на сервер
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def build_curves(y, proba):
    fpr, tpr, _ = metrics.roc_curve(y, proba)
    precision, recall, _ = metrics.precision_recall_curve(y, proba)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "roc_auc": metrics.roc_auc_score(y, proba),
        "pr_auc": metrics.average_precision_score(y, proba),
    }


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)


def wine_tab(model, X_test, features, threshold):
    st.subheader("Оценка одного вина")
    st.caption("Диапазоны и значения по умолчанию взяты из тестовой выборки (минимум, максимум, медиана).")

    is_red = st.radio("Тип вина", ["белое", "красное"], horizontal=True) == "красное"

    values = {}
    columns = st.columns(3)
    for i, (feature, (label, step, fmt)) in enumerate(FEATURE_INPUTS.items()):
        values[feature] = columns[i % 3].number_input(
            label,
            min_value=float(X_test[feature].min()),
            max_value=float(X_test[feature].max()),
            value=float(X_test[feature].median()),
            step=step,
            format=fmt,
        )
    values["is_red"] = int(is_red)

    wine = pd.DataFrame([values])[features]
    proba = model.predict_proba(wine)[0, 1]

    st.metric("Вероятность, что вино хорошее", f"{proba:.1%}")
    st.progress(float(proba))
    if proba >= threshold:
        st.success(f"Берём в подборку: вероятность не ниже порога {threshold:.3f}.")
    else:
        st.warning(f"Не берём в подборку: вероятность ниже порога {threshold:.3f}.")


def confusion_figure(cm):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=["не хорошее", "хорошее"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set(xlabel="Решение модели", ylabel="На самом деле")
    return fig


def curve_figure(x, y, point, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, color="#2a78b5", linewidth=2)
    if point is not None:
        ax.scatter(*point, color="#c2452d", s=70, zorder=3, edgecolor="white", label="текущий порог")
        ax.legend(loc="lower right" if title.startswith("ROC") else "upper right")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    return fig


def metrics_tab(y_test, proba, threshold):
    st.subheader("Модельные метрики на тестовой выборке")

    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    first, second, third, fourth = st.columns(4)
    first.metric("Precision", f"{precision:.3f}")
    second.metric("Recall", f"{recall:.3f}")
    third.metric("F1", f"{f1:.3f}")
    fourth.metric("Вин в подборке", f"{pred.sum()} из {len(pred)}")

    left, right = st.columns([1, 1.2])
    with left:
        show_figure(confusion_figure(np.array([[tn, fp], [fn, tp]])))
    with right:
        st.markdown(
            f"""
**Как читать порог**

- **Увеличить порог.** В подборку попадут только вина, в которых модель уверена: подборка станет
  короче, посредственных вин в ней будет меньше (растёт precision), но больше хороших останется
  за бортом (падает recall).
- **Уменьшить порог.** Подборка станет шире и хороших вин в ней окажется больше, но вместе с ними
  пройдёт и больше посредственных.

Сейчас из {tp + fn} хороших вин в подборку попало {tp}, а ещё {fp} вин в ней посредственные.
"""
        )

    curves = build_curves(y_test, proba)
    left, right = st.columns(2)
    with left:
        fpr = fp / (fp + tn)
        show_figure(curve_figure(
            curves["fpr"], curves["tpr"], (fpr, recall), "FPR", "TPR (recall)",
            f"ROC-кривая, ROC-AUC = {curves['roc_auc']:.3f}",
        ))
    with right:
        point = (recall, precision) if pred.sum() > 0 else None
        show_figure(curve_figure(
            curves["recall"], curves["precision"], point, "Recall", "Precision",
            f"PR-кривая, PR-AUC = {curves['pr_auc']:.3f}",
        ))
        if point is None:
            st.caption("При таком пороге подборка пуста, поэтому precision не определён.")


def read_batch(uploaded, features):
    """Читает CSV и возвращает таблицу и числовые признаки; при ошибке показывает st.error."""
    try:
        batch = pd.read_csv(uploaded)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        st.error("Не удалось прочитать файл: нужен CSV с заголовком в первой строке.")
        return None, None

    missing = [feature for feature in features if feature not in batch.columns]
    if missing:
        st.error(f"В файле не хватает столбцов: {', '.join(missing)}")
        return None, None
    if batch.empty:
        st.error("В файле нет ни одной строки с вином.")
        return None, None

    X = batch[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad_rows = X.isna().any(axis=1)
    if bad_rows.any():
        rows = ", ".join(str(i + 1) for i in batch.index[bad_rows][:10])
        st.error(f"Строк с пустыми или нечисловыми значениями: {bad_rows.sum()}. Номера первых из них: {rows}.")
        return None, None
    return batch, X


def batch_tab(model, features, threshold):
    st.subheader("Пакетное предсказание")
    st.caption(f"Решение принимается при пороге из боковой панели: {threshold:.3f}.")

    uploaded = st.file_uploader("CSV с признаками вин (без столбца target)", type="csv")
    if uploaded is None:
        st.info(f"Нужные столбцы: {', '.join(features)}")
        return

    batch, X = read_batch(uploaded, features)
    if batch is None:
        return

    result = batch.copy()
    result["proba"] = model.predict_proba(X)[:, 1]
    result["decision"] = np.where(result["proba"] >= threshold, "в подборку", "мимо")

    st.metric("Вин в подборке", f"{(result['proba'] >= threshold).sum()} из {len(result)}")
    st.dataframe(result, hide_index=True)
    st.download_button(
        "Скачать таблицу",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
        on_click="ignore",
    )


def coef_tab(model):
    st.subheader("Что выучила модель")
    st.caption("Коэффициенты посчитаны после масштабирования, поэтому их можно сравнивать между собой.")

    coefs = pd.Series(model[-1].coef_[0], index=model.feature_names_in_).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = np.where(coefs >= 0, "#2a78b5", "#c2452d")
    ax.barh(coefs.index, coefs.to_numpy(), color=colors, height=0.6)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_xlabel("Коэффициент логистической регрессии")
    ax.grid(axis="x", alpha=0.3)
    show_figure(fig)

    positive = coefs[coefs > 0].sort_values(ascending=False).head(3)
    negative = coefs[coefs < 0].head(3)
    st.markdown(
        f"**Сильнее всего за хорошее вино:** {', '.join(positive.index)}.\n\n"
        f"**Сильнее всего против:** {', '.join(negative.index)}."
    )
    st.caption(
        "density, residual_sugar и alcohol сильно связаны между собой, поэтому их крупные "
        "коэффициенты частично компенсируют друг друга: смотреть на них по одному нужно осторожно."
    )


st.set_page_config(page_title="Подборка вин", layout="wide")

missing_files = [path.name for path in (MODEL_PATH, CONFIG_PATH, TEST_PATH) if not path.exists()]
if missing_files:
    st.error(f"Нет файлов: {', '.join(missing_files)}. Сначала выполните ноутбук с домашним заданием.")
    st.stop()

model = load_model()
config = load_config()
test = load_test()
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test["target"]
proba = test_proba(model, X_test)

st.title("Подборка хороших вин")
st.caption("Логистическая регрессия по лабораторному анализу вина. Модель внутри приложения не обучается.")

threshold = st.sidebar.slider(
    "Порог «хорошее вино»", 0.0, 1.0, float(config["threshold"]), step=0.001, format="%.3f"
)
st.sidebar.caption(f"Порог по умолчанию из config.json: {config['threshold']:.3f}.")

wine, model_metrics, batch, coefficients = st.tabs(
    ["Оценка вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)
with wine:
    wine_tab(model, X_test, features, threshold)
with model_metrics:
    metrics_tab(y_test, proba, threshold)
with batch:
    batch_tab(model, features, threshold)
with coefficients:
    coef_tab(model)
