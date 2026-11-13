"""КТвД — Wine Recognition | интерактивный Spectral Nightfall.

Запуск: python -m marimo run "KTvD_Wine_marimo.py"
Файл spectral_nightfall.css должен лежать рядом с этим файлом.
"""
# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
import marimo

app = marimo.App(width="full", css_file="spectral_nightfall.css")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn.datasets import load_wine

    path = Path(__file__).resolve().parent / "data" / "wine_two_variables.csv"
    if path.exists():
        table = pd.read_csv(path)
    else:
        source = load_wine(as_frame=True).frame
        table = source[["alcohol", "color_intensity"]].copy()
    table = table.rename(columns={"Alcohol": "alcohol", "Color intensity": "color_intensity"})
    assert {"alcohol", "color_intensity"}.issubset(table.columns), "Не найдены две переменные Wine"
    return mo, np, pd, plt, table


@app.cell
def _(mo):
    mo.Html("""
    <div class="sn-hero">
      <div class="sn-eyebrow">КТвД / ИНТЕРАКТИВНОЕ ИССЛЕДОВАНИЕ · WINE RECOGNITION</div>
      <div class="sn-hero-title">Статистика, которую<br><span>можно исследовать.</span></div>
      <p class="sn-hero-desc">Две случайные величины — содержание алкоголя и интенсивность окраски.
      Меняйте подвыборку и режим связности, чтобы увидеть, как меняется корреляция и результат перестановочного теста.</p>
      <p class="sn-authors">Выполнили: Ким Дарья (503310) · Козырева Екатерина (504613)</p>
      <div class="sn-hero-tags"><span>178 образцов</span><span>2 признака</span><span>Перестановочный тест</span><span>Воспроизводимый seed</span></div>
    </div>
    """)
    return


@app.cell
def _(mo, table):
    count = mo.ui.slider(start=15, stop=len(table), step=1, value=len(table), show_value=True, label="Размер подвыборки")
    mode = mo.ui.dropdown(options={"Сохранить пары": "observed", "Перемешать Y": "shuffled"}, value="Сохранить пары", label="Режим данных")
    seed = mo.ui.number(start=1, stop=999999, step=1, value=2026, label="Зерно генератора (seed)")
    mo.vstack([
        mo.Html('<div class="sn-section-kicker">01 / ПАРАМЕТРЫ ЭКСПЕРИМЕНТА</div>'),
        mo.Html('<div class="sn-section-title">Управляйте наблюдениями</div>'),
        mo.hstack([count, mode, seed], gap=2, justify="space-between", align="center"),
        mo.md('Изменение параметров автоматически пересчитывает графики и статистику. Перемешивание **Y** разрушает исходное соответствие пар X–Y.'),
    ], gap=1)
    return count, mode, seed


@app.cell
def _(np, plt, mo, table, count, mode, seed):
    rng = np.random.default_rng(int(seed.value))
    idx = rng.choice(len(table), size=int(count.value), replace=False)
    x = table["alcohol"].to_numpy(dtype=float)[idx]
    y = table["color_intensity"].to_numpy(dtype=float)[idx]
    if mode.value == "shuffled":
        y = rng.permutation(y)

    r_obs = float(np.corrcoef(x, y)[0, 1])
    n_permutations = 1000
    r_null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_permutations)])
    p_perm = (1 + int(np.count_nonzero(np.abs(r_null) >= abs(r_obs)))) / (n_permutations + 1)

    fgc, textc, muted, lilac, cyan = "#242329", "#f2f1f4", "#a8a4b7", "#9587e7", "#d8f8f8"
    with plt.rc_context({
        "figure.facecolor": fgc, "savefig.facecolor": fgc,
        "axes.facecolor": "#302e36", "axes.edgecolor": "#656075",
        "axes.labelcolor": textc, "text.color": textc,
        "xtick.color": muted, "ytick.color": muted,
        "font.size": 11, "axes.titleweight": "bold", "axes.titlesize": 12,
        "grid.color": "#504c5a", "grid.alpha": 0.24,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.5), constrained_layout=True)
        axes[0].scatter(x, y, color=cyan, edgecolors="#e7ffff", linewidths=0.35, alpha=.75, s=35)
        axes[0].set(xlabel="Содержание алкоголя, % об. (шкала датасета)",
                    ylabel="Интенсивность окраски, усл. ед.",
                    title=f"Диаграмма рассеяния · n={len(x)}")
        axes[0].grid(alpha=.15)
        axes[1].hist(r_null, bins=32, color=lilac, edgecolor=fgc, alpha=.92)
        axes[1].axvline(r_obs, color=cyan, linewidth=2, label=f"наблюдаемое r = {r_obs:.3f}")
        axes[1].set(xlabel="Корреляция r при перестановках Y", ylabel="Количество перестановок",
                    title="Нулевое распределение корреляции")
        axes[1].legend(frameon=False, labelcolor=textc, fontsize=9)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)

    label_mode = "Исходные пары" if mode.value == "observed" else "Y перемешана"
    stats = f"""
    <div class="sn-section-kicker">02 / РЕЗУЛЬТАТЫ В РЕАЛЬНОМ ВРЕМЕНИ</div>
    <div class="sn-section-title">Что показывают данные</div>
    <div class="sn-metrics">
      <div class="sn-metric"><span>ЧИСЛО НАБЛЮДЕНИЙ</span><strong>{len(x)}</strong><small>из {len(table)} образцов</small></div>
      <div class="sn-metric"><span>КОРРЕЛЯЦИЯ ПИРСОНА</span><strong>{r_obs:+.4f}</strong><small>коэффициент r</small></div>
      <div class="sn-metric"><span>ПЕРЕСТАНОВОЧНОЕ p</span><strong>{p_perm:.4f}</strong><small>{n_permutations} перестановок</small></div>
      <div class="sn-metric"><span>РЕЖИМ</span><strong class="sn-metric-mode">{label_mode}</strong><small>seed = {int(seed.value)}</small></div>
    </div>
    """
    mo.vstack([
        mo.Html(stats),
        fig,
        mo.Html('<div class="sn-graph-caption">Левая панель: наблюдаемые пары значений. Правая: эмпирическое нулевое распределение при случайных перестановках Y; голубая линия — наблюдаемая корреляция.</div>'),
    ], gap=1)
    return


@app.cell
def _(mo):
    mo.Html("""
    <div class="sn-footer">
      <div class="sn-section-kicker">03 / КАК ЧИТАТЬ РЕЗУЛЬТАТ</div>
      <div class="sn-footer-title">Корреляция — не причинность.</div>
      <p>Это наблюдательные данные из учебного датасета Wine Recognition.
      Небольшое перестановочное p-значение указывает на статистическую несовместимость наблюдаемой связи с гипотезой независимости в рамках теста,
      но не доказывает причинно-следственную связь. Поменяйте режим на «Перемешать Y» и сравните результаты.</p>
      <div class="sn-endnote">SPECTRAL NIGHTFALL · КТвД · ДОПОЛНИТЕЛЬНОЕ ИССЛЕДОВАНИЕ</div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
