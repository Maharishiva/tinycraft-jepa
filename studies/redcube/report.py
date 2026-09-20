"""The red-cube report: figures and a standalone HTML page from runs/redcube/<config>/ results.

    python -m studies.redcube.report            # re-runnable; configurations without results are skipped
"""
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from studies.redcube import env as R

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "redcube"
OUT = ROOT / "blog" / "redcube"
ASSETS = OUT / "assets"
INK, GREY, GREEN, PURPLE, ORANGE = "#1c1c1c", "#9a9a9a", "#2a9d8f", "#7b4ea3", "#e07b39"
MODEL_ORDER = ("persistence", "tile_factored", "lewm6", "flow")
MODEL_COLOURS = {"persistence": GREY, "tile_factored": PURPLE, "lewm6": ORANGE, "flow": GREEN}
MODEL_NAMES = {"persistence": "копия последнего кадра", "tile_factored": "JEPA", "lewm6": "LeWM", "flow": "flow matching"}
ARMS_MODELS = ("lewm6",)
BG = {"off": "чёрный фон", "color_frame": "цветной шум на кадр", "color_episode": "цветной шум на эпизод",
      "gray_frame": "серый шум на кадр", "gray_episode": "серый шум на эпизод"}
WIND = (("wind", "с ветром", 35), ("still", "без ветра", 14))
RU = {f"{bg}_{w}": f"{BG[bg]}, {label}" for bg in BG for w, label, _ in WIND}
STEPS = (0, 3, 7, 15)
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                            "axes.spines.right": False, "figure.facecolor": "#faf9f6", "axes.facecolor": "#faf9f6",
                            "savefig.facecolor": "#faf9f6"})


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def load(cfg):
    """The page benchmark (JEPA + flow matching), the oracles, and the all-arms benchmark for the LeWM line."""
    d = RUNS / cfg
    r = {"bench": read(d / "bench.json"), "oracle": read(d / "oracle.json"), "arms": read(d / "allarms" / "bench.json")}
    return r if r["bench"] is not None else None


def bench_of(r, model):
    return r["arms"] if model in ARMS_MODELS else r["bench"]


def env_preview():
    import jax
    tiles = []
    for name in R.ORDER:
        obs, _ = R.make(R.CONFIGS[name])[0](jax.random.PRNGKey(3))
        tiles.append(np.repeat(np.repeat(np.asarray(obs), 3, 0), 3, 1))
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.2))
    for ax, name, tile in zip(axes.ravel(), R.ORDER, tiles):
        ax.imshow(tile)
        ax.set_title(RU[name], fontsize=10, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ASSETS / "env_configs.png", dpi=130)
    plt.close(fig)


def models_of(results):
    present = set()
    for r in results.values():
        present |= set(r["bench"]["benchmark"])
        if r["arms"] is not None:
            present |= set(r["arms"]["benchmark"]) & set(ARMS_MODELS)
    return [m for m in MODEL_ORDER if m in present]


def legend_below(fig, handles, names, ncol, gap=0.07):
    fig.tight_layout(rect=(0, gap, 1, 1))
    fig.legend(handles, names, loc="lower center", ncol=ncol, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))


def grid_figure(results, ylabel, draw, handles, names, name, ncol=3, gap=0.11):
    """2 x 5 panels (wind / no wind by background); `draw(ax, result, wind)` plots one configuration."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), sharex=True)
    for i, (w, wlabel, ymax) in enumerate(WIND):
        for j, bg in enumerate(BG):
            ax, c = axes[i, j], f"{bg}_{w}"
            if i == 0:
                ax.set_title(BG[bg], fontsize=11, loc="left", color=INK)
            ax.set_ylim(0, ymax)
            ax.set_xticks([1, 4, 8, 12, 16])
            if j == 0:
                ax.set_ylabel(f"{ylabel}\n{wlabel}", fontsize=10)
            if i == 1:
                ax.set_xlabel("шаг прогноза", fontsize=10)
            if c not in results:
                ax.text(8.5, ymax / 2, "ещё считается", ha="center", va="center", color=GREY, fontsize=10)
                continue
            draw(ax, results[c], w == "wind")
    legend_below(fig, handles, names, ncol, gap=gap)
    fig.savefig(ASSETS / name, dpi=150)
    plt.close(fig)


STEPS_AXIS = np.arange(1, 17)


def position(r, model):
    """(per-step error of the prediction, per-step floor): JEPA and LeWM read by the latent probe on their latent (the
    decoder is only for pictures), flow matching read by the pixel CNN on its decoded frames."""
    bench = bench_of(r, model)
    if bench is None:
        return None
    if model == "flow":
        pp = bench.get("pixel_probe", {}).get("flow")
        return (pp["predicted"], [float(np.mean(pp["context"]))] * 16) if pp else None
    pe = bench.get("probe_error", {}).get(model)
    return (pe["predicted"], pe["true_latent"]) if pe and "pixel_probe" in bench else None


def metric(r, model, key, digits):
    bench = bench_of(r, model)
    return fmt(bench["benchmark"][model][key], digits) if bench and model in bench["benchmark"] else "—"


def position_figure(results):
    models = [m for m in models_of(results) if m != "persistence"]

    def draw(ax, r, wind):
        cube, o = r["bench"]["cube"], r["oracle"]
        ax.plot(STEPS_AXIS, cube["persistence"]["position_error"], color=GREY, lw=1.6, ls=":")
        if wind and o is not None:
            ax.plot(STEPS_AXIS, o["actions_last_wind"], color=INK, lw=1, ls="-.")
        ymax = ax.get_ylim()[1]
        for m in models:
            p = position(r, m)
            if p is None:
                continue
            if min(p[0]) > ymax:
                ax.text(1, ymax * 0.93, f"{MODEL_NAMES[m]}: {np.mean(p[0]):.0f} px, выше графика", color=MODEL_COLOURS[m],
                        fontsize=8.5, va="top")
                continue
            ax.plot(STEPS_AXIS, p[1], color=MODEL_COLOURS[m], lw=1.2, ls="--")
            ax.plot(STEPS_AXIS, p[0], color=MODEL_COLOURS[m], lw=2.4)

    handles = [Line2D([], [], color=GREY, lw=1.6, ls=":"), Line2D([], [], color=INK, lw=1, ls="-.")]
    names = ["копия последнего кадра", "известные действия + последний ветер"]
    floors = {"tile_factored": "probe на latent настоящего кадра", "lewm6": "probe на latent настоящего кадра",
              "flow": "CNN на decoded context"}
    for m in models:
        handles += [Line2D([], [], color=MODEL_COLOURS[m], lw=2.4), Line2D([], [], color=MODEL_COLOURS[m], lw=1.2, ls="--")]
        names += [f"{MODEL_NAMES[m]}, прогноз", f"{MODEL_NAMES[m]}, {floors[m]}"]
    grid_figure(results, "ошибка позиции, px", draw, handles, names, "position_error.png", ncol=4, gap=0.11)


def fmt(x, digits=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def table(headers, rows):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def summary_table(results, cfgs):
    models = [m for m in models_of(results) if m != "persistence"]
    rows = []
    for c in cfgs:
        r = results[c]
        each = lambda f: " / ".join(f(m) for m in models)
        err = lambda m: fmt(position(r, m)[0][7], 1) if position(r, m) else "—"
        floor = lambda m: fmt(np.mean(position(r, m)[1]), 1) if position(r, m) else "—"
        rows.append([RU[c], each(err), each(floor), each(lambda m: metric(r, m, "gFVD", 4)), each(lambda m: metric(r, m, "AF", 2))])
    return table([f"конфигурация, {' / '.join(MODEL_NAMES[m] for m in models)}", "ошибка позиции на 8 шаге, px",
                  "пол probe, px", "gFVD", "AF"], rows)


def rebenched(r):
    return set(r["bench"]["benchmark"]) <= {"persistence", "tile_factored", "flow"} and "pixel_probe" in r["bench"]


def html(results):
    cfgs = [c for c in R.ORDER if c in results]
    scenes = []
    for c in cfgs:
        d = RUNS / c
        if not rebenched(results[c]):
            scenes.append(f"\n<h4>{RU[c]}</h4>\n<p class=\"note\">сцены пересчитываются</p>")
            continue
        probe_gif = next((n for n in ("probe_arms_scene0.gif", "probe_scene0.gif") if (d / n).exists()), None)
        names = ["scene0.gif", "scene0_strip.png"] + ([probe_gif] if probe_gif else [])
        for name in names:
            shutil.copy(d / name, ASSETS / f"{c}_{name}")
        columns = "JEPA, LeWM и flow matching" if probe_gif == "probe_arms_scene0.gif" else "JEPA и flow matching"
        probe = (f'<figure><img src="assets/{c}_{probe_gif}" alt="{c} probe"><figcaption>Та же сцена, свежий '
                 f'rollout, колонки {columns}. Рамка = оценка probe: жёлтая по latent, из которого кадр декодирован, '
                 f'белая по самому кадру у flow matching.</figcaption></figure>' if probe_gif else "")
        scenes.append(f"""
<h4>{RU[c]}</h4>
<figure><img src="assets/{c}_scene0.gif" alt="{c}"></figure>
<figure><img src="assets/{c}_scene0_strip.png" alt="{c} strip"></figure>
{probe}""")
    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Красный кубик: JEPA против flow matching в стохастическом мире</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font-family:Georgia,"Times New Roman",serif;background:#faf9f6;color:#1c1c1c;margin:0;line-height:1.55}}
.wrap{{max-width:900px;margin:0 auto;padding:32px 20px 80px}}
h1{{font-size:2rem;line-height:1.15;margin:.2em 0 .4em}} h2{{font-size:1.4rem;margin:2em 0 .6em}}
h4{{font-size:1rem;margin:1.4em 0 .4em;color:#444}}
p{{font-size:1.05rem;margin:.6em 0}} figure{{margin:1.2em 0}} img{{max-width:100%;display:block;border:1px solid #e4e1da}}
figcaption{{font-size:.9rem;color:#555;margin-top:.4em}}
table{{border-collapse:collapse;font-size:.9rem;margin:.8em 0;font-family:"DejaVu Sans",Arial,sans-serif}}
th,td{{border-bottom:1px solid #ddd;padding:5px 10px;text-align:left;vertical-align:top}} th{{background:#f0ede6}}
code{{background:#efece5;padding:1px 4px;font-size:.9em}} .note{{color:#666;font-size:.95rem}}
</style></head><body><div class="wrap">
<h1>Красный кубик: JEPA против flow matching в стохастическом мире</h1>
<p class="note">Черновик раздела для поста. {"Досчитаны все 10 конфигураций." if len(cfgs) == 10 else
f"Досчитано {len(cfgs)} конфигураций из 10, страница пересобирается по мере поступления результатов."}</p>

<h2>Коротко</h2>
<p>Игрушечный мир: красный кубик 6×6 на кадре 63×63, единственный предсказуемый объект, а фон либо чёрный, либо шум.
Кубиком управляют четыре действия, к каждому ходу добавляется скрытый ветер. Три world model по 10 минут на одной
A100: JEPA с objective LeWM на решётке latent tinywmcraft, LeWM как в статье, с class token и autoregressive
predictor, и flow matching из tinywmcraft. Смотрим, кто ведёт кубик под шумом и ветром.</p>
<p>На чёрном фоне с ветром работают все три, и LeWM ведёт кубик лучше всех. На чёрном фоне без ветра LeWM с class
token схлопывает кадр в две сотни дискретных прототипов, в которых далёкие позиции перемешаны; точная позиция остаётся
только в направлениях latent с ничтожной дисперсией, и predictor её не проносит: probe читает позицию с настоящего
кадра, а с предсказанного latent нет. Шум ломает JEPA-объектив по-разному.
LeWM с class token на любом шуме выучивает только текстуру фона: predictive loss близок к нулю, а кубика в latent
нет, probe читает его на уровне случайного угадывания. JEPA на решётке latent держит кубик на шуме без ветра и на
шуме, общем для эпизода, но на сером шуме, который перерисовывается каждый кадр, с ветром уходит в другой минимум:
predictive loss не падает, latent кодирует шум, кубик потерян, три seed из трёх. Flow matching ведёт кубик на всех
шумах, кроме цветного шума на эпизод без ветра, где она развалилась в серую кашу. gFVD у flow matching почти всегда
лучше, потому что она рисует свежий шум, а JEPA декодирует ровный фон с кубиком: здесь gFVD измеряет фон.</p>

<h2>Среда</h2>
<p>Кубик двигается на 3 пикселя за действие: влево, вверх, вправо, вниз. Ветер это целочисленное случайное блуждание
от −2 до 2 по каждой оси, начинается с нуля и на каждом шаге с вероятностью 0.8 остаётся прежним. Красная рамка
шириной в пиксель стоит в одном пикселе от края, касание рамки это смерть с наградой −1. Эпизод обрывается на 128 шаге.
Фон: чёрный, серый шум или цветной шум; шум либо новый на каждом кадре, либо один на весь эпизод. Ветер включён или
выключен, всего 10 конфигураций. На чёрный фон и цветной шум собрано по 100 тысяч переходов случайной политики,
на серый шум по 500 тысяч.</p>
<figure><img src="assets/env_configs.png" alt="конфигурации среды"><figcaption>По одному стартовому кадру на конфигурацию.</figcaption></figure>

<h2>Модели</h2>
<p>Общая архитектура взята из tinywmcraft. Encoder: тайлы 7×7, шесть transformer blocks на решётке 9×9, свёртка со
stride 2 до решётки 4×4, latent 4×4×32, 2.6M параметров. Dynamics: transformer из шести блоков ширины 256, spatial
attention по 16 токенам кадра и causal attention по кадрам, 8.3M параметров. У JEPA этот transformer выдаёт latent
следующего кадра и обучается на MSE до него плюс SIGReg с весом 0.09, без stop-gradient и без EMA, ровно как в LeWM;
history 8 кадров. Пиксельный decoder для JEPA обучается после бюджета на замороженных latent и нужен только для картинок
и метрик. Flow matching это рецепт tinywmcraft как есть, 8 sampling steps.</p>
<p>LeWM повторяет статью, только encoder меньше под наш бюджет: ViT из шести блоков ширины 192, кадр читается через
class token, latent 192 чисел, predictor из статьи: шесть AdaLN blocks с 16 головами, MLP 2048, history 3 кадра,
dropout 0.1, тот же objective. Всего 15M параметров. На сером шуме на кадр без ветра прогнали и полный ViT из
12 блоков, 18M: результат тот же, что у шести.</p>

<h2>Результаты</h2>
<p>Все окна одни и те же для обеих моделей: 8 кадров context, 16 шагов прогноза, 512 окон для метрик по кубику и
2048 для gFVD и action following. Позицию кубика читаем у каждой модели там, где она у неё живёт. JEPA: probe по
predicted latent, decoder ей не нужен и на странице он только для картинок. Probe per-tile: один MLP, общий для 16
токенов latent, даёт каждому тайлу logit и offset, позиция = softmax по тайлам × (центр тайла + offset); на latent
настоящего кадра он ошибается на 0.25 px. У LeWM latent один вектор, probe тот же MLP по всему вектору; latent
нормируем по размерностям или PCA-whitening, что лучше на отложенных кадрах. Пол probe около 20 px значит, что
позиции в latent нет: угадывание центра кадра даёт столько же. Flow matching: CNN по декодированному кадру,
decoder у неё часть модели.
CNN: свёртки до сетки 8×8, logit и offset на ячейку, тот же softmax; обучен на 100 тысячах настоящих кадров, у
которых фон вокруг кубика случайно оставлен, залит серым или серым с шумом; на настоящих кадрах ошибается на 0.3 px.
Оба probe обучены на training кадрах, ошибка L1 в пикселях.</p>
<figure><img src="assets/position_error.png" alt="ошибка позиции"><figcaption>Ошибка позиции прогноза. Тонкий пунктир
того же цвета: пол probe, у JEPA на latent настоящего будущего кадра, у flow matching на декодированных кадрах
context. Чёрный штрих-пунктир: известные действия плюс ветер последнего кадра context, без модели.</figcaption></figure>
{summary_table(results, cfgs)}
<p>Ветер ограничивает всех. Он блуждает со средней силой 0.8 пикселя на ось, ходы случайной политики взаимно гасятся,
и после 8 шагов позицию определяет ветер: даже зная все действия и последний ветер, на 16 шаге ошибаешься на 24 пикселя.
Информативны первые 8 шагов и расстояние до пунктира.</p>
<p>Чёрный фон с ветром: LeWM на 8 шаге ошибается на 9 px, JEPA и flow matching на 12, а нижняя граница по известным
действиям и последнему ветру 9. Шум на кадр без ветра: flow matching держит кубик с ошибкой меньше пикселя на всём
горизонте, JEPA на решётке уходит на 4–6 px к 8 шагу, LeWM кубика не видит. Шум на эпизод: JEPA и flow matching
идут вровень, на сером с ветром JEPA чуть впереди, на цветном без ветра flow matching развалилась и JEPA одна ведёт
кубик. Серый шум на кадр с ветром: у JEPA пол probe 21 px, latent кодирует шум вместо кубика; без ветра тот же
рецепт сходится три seed из трёх, с ветром ни разу, batch 256 не помогает.</p>
<figure><img src="assets/infographic_black.png" alt="чёрный фон: три модели"><figcaption>Чёрный фон без ветра и с
ветром: прогноз трёх моделей от одних и тех же кадров context, t-SNE latent настоящих кадров с цветом по x кубика и
ошибка позиции по шагам. LeWM без ветра схлопывает latent в ~200 значений: в 48 направлениях с дисперсией кадры
неразличимы, позиция остаётся в направлениях с дисперсией в сотни раз меньше, decoder её не видит и predictor не
проносит. Та же архитектура с ветром даёт гладкое многообразие с позицией в первых направлениях.</figcaption></figure>

<h2>Сцены</h2>
<p class="note">Слева реальная среда, дальше JEPA и flow matching. Первые 8 кадров в колонках моделей прошли через
encoder и decoder модели, красная рамка отмечает предсказанные кадры. На чёрном и сером фоне во второй гифке есть
колонка LeWM.</p>
{"".join(scenes)}

<h2>Поиграть самому</h2>
<p>Реальная среда и обе модели рядом, в одном окне. Первые 7 кадров эпизода модели видят настоящими, дальше каждая
живёт в своём воображении и отвечает на те же клавиши. Окно работает, пока запущен сервер на multigpu-vm и проброшен
порт:</p>
<pre><code>ssh multigpu-vm 'cd ~/personal/tinycraft-jepa &amp;&amp; JAX_PLATFORMS=cpu setsid nohup \\
    ~/personal/tinywmcraft-slim/.venv/bin/python -m studies.redcube.serve --root /dev/shm/redcube \\
    --flow-code ~/personal/codex-wm-push-20260916/recipes/model-51ffe8df --port 8765 &gt; runs/redcube/serve.log 2&gt;&amp;1 &amp;'
ssh -f -N -L 8765:localhost:8765 multigpu-vm</code></pre>
<iframe src="http://localhost:8765/" title="red cube play" style="width:100%;height:560px;border:1px solid #ddd;border-radius:6px;background:#141416"></iframe>
<p class="note">Стрелки или WASD = ход, R = новый эпизод, выпадающий список = конфигурация. Если рамка пустая, сервер
не запущен. То же в отдельной вкладке: <a href="http://localhost:8765/">localhost:8765</a>.</p>
</div></body></html>"""
    (OUT / "index.html").write_text(page)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    results = {c: load(c) for c in R.ORDER}
    results = {c: r for c, r in results.items() if r is not None}
    if not (ASSETS / "env_configs.png").exists():
        env_preview()
    if results:
        position_figure(results)
    html(results)
    print("report ->", OUT / "index.html", "configs:", list(results))


if __name__ == "__main__":
    main()
