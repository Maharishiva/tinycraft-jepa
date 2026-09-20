"""Figures for the independent Russian note, generated from measured JSON artifacts.

Run with a Python environment containing numpy and matplotlib. No training or image edits.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'blog/assets/research-note'
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED = '#18283b', '#536176'
COLORS = {'tile_factored': '#7650a5', 'lewm6': '#bd6616', 'flow': '#147d75', 'tile_ar': '#1553aa'}
LABELS = {'tile_factored': 'JEPA, решётка', 'lewm6': 'LeWM, общий вектор', 'flow': 'Flow matching', 'tile_ar': 'JEPA, другой предиктор'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.edgecolor': '#cdd6e1',
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'svg.fonttype': 'none', 'savefig.bbox': 'tight'})

def read(path):
    return json.loads((ROOT/path).read_text())

def save(fig, name):
    fig.savefig(OUT/f'{name}.svg', metadata={'Date': None})
    fig.savefig(OUT/f'{name}.png', dpi=170)
    plt.close(fig)

def cube_curves(mobile=False):
    panels = [('off_still', 'Чёрный фон · без ветра'), ('off_wind', 'Чёрный фон · ветер'),
              ('gray_frame_still', 'Свежий серый шум · без ветра'), ('gray_frame_wind', 'Свежий серый шум · ветер')]
    fig, axs = plt.subplots(4 if mobile else 2, 1 if mobile else 2,
                            figsize=(4.3, 14.1) if mobile else (10.5, 7.4))
    steps = np.arange(1, 17)
    for ax, (cfg, title) in zip(axs.flat, panels):
        base = read(f'runs/redcube/{cfg}/bench.json')
        arms = read(f'runs/redcube/{cfg}/allarms/bench.json')
        for model in ('tile_factored', 'lewm6', 'flow'):
            values = (base['pixel_probe']['flow']['predicted'] if model == 'flow' else
                      (arms if model == 'lewm6' else base)['probe_error'][model]['predicted'])
            assert len(values) == 16 and np.isfinite(values).all()
            ax.plot(steps, values, color=COLORS[model], lw=2.2, label=LABELS[model])
        if cfg == 'off_still':
            ax.plot(steps, arms['probe_error']['tile_ar']['predicted'], color=COLORS['tile_ar'], lw=2,
                    ls=(0, (3, 2)), label=LABELS['tile_ar'])
        ax.set_title(title, loc='left', fontsize=12.5 if mobile else 12, weight='bold', pad=13)
        ax.set(xlabel='Шаг прогноза', ylabel='Ошибка позиции, px', xlim=(1,16), ylim=(0,None))
        ax.set_xticks([1,4,8,12,16])
        ax.grid(axis='y', alpha=.25)
    handles = [Line2D([0],[0],color=COLORS[m], lw=2.2, ls='--' if m == 'tile_ar' else '-') for m in COLORS]
    fig.legend(handles, [LABELS[m] for m in COLORS], loc='lower center', ncol=1 if mobile else 2,
               frameon=False, fontsize=10.5, bbox_to_anchor=(.5,0))
    fig.tight_layout(rect=(0,.105 if mobile else .12,1,1), h_pad=2.5, w_pad=2)
    save(fig, 'cube-horizons-mobile' if mobile else 'cube-horizons')

def craftax_quality(mobile=False):
    rows = {r['tag']: r for r in map(json.loads,(ROOT/'ledger.jsonl').read_text().splitlines())}
    ev = read('output/research-review/evidence.json')
    flow = next(r for r in ev['craftax_flow_ledger'] if r['train_seed'] == 0)
    labels = ['LeWM, общий вектор', 'LeWM, решётка', 'Копия последнего кадра', 'Flow matching']
    values = [rows['lewm_s0_lr1e-3']['gFVD'], rows['lewm_s0_grid']['gFVD'], rows['persistence']['gFVD'],flow['gFVD']]
    colors = [COLORS['lewm6'], COLORS['tile_factored'], '#8490a2', COLORS['flow']]
    fig, ax = plt.subplots(figsize=(4.3,4.2) if mobile else (9.8,3.25))
    y = np.arange(4)
    ax.barh(y,values,color=colors,height=.58)
    if mobile:
        labels=['LeWM\nобщий вектор','LeWM\nрешётка','Копия\nкадра','Flow\nmatching']
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.tick_params(axis='y',length=0,pad=12)
    ax.set_xscale('log');ax.set_xlim(.0005,.085)
    ax.set_xticks([.001,.01],['0,001','0,01']);ax.minorticks_off()
    ax.set_xlabel('gFVD · меньше лучше\nлогарифмическая шкала' if mobile else 'gFVD, логарифмическая шкала · меньше лучше')
    for i,v in enumerate(values):ax.text(v*1.09,i,f'{v:.5f}'.replace('.',','),va='center',fontsize=11)
    fig.tight_layout();save(fig,'craftax-quality-mobile' if mobile else 'craftax-quality')

def control(mobile=False):
    goal = read('artifacts/s0_grid_h4_plan.json')['craftax']
    cls = read('artifacts/s0_lr1e-3_h4_plan.json')['craftax']
    reward = read('artifacts/s0_grid_reward.json')['reward_mpc']
    fig, axs = plt.subplots(2 if mobile else 1,1 if mobile else 2,figsize=(4.3,8.8) if mobile else (10.5,3.9))
    ax,b = axs.flat
    vals=[goal['random']['reached']*100,cls['planned']['reached']*100,goal['planned']['reached']*100]
    goal_labels = (['Случайный\nплан','LeWM\nвектор','LeWM\nрешётка'] if mobile else
                   ['Случайный\nплан','LeWM\nобщий вектор','LeWM\nрешётка'])
    ax.bar(goal_labels,vals,color=['#8490a2',COLORS['lewm6'],COLORS['tile_factored']],width=.58)
    for i,v in enumerate(vals):ax.text(i,v+2,f'{v:.0f}%',ha='center',fontsize=12)
    ax.set_title('Дойти до близкой цели',loc='left',fontsize=13,weight='bold',pad=13)
    ax.set(ylabel='Успех, %',ylim=(0,102));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    keys=['random','myopic','greedy','plan']
    vals=[reward[k]['return'] for k in keys]
    sem=[reward[k]['return_std']/np.sqrt(reward['envs']) for k in keys]
    reward_labels = (['Случ.\nходы','Награда\nшага','Шаг\n+ value','CEM\nh=6'] if mobile else
                     ['Случайные\nдействия','По награде\nодного шага','Один шаг\n+ value','CEM\n6 шагов'])
    b.bar(reward_labels,vals,yerr=sem,capsize=3,width=.58,
          color=['#8490a2']+[COLORS['tile_factored']]*3)
    for i,(v,e) in enumerate(zip(vals,sem)):b.text(i,v+e+.035,f'{v:.2f}'.replace('.',','),ha='center',fontsize=11)
    b.set_title('Накопить награду',loc='left',fontsize=13,weight='bold',pad=13)
    b.set(ylabel='Return за 150 шагов',ylim=(0,1.9));b.grid(axis='y',alpha=.2);b.set_axisbelow(True)
    fig.tight_layout(h_pad=2.8,w_pad=2)
    save(fig,'craftax-control-mobile' if mobile else 'craftax-control')

if __name__ == '__main__':
    cube_curves();craftax_quality();control()
    with plt.rc_context({'font.size':12.5}):
        cube_curves(mobile=True);craftax_quality(mobile=True);control(mobile=True)
    print('Figures:', ', '.join(p.name for p in OUT.glob('*.svg')))
