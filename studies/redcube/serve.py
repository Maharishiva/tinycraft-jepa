"""Play the red cube in the browser: the real environment next to every world model of a configuration.

Every model is primed on the same real frames, then free-runs on your keystrokes (play.py's wrappers).

  JAX_PLATFORMS=cpu python -m studies.redcube.serve --root /dev/shm/redcube --flow-code <recipe dir> --port 8765
  open http://localhost:8765            (from the laptop: ssh -N -L 8765:localhost:8765 multigpu-vm)
"""
import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import probe
import train as T
from studies.redcube import env as R
from studies.redcube import flow as F
from studies.redcube.bench import ARMS
from studies.redcube.play import CTX, FlowWM, JepaWM, episode_start

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>red cube: play</title>
<style>
 body{margin:0;padding:18px;background:#141416;color:#e6e6e6;font:15px/1.4 Menlo,Consolas,monospace}
 #top{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
 select,button{font:inherit;padding:4px 8px;background:#26262a;color:#e6e6e6;border:1px solid #444;border-radius:4px}
 #panels{display:flex;gap:16px;flex-wrap:wrap}
 .pane{display:flex;flex-direction:column;gap:6px}
 .pane .label{font-weight:bold;font-size:14px;max-width:315px;white-space:normal}
 canvas{image-rendering:pixelated;background:#000;border:3px solid #2a2a2e}
 .pane.model canvas{border-color:#c42828}
 #status{color:#7fd08a} #dead{color:#f05a5a;font-weight:bold} .hint{color:#9aa4ae;margin-top:14px}
</style></head><body>
<div id="top"><label>конфигурация <select id="cfg"></select></label>
<button id="reset">новый эпизод (R)</button><span id="status"></span><span id="dead"></span></div>
<div id="panels"></div>
<p class="hint">стрелки / WASD = ход (влево, вверх, вправо, вниз); R или Backspace = новый эпизод. Первые 7 кадров эпизода
все модели видят настоящими, дальше каждая живёт в своём воображении. Красная рамка = кадр модели.</p>
<script>
const KEY={ArrowLeft:0,ArrowUp:1,ArrowRight:2,ArrowDown:3,a:0,w:1,d:2,s:3,A:0,W:1,D:2,S:3};
const SCALE=5; let busy=false, dead=false;
async function api(path, body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});return r.json();}
function draw(p){
  const box=document.getElementById('panels');
  if(box.children.length!==p.panels.length){box.innerHTML='';p.panels.forEach((q,i)=>{const d=document.createElement('div');d.className='pane'+(i?' model':'');
    d.innerHTML=`<div class="label">${q.label}</div><canvas width="${63*SCALE}" height="${63*SCALE}"></canvas>`;box.appendChild(d);});}
  p.panels.forEach((q,i)=>{const c=box.children[i].querySelector('canvas');const g=c.getContext('2d');g.imageSmoothingEnabled=false;
    const im=new Image();im.onload=()=>g.drawImage(im,0,0,c.width,c.height);im.src='data:image/png;base64,'+q.png;});
  dead=p.dead;document.getElementById('dead').textContent=dead?'СМЕРТЬ: коснулись рамки. R = заново':'';
  document.getElementById('status').textContent=`шаг ${p.step}   ветер ${JSON.stringify(p.wind)}   награда ${p.reward}`;
}
async function call(path, body){if(busy)return;busy=true;document.getElementById('status').textContent+=' ...';try{draw(await api(path,body));}finally{busy=false;}}
document.addEventListener('keydown',e=>{if(e.key in KEY){e.preventDefault();if(!dead)call('/api/step',{action:KEY[e.key]});}
  else if(e.key==='r'||e.key==='R'||e.key==='Backspace'){e.preventDefault();call('/api/reset',{});}});
document.getElementById('reset').onclick=()=>call('/api/reset',{});
document.getElementById('cfg').onchange=e=>call('/api/load',{config:e.target.value});
(async()=>{const c=await api('/api/configs');const s=document.getElementById('cfg');
  c.configs.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;if(n===c.current)o.selected=true;s.appendChild(o);});
  draw(await api('/api/state'));})();
</script></body></html>"""


def to_u8(img):
    img = np.asarray(img)
    return img if img.dtype == np.uint8 else np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def png64(u8):
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Game:
    def __init__(self, root, flow_code, steps, seed, prime):
        self.root, self.flow_code, self.steps, self.prime = Path(root), flow_code, steps, prime
        self.key = jax.random.PRNGKey(seed)
        self.lock = threading.Lock()
        self.cfg, self.wms, self.labels = None, [], []
        for mod in (T, probe):
            mod.CDT = jnp.float32
            mod.MP = dict(dtype=jnp.float32, param_dtype=jnp.float32)

    def configs(self):
        return [c for c in R.ORDER if (self.root / c / "flow" / "model.npz").exists()
                or any((self.root / c / a / "model_probe.npz").exists() for a in ARMS)]

    def load(self, cfg):
        wms, labels = [], []
        flow = self.root / cfg / "flow" / "model.npz"
        if flow.exists() and self.flow_code is not None:
            train_mod, _, model = F.load(self.flow_code, flow, R.N_ACTIONS)
            train_mod.CDT = jnp.float32
            train_mod.MP = dict(dtype=jnp.float32, param_dtype=jnp.float32)
            wms.append(FlowWM(train_mod, model, self.steps))
            labels.append("flow-matching WM")
        for arm, label in ARMS.items():
            path = self.root / cfg / arm / "model_probe.npz"
            if path.exists():
                bundle, _ = T.load_model(path)
                wms.append(JepaWM(bundle))
                labels.append(label)
        self.cfg, self.wms, self.labels = cfg, wms, labels
        self.reset()

    def reset(self):
        spec = R.CONFIGS[self.cfg]
        self.step_fn, self.key, self.state, self.real, self.imgs, self.dead = episode_start(spec, self.wms, self.key, self.prime)
        self.t, self.reward = 0, 0.0

    def step(self, action):
        if self.dead:
            return
        self.key, ks = jax.random.split(self.key)
        obs, self.state, reward, done = self.step_fn(self.state, action, ks)
        self.real, self.dead, self.reward = np.asarray(obs), bool(done), float(reward)
        self.imgs = [wm.step(action) for wm in self.wms]
        self.t += 1

    def payload(self):
        panels = [{"label": "РЕАЛЬНАЯ СРЕДА", "png": png64(to_u8(self.real))}]
        panels += [{"label": l, "png": png64(to_u8(im))} for l, im in zip(self.labels, self.imgs)]
        return {"config": self.cfg, "step": self.t, "dead": self.dead, "reward": self.reward,
                "wind": np.asarray(self.state.wind).tolist(), "panels": panels}


class Handler(BaseHTTPRequestHandler):
    game = None

    def _send(self, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        g = self.game
        if self.path == "/":
            return self._send(PAGE, "text/html; charset=utf-8")
        if self.path == "/api/configs":
            return self._send(json.dumps({"configs": g.configs(), "current": g.cfg}))
        if self.path == "/api/state":
            with g.lock:
                return self._send(json.dumps(g.payload()))
        self.send_error(404)

    def do_POST(self):
        g = self.game
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        with g.lock:
            if self.path == "/api/load":
                g.load(body["config"])
            elif self.path == "/api/reset":
                g.reset()
            elif self.path == "/api/step":
                g.step(int(body["action"]))
            else:
                return self.send_error(404)
            return self._send(json.dumps(g.payload()))

    def log_message(self, fmt, *args):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/dev/shm/redcube"))
    p.add_argument("--flow-code", type=Path, default=None)
    p.add_argument("--config", default=None, help="configuration to load first (default: the first available)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prime", type=int, default=CTX - 1)
    p.add_argument("--steps", type=int, default=8, help="Euler steps per frame of the flow model")
    args = p.parse_args()
    game = Game(args.root, args.flow_code, args.steps, args.seed, args.prime)
    first = args.config or game.configs()[0]
    print(f"[serve] loading {first} ...", flush=True)
    game.load(first)
    Handler.game = game
    print(f"[serve] http://localhost:{args.port}  configs {game.configs()}  models {game.labels}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
