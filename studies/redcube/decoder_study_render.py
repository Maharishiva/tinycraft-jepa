"""Render measured decoder-study arrays with translucent yellow probe boxes.

Every row uses the same source window. Upper row: true future latents (decoder isolation).
Lower row: predicted latents (dynamics plus decoder). No synthetic replacement of model frames.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont

YELLOW=np.array([255,230,0],float)

def box(frame,position,alpha=.55):
    out=frame.copy()
    if not np.isfinite(position).all():return out
    x,y=np.clip(np.rint(position),0,62).astype(int)
    x0,y0,x1,y1=max(x-1,0),max(y-1,0),min(x+6,62),min(y+6,62)
    mask=np.zeros((63,63),bool)
    mask[y0,x0:x1+1]=mask[y1,x0:x1+1]=True
    mask[y0:y1+1,x0]=mask[y0:y1+1,x1]=True
    out[mask]=np.rint((1-alpha)*out[mask]+alpha*YELLOW).astype(np.uint8)
    return out

def font(size):
    candidates=['/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    return next((ImageFont.truetype(p,size) for p in candidates if Path(p).exists()),ImageFont.load_default())

def render(root,out,scene=0):
    out.mkdir(parents=True,exist_ok=True)
    a=np.load(root/'visuals.npz');report=json.loads((root/'results.json').read_text())
    labels=['Реальный кадр','Старый декодер','MSE + whitening','Диффузия + whitening']
    keys=['gt','old','mse','diffusion'];scale=4;cell=63*scale;gap=12;left=168;header=45;rowh=cell+36
    w=left+4*(cell+gap)+gap;h=header+2*rowh+37
    frames=[]
    for t in range(16):
        im=Image.new('RGB',(w,h),'#f5f7fa');draw=ImageDraw.Draw(im)
        for j,label in enumerate(labels):draw.text((left+j*(cell+gap),12),label,font=font(17),fill='#18283b')
        for row,stage in enumerate(['true','predicted']):
            y=header+row*rowh
            draw.multiline_text((12,y+cell//2-25),'Истинный латент\nТолько декодер' if stage=='true' else 'Латент прогноза\nДинамика\nи декодер',font=font(15),fill='#18283b',spacing=6)
            for j,k in enumerate(keys):
                if k=='gt':raw=a['gt'][scene,8+t];p=a['pos'][scene,8+t]
                else:
                    raw=a[f'{k}_{stage}'][scene,t]
                    # Yellow indicates the POSITION READ OUT OF THE LATENT for all decoder columns.
                    p=a['latent_probe_true'][scene,8+t] if stage=='true' else a['latent_probe_pred'][scene,t]
                rendered=Image.fromarray(box(raw,p)).resize((cell,cell),Image.Resampling.NEAREST)
                x=left+j*(cell+gap);im.paste(rendered,(x,y))
                if row==1 and j>0:draw.rectangle((x-2,y-2,x+cell+1,y+cell+1),outline='#b83242',width=2)
            draw.text((left,y+cell+7),'Жёлтая рамка: истинная позиция слева; latent probe у всех декодеров. Латент один и тот же.',font=font(14),fill='#536176')
        draw.text((12,h-28),f'{report["provenance"]["config"]} / {report["provenance"]["arm"]}   window {scene}, future step {t+1}/16   diffusion sample 0',font=font(15),fill='#18283b')
        frames.append(np.asarray(im))
    imageio.mimsave(out/f'decoders-scene{scene}.gif',frames+[frames[-1]]*4,duration=250,loop=0)
    Image.fromarray(frames[7]).save(out/f'decoders-scene{scene}-step8.png')
    np.savez(out/f'decoders-scene{scene}-probes.npz',true=a['pos'][scene],latent_true=a['latent_probe_true'][scene],latent_pred=a['latent_probe_pred'][scene])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();render(a.root,a.out)
