# 预下载/初始化模型权重：rembg(u2net_human_seg, isnet-general-use), LaMa(big-lama)
import numpy as np, time
t=time.time()
from rembg import new_session
for name in ["u2net_human_seg","isnet-general-use"]:
    s=new_session(name); print("rembg session ok:", name, round(time.time()-t,1),"s", flush=True)
from simple_lama_inpainting import SimpleLama
from PIL import Image
lama=SimpleLama()
img=Image.new("RGB",(256,256),(120,80,200)); m=Image.new("L",(256,256),0)
out=lama(img,m); print("lama ok", out.size, round(time.time()-t,1),"s", flush=True)
