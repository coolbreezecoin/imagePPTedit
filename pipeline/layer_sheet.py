"""把某页的所有图层拼成一张检查图：背景 + 每个 RGBA 图层（棋盘底）"""
import sys, json, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw, ImageFont
deck=sys.argv[1]
for n in [int(p) for p in sys.argv[2:]]:
    d=f"{ROOT}/work/{deck}/layers/p{n:02d}"; man=json.load(open(f"{d}/manifest.json"))
    tiles=[]
    font=ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 20)
    for L in man["layers"]:
        im=Image.open(f"{d}/{L['file']}").convert("RGBA")
        # 棋盘底
        W,H=im.size; bg=Image.new("RGBA",(W,H),(200,200,200,255)); dr=ImageDraw.Draw(bg); s=16
        for y in range(0,H,s):
            for x in range(0,W,s):
                if (x//s+y//s)%2==0: dr.rectangle([x,y,x+s-1,y+s-1],fill=(150,150,150,255))
        comp=Image.alpha_composite(bg,im)
        scale=min(1.0, 600/max(W,H)); comp=comp.resize((max(1,int(W*scale)),max(1,int(H*scale))))
        canvas=Image.new("RGBA",(comp.width,comp.height+26),(40,40,40,255)); canvas.paste(comp,(0,26))
        ImageDraw.Draw(canvas).text((4,2),f"{L['name']} {L['box']} {W}x{H}",fill=(255,255,0,255),font=font)
        tiles.append(canvas)
    # 排版：每行最多 3 个
    rows=[tiles[i:i+3] for i in range(0,len(tiles),3)]
    RW=max(sum(t.width for t in r)+10*len(r) for r in rows); RH=sum(max(t.height for t in r)+10 for r in rows)
    sheet=Image.new("RGBA",(RW,RH),(30,30,30,255)); y=0
    for r in rows:
        x=0; h=max(t.height for t in r)
        for t in r: sheet.paste(t,(x,y)); x+=t.width+10
        y+=h+10
    sheet.convert("RGB").save(f"{d}/_sheet.jpg", quality=85); print("sheet", n, sheet.size)
