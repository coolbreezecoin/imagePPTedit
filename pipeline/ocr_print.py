import json,sys
for p in sys.argv[1:]:
    items=json.load(open(f"work/繁星02/ocr/{p}.json"))
    print(f"== {p} ==")
    for i,it in enumerate(items):
        b=[int(round(v)) for v in it["box"]]
        print(f"{i:>2} {b} h={b[3]-b[1]:>3} {it['text']}")
