with open("app/backend/detection/gliner_engine.py", "r", encoding="utf-8") as f:
    c = f.read()
with open("app/backend/detection/gliner_engine.py", "w", encoding="utf-8") as f:
    f.write(c.replace("split(char(10))[0]", "split(chr(10))[0]".replace("chr(10)", chr(10).ToString()))
