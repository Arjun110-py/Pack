import sys
from tkinter import filedialog

def read_file(filepath):
    with open(filepath) as f:
        data = f.read()
    return data

class Lexer:
    def __init__(self, data, filepath):
        self.tokens = []
        self.data = data
        self.filepath = filepath
        self.char = None
        self.idx = -1
        

def run(filepath):
    data = read_file(filepath)
    lexer = Lexer(data, filepath)
    lexer.tokenize()
    print(lexer.tokens)

if __name__ == "__main__":
    try:
        run(sys.argv[1])
    except Exception:
        run(filedialog.askopenfilename())