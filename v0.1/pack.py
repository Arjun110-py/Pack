import sys, os, itertools, string

# Constants
DIGITS = "0123456789"
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS
KEYWORDS = ["let", "const"]
TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_INCREMENT = "INCREMENT"
TT_DECREMENT = "DECREMENT"
TT_PLUS = "PLUS"
TT_MINUS = "MINUS"
TT_MUL = "MUL"
TT_DIV = "DIV"
TT_IDIV = "IDIV"
TT_POW = "POW"
TT_KEYWORD = "KEYWORD"
TT_IDENTIFIER = "IDENTIFIER"
TT_EQ = "EQ"
TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"
TT_EOL = "EOL"
NEWLN = '\n'

# Functions
def split_list(lst, delimiter, attribute=None):
    return [list(y) for x, y in itertools.groupby(lst, lambda z: (getattr(z, attribute) if attribute else z) == delimiter) if not x]

# Token
class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value
        self.pos_start = pos_start
        if pos_end: self.pos_end = pos_end
        else: self.pos_end = self.pos_start
    
    def matches(self, type_, values):
        return self.type == type_ and self.value in values
    
    def __repr__(self):
        return f"{self.type}{(':' + str(self.value)) if self.value != None else ''}"

# Errors
class Error:
    def __init__(self, pos_start, pos_end, name, details, fn):
        self.details = details
        self.pos_start = pos_start
        self.pos_end = pos_end if pos_end else pos_start
        self.name = name
        self.fn = fn
    
    def as_string(self):
        return f"""File '{self.fn}', Line {self.pos_start.ln_count + 1}
{self.name}: {self.details}
{self.pos_start.data.split(NEWLN)[self.pos_start.ln_count]}
{' ' * (self.pos_start.col_count - 1) + ('^' * (self.pos_end.col_count - (self.pos_start.col_count if self.pos_start.col_count != self.pos_end.col_count else 1) if self.pos_start.col_count != self.pos_end.col_count else 1))}"""

class NameError_(Error): 
    def __init__(self, fn, pos_start, pos_end=None):
        super().__init__(pos_start, pos_end, 'NameError', f"'{pos_start.char}' is not defined", fn)

class SyntaxError_(Error):
    def __init__(self, pos_start, pos_end, details, fn):
        super().__init__(pos_start, pos_end, "SyntaxError", details, fn)


class RTError(Error):
    def __init__(self, pos_start, pos_end, name, details, context):
        super().__init__(pos_start, pos_end, name, details, context.display_name)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f"{self.name}: {self.details}\n"
        result += self.pos_start.data.split(NEWLN)[self.pos_start.ln_count] + '\n'
        result += f"{' ' * (self.pos_start.col_count - 1) + ('^' * (self.pos_end.col_count - (self.pos_start.col_count if self.pos_start.col_count != self.pos_end.col_count else 1) if self.pos_start.col_count != self.pos_end.col_count else 1))}"
        return result
    
    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context
        while ctx:
            result = f'    File {pos.fn}, line {str(pos.ln_count + 1)}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent
        
        return 'Traceback (most recent call last):\n' + result

class ZeroDivisionError_(RTError):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, "ZeroDivisionError", details, context)

class RTNameError_(RTError): 
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'NameError', details, context)

class TypeError_(RTError): 
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'TypeError', details, context)

# Position
class Position:
    def __init__(self, data, fn, ln_count=0, col_count=0):
        self.data = data
        self.ln_count = ln_count
        self.col_count = col_count
        self.idx = -1
        self.fn = fn
        try: self.char = data.split('\n')[ln_count][col_count]
        except: self.char = None
    
    def advance(self):
        self.idx += 1
        try:
            self.col_count += 1
            if self.data[self.idx] == '\n':
                self.ln_count += 1
                self.col_count = 0
            self.char = self.data[self.idx]
        except Exception: self.char = None
        return self
    
    def copy(self):
        return Position(self.data, self.fn, self.ln_count, self.col_count)

# Lexer
class Lexer:
    def __init__(self, data, path):
        self.pos = Position(data, path)
        self.ln_count = 1
        self.char = None
        self.path = path
        self.tokens = []
        self.advance()
    
    def advance(self):
        self.char = self.pos.advance().char
    
    def get_number(self):
        num_str = ""
        dot_count = 0
        pos_start = self.pos.copy()
        while self.char and self.char in DIGITS + '.':
            if self.char == '.':
                if dot_count: break
                dot_count += 1
            num_str += self.char
            self.advance()
        
        return Token(TT_FLOAT, float(num_str), pos_start, self.pos.copy()) if dot_count else Token(TT_INT, int(num_str), pos_start, self.pos.copy())

    def get_id(self):
        id_str = ''
        pos_start = self.pos.copy()
        while self.char != None and self.char in LETTERS_DIGITS + '_':
            id_str += self.char
            self.advance()
        
        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos.copy())
    
    def tokenize(self):
        tokens = []
        while self.char:
            if self.char in ' \t': self.advance()
            elif self.char in DIGITS + '.':
                tokens.append(self.get_number())

            elif self.char == '+':
                pos_start = self.pos.copy()
                self.advance()
                if self.char == "+":
                    tokens.append(Token(TT_INCREMENT, pos_start=pos_start, pos_end=self.pos.copy()))
                    self.advance()
                else: tokens.append(Token(TT_PLUS, pos_start=self.pos.copy()))

            elif self.char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos.copy()))
                self.advance()

            elif self.char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos.copy()))
                self.advance()

            elif self.char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos.copy()))
                self.advance()

            elif self.char == '/':
                pos = self.pos.copy()
                self.advance()
                if self.char == '/':
                    tokens.append(Token(TT_IDIV, pos_start=pos, pos_end=self.pos.copy()))
                    self.advance()
                else: tokens.append(Token(TT_DIV, pos_start=pos))

            elif self.char == '=':
                tokens.append(Token(TT_EQ, pos_start=self.pos.copy()))
                self.advance()
            
            elif self.char in LETTERS_DIGITS + "_":
                tokens.append(self.get_id())
            
            elif self.char in ';\n':
                tokens.append(Token(TT_EOL, pos_start=self.pos.copy()))
                self.advance()

            elif self.char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos.copy()))
                self.advance()

            elif self.char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos.copy()))
                self.advance()

            else:
                return NameError_(self.path, self.pos.copy())
        tokens.append(Token(TT_EOL, pos_start=self.pos.copy()))
        self.tokens = tokens
        return None

# Nodes
class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = tok.pos_start
        self.pos_end = tok.pos_end

class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = left_node.pos_start
        self.pos_end = right_node.pos_end

class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = op_tok.pos_start
        self.pos_end = node.pos_end

class VarAssignNode: 
    def __init__(self, type_, name, node):
        self.type = type_
        self.name = name
        self.node = node

        self.pos_start = name.pos_start
        self.pos_end = node.pos_end

class VarAccessNode:
    def __init__(self, name):
        self.name = name
        self.pos_start = name.pos_start
        self.pos_end = name.pos_end

class VarChangeNode:
    def __init__(self, name, op_tok, value=None):
        self.name = name
        self.pos_start = name.pos_start
        if value: self.pos_end = value.pos_end
        else: self.pos_end = op_tok.pos_end

# Parse Result
class ParseResult:
    def __init__(self):
        self.node = None
        self.error = None
    
    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            return res.node
        
        return res

    def success(self, node):
        self.node = node
        return self
    
    def failure(self, error):
        self.error = error
        return self
    

# Parser
class Parser:
    def __init__(self, tokens, fn):
        self.tokens = tokens
        self.tok_idx = -1
        self.fn = fn
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        else: self.current_tok = None
        return self.current_tok

    def parse(self):
        res = self.expr()
        if not res.error and self.current_tok.type != TT_EOL:
            return res.failure(SyntaxError_(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected valid operator",
                self.fn
            ))
        return res

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        elif tok.type == TT_IDENTIFIER:
            res.register(self.advance())
            return res.success(VarAccessNode(tok))

        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(SyntaxError_(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')'"
                ))
        return res.failure(SyntaxError_(
            tok.pos_start, tok.pos_end,
            "Expected int, float, '+', '-', '('"
        ))
    
    def power(self):
        return self.bin_op(self.atom, (TT_POW, ), self.factor)

    
    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))
        
        return self.power()
    
    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV, TT_IDIV))

    def expr(self):
        res = ParseResult()

        if self.current_tok.matches(TT_KEYWORD, ("let", "const")):
            var_type = self.current_tok
            res.register(self.advance())
            print(self.current_tok)
            if self.current_tok.type != TT_IDENTIFIER:
                return res.failure(SyntaxError_(self.current_tok.pos_start, self.current_tok.pos_end, "Expected identifier", self.fn))
            var_name = self.current_tok
            res.register(self.advance())
            if self.current_tok.type != TT_EQ:
                return res.failure(SyntaxError_(self.current_tok.pos_start, self.current_tok.pos_end, "Expected equal", self.fn))
            res.register(self.advance())
            expr = self.expr()
            if res.error: return res
            return res.success(VarAssignNode(var_type, var_name, expr.node))
        
        elif self.current_tok.type == TT_IDENTIFIER:
            var_name = self.current_tok.value
            res.register(self.advance())
            op_tok = self.current_tok
            if not op_tok.type in (TT_PLUS, TT_MINUS, TT_MUL, TT_DIV, TT_POW, TT_IDIV, TT_INCREMENT, TT_DECREMENT):
                return res.failure(SyntaxError_(op_tok.pos_start, op_tok.pos_end, "Expected operator", self.fn))
            res.register(self.advance())
            expr = self.expr()
            if expr.error: return expr
            return res.success(VarChangeNode(var_name, op_tok, expr.node))
            

        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def bin_op(self, a_func, ops, b_func=None):
        if b_func == None:
            b_func = a_func

        res = ParseResult()
        left = res.register(a_func())
        if res.error: return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(b_func())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)

        return res.success(left)

# Values
class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()
    
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        self.context = context
        return self
    
    def added_by(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, ZeroDivisionError_(other.pos_start, other.pos_end, 'division by zero', self.context)
            return Number(self.value / other.value).set_context(self.context), None
    
    def powed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None

# RTResult
class RTResult:
    def __init__(self):
        self.value = None
        self.error = None
    
    def register(self, res):
        if res.error: self.error = res.error
        return res.value
    
    def success(self, value):
        self.value = value
        return self
    
    def failure(self, error):
        self.error = error
        return self

# Context
class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.parent = parent
        self.display_name = display_name
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None

# Symbol Table
class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None
    
    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value
    
    def set(self, name, value):
        self.symbols[name] = value
    
    def remove(self, name):
        del self.symbols[name]

# Interpreter
class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)
    
    def no_visit_method(self, node, context):
        raise Exception(f"No visit method defined for {type(node).__name__}")
    
    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.name.value
        variable = context.symbol_table.get(var_name)
        if not variable:
            return res.failure(RTNameError_(node.pos_start, node.pos_end, f"'{var_name}' is not defined", context))
        value = variable[1]

        return res.success(value)
    
    def visit_VarChangeNode(self, node, context):
        res = RTResult()
        var_name = node.name
        op_tok = node.op_tok
        value = node.node
        if (value_recent := context.symbol_table.get(var_name)) == None:
            return res.failure(RTNameError_(node.pos_start, node.pos_end, f"'{var_name}' is not defined", context))
        
        if value_recent[0] == "const":
            return res.failure(RTNameError_(node.pos_start, node.pos_end, f"'{var_name}' is a constant", context))
        
        if op_tok.type == TT_INCREMENT and isinstance(value_recent[1], Number):
            return res.register()
        else: return res.failure(TypeError_())

        if op_tok.type == TT_PLUS: pass

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_type = node.type
        var_name = node.name.value
        value = res.register(self.visit(node.value_node))
        if res.error: return res
        context.symbol_table.set(var_name, (var_type, value))

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: return res

        right = res.register(self.visit(node.right_node, context))
        if res.error: return res

        if node.op_tok.type == TT_PLUS: result, error = left.added_by(right)
        elif node.op_tok.type == TT_MINUS: result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL: result, error = left.multed_by(right)
        elif node.op_tok.type in (TT_DIV, TT_IDIV): result, error = left.dived_by(right)
        elif node.op_tok.type == TT_POW: result, error = left.powed_by(right)
        if error: return res.failure(error)

        if node.op_tok.type == TT_IDIV: result.value.value = int(result.value.value)
        return res.success(result.set_pos(node.pos_start, node.pos_end))
        
    
    def visit_NumberNode(self, node, context):
        return RTResult().success(Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res
        error = None
        if node.op_tok.type == TT_MINUS: number, error = number.multed_by(Number(-1))
        if error: return res.failure(error)
        return res.success(number.set_pos(node.pos_start, node.pos_end))

# Run
def readfile(path):
    with open(path) as f: data = f.read()
    return data, os.path.abspath(path)

def run(path):
    data, whole_path = readfile(path)
    lexer = Lexer(data, whole_path)
    error = lexer.tokenize()
    if error:
        print(error.as_string())
        quit(1)

    nodes = []
    for toks in split_list(lexer.tokens, TT_EOL, "type"):
        parser = Parser([*toks, Token(TT_EOL, pos_start=lexer.pos)], whole_path)
        ast = parser.parse()
        if ast.error:
            print(ast.error.as_string())
            quit(1)
        nodes.append(ast.node)
    
    context = Context(whole_path)
    for node in nodes:
        interpreter = Interpreter()
        result = interpreter.visit(node, context)
        if result.error:
            print(result.error.as_string())
            quit(1)
        print(result.value.value)

def shell():
    while True:
        text = input(">>> ")
        run(text)

global_symbol_table = SymbolTable()
global_symbol_table.set('null', ("const", Number(0)))

stats = {"version": "Pack 0.0"}
options = {"e": run}
if __name__ == '__main__':
    match len((argv := sys.argv)):
        case 1:
            shell()
        case 2:
            if argv[1].startswith('--'):
                try:
                    print(stats[argv[1][2:]])
                except:
                    print("Error")
            else:
                run(sys.argv[1])
            
        case other:
            if argv[1].startswith("-"):
                options[argv[1][1:]](*argv)
            else: print("Error")