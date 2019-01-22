# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals, division, print_function

import re
import hashlib
import ply.lex as lex
import ply.yacc as yacc

tokens = (
    'VARIABLE', 'FILTER', 'SYMBOL', 'FLOAT', 'INT', 'STRING',
    'ADD', 'SUB', 'MUL', 'FLOORDIV', 'DIV', 'MOD', 'NOT', 'AND', 'OR',
    'EQ', 'NE', 'LE', 'GE', 'LT', 'GT', 'IN', 'NI',
    'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET',
    'CONDITION', 'COLON', 'COMMA', 'EQUALS', 'PERIOD',
)

# Tokens

# order by declaration order for t_* functions
# order by length of pattern desc for t_* variables

t_ADD = r'\+'
t_SUB = r'-'
t_MUL = r'\*'
t_FLOORDIV = r'//'
t_DIV = r'/'
t_MOD = r'%'
t_CONDITION = r'\?'
t_COLON = r':'
t_EQ = r'=='
t_NE = r'!='
t_LE = r'<='
t_GE = r'>='
t_LT = r'<'
t_GT = r'>'
t_EQUALS = r'='

t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'

t_COMMA = r','
t_PERIOD = r'\.'

reserved = {
    "not": "NOT",
    "and": "AND",
    "or": "OR",
    "in": "IN",
}


# Read in a float. This rule has to be done before the int rule.
def t_FLOAT(t):
    r'\d+\.\d*(e-?\d+)?'
    # t.value = float(t.value)
    return t


# Read in an int.
def t_INT(t):
    r'\d+'
    # t.value = int(t.value)
    return t


# Read in a string, as in C. The following backslash sequences have their
# usual special meaning: \", \\, \n, and \t.
def t_STRING(t):
    r'\"([^\\"]|(\\.))*\"'
    t.value = t.value[1:-1]
    return t


def t_NI(t):
    r'not\ +in'
    t.value = "not in"
    return t


# Read in a variable.
def t_VARIABLE(t):
    r'\$[a-zA-Z_][a-zA-Z0-9_]*'
    t.value = t.value[1:]
    return t


# Read in a filter expression
def t_FILTER(t):
    r'\$\{\s*(?P<var>[a-zA-Z_][a-zA-Z0-9_]*)(\s*\|\s*(?P<filter_name>[a-zA-Z_][a-zA-Z0-9_]*)(\s*\:\s*((?P<constant_arg>\"([^\\"]|(\\.))*\")|(?P<var_arg>[a-zA-Z_][a-zA-Z0-9_]*)))?)+\s*\}'
    t.value = t.value[2:-1]
    return t


filter_raw = r"""
^(?P<var>[a-zA-Z_][a-zA-Z0-9_]*)|
(?:\s*\|\s*
    (?P<filter_name>[a-zA-Z_][a-zA-Z0-9_]*)
        (?:\s*\:\s*
            (?:
                (?P<constant_arg>\"([^\\"]|(\\.))*\")|
                (?P<var_arg>[a-zA-Z_][a-zA-Z0-9_]*)
            )
        )?
)"""
filter_pattern = re.compile(filter_raw, re.UNICODE | re.VERBOSE)


def unescape_literal(str_val):
    escaped = 0
    new_str = ""
    for i in range(0, len(str_val)):
        c = str_val[i]
        if escaped:
            if c == "n":
                c = "\n"
            elif c == "t":
                c = "\t"
            new_str += c
            escaped = 0
        else:
            if c == "\\":
                escaped = 1
            else:
                new_str += c
    return new_str


def parse_filter(token):
    from .rule_filter import args_check, find_filter

    matches = filter_pattern.finditer(token)
    var_node = None
    filters = []
    upto = 0
    for match in matches:
        start = match.start()
        if upto != start:
            raise Exception("Could not parse some characters: %s|%s|%s" % (
                token[:upto], token[upto:start], token[start:]
            ))
        if var_node is None:
            var_name = match.group(b"var")
            if var_name is None:
                raise Exception("Could not find variable at start of %s." % token)
            else:
                var_node = VariableNode(var_name)
        else:
            filter_name = match.group(b"filter_name")
            args = []
            constant_arg, var_arg = match.group(b"constant_arg", b"var_arg")
            if constant_arg:
                args.append(StringNode(unescape_literal(constant_arg[1:-1])))
            elif var_arg:
                args.append(VariableNode(var_arg))
            filter_func = find_filter(filter_name)
            args_check(filter_name, filter_func, args)
            filters.append((filter_func, args))
        upto = match.end()
    if upto != len(token):
        raise Exception("Could not parse the remainder: '%s' from '%s'" % (token[upto:], token))

    return token, var_node, filters


def tokenize_filter(token):
    matches = filter_pattern.finditer(token)
    var_node = None
    filters = []
    upto = 0
    for match in matches:
        start = match.start()
        if upto != start:
            raise Exception("Could not parse some characters: %s|%s|%s" % (
                token[:upto], token[upto:start], token[start:]
            ))
        if var_node is None:
            var_name = match.group(b"var")
            if var_name is None:
                raise Exception("Could not find variable at start of %s." % token)
            else:
                var_node = var_name
        else:
            filter_name = match.group(b"filter_name")
            args = []
            constant_arg, var_arg = match.group(b"constant_arg", b"var_arg")
            if constant_arg:
                args.append(constant_arg)
            elif var_arg:
                args.append(var_arg)
            filters.append((filter_name, args))
        upto = match.end()
    if upto != len(token):
        raise Exception("Could not parse the remainder: '%s' from '%s'" % (token[upto:], token))

    return [var_node] + filters


# Ignored characters
t_ignore = " \t"


# Ignore comments
def t_comment(t):
    r'[#][^\n]*'
    pass


# Track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")


# Read in a symbol.
# This rule must be practically last
# since there are so few rules concerning what constitutes a symbol
# r'[^0-9()][^()\ \t\n]*'
def t_SYMBOL(t):
    r'[a-zA-Z_][^()\ \t\n]*'
    t.type = reserved.get(t.value, "SYMBOL")
    return t


def resolve_symbol(field):
    val = field.get("value")
    typ = field.get("type") or type(val).__name__

    if typ == "list":
        if val:
            typ = "list[%s]" % type(val[0]).__name__
        else:
            typ = "list[str]"

    coerce_mapping = {
        "str": (StringNode, str),
        "int": (IntegerNode, int),
        "float": (FloatNode, float),
    }

    # str|int|float
    if typ in coerce_mapping:
        _node, _coerce = coerce_mapping[typ]
        return _node(_coerce(val))

    # list[str|int|float]
    m = re.match(r"^list\[(str|int|float)\]$", typ)
    if m and m.group(1) in coerce_mapping:
        _node, _coerce = coerce_mapping[m.group(1)]
        val = [_coerce(_val) for _val in list(val)]
        val = [_node(_val) for _val in sorted(val)]
        return ListNode(val)

    # unknown type
    return ValueNode(val, typ)


# Handle errors
def t_error(t):
    # print("Illegal character '%s'" % t.value[0])
    # t.lexer.skip(1)
    raise SyntaxError("Illegal character '%s'" % t.value[0])


# Build the lexer
lexer = lex.lex()

# Precedence rules for the arithmetic operators
# see: https://zh.cppreference.com/w/c/language/operator_precedence
precedence = (
    ('right', 'COMMA'),
    ('nonassoc', 'EQUALS'),
    ('nonassoc', 'FILTER'),
    ('right', 'CONDITION', 'COLON'),
    ('left', 'OR'),
    ('left', 'AND'),
    ('nonassoc', 'EQ', 'NE'),
    ('nonassoc', 'LE', 'LT', 'GE', 'GT', 'IN', 'NI'),
    ('left', 'ADD', 'SUB'),
    ('left', 'MUL', 'FLOORDIV', 'DIV', 'MOD'),
    ('right', 'NEG', 'POS', 'NOT'),  # Unary minus operator: NEG, POS
    ('left', 'PERIOD'),
)


def p_statement_expr(p):
    """statement : expression_list
                 | empty
    """
    p[0] = p[1]


def p_expression_list_1(p):
    """expression_list : logical_expression"""
    p[0] = p[1]


def p_expression_list_2(p):
    """expression_list : expression_decl_list"""
    p[0] = p[1]


def p_expression_list_3(p):
    """expression_list : expression_decl_list COMMA logical_expression"""
    p[0] = OperatorAnd(p[1], p[3])


def p_expression_decl_list_1(p):
    """expression_decl_list : expression_decl"""
    p[0] = p[1]


def p_expression_decl_list_2(p):
    """expression_decl_list : expression_decl_list COMMA expression_decl"""
    p[0] = OperatorAnd(p[1], p[3])


def p_expression_decl_1(p):
    """expression_decl : assign_expression"""
    p[0] = p[1]


def p_expression_decl_2(p):
    """expression_decl : bind_expression"""
    p[0] = p[1]


def p_statement_assign(p):
    """assign_expression : VARIABLE EQUALS conditional_expression"""
    p[0] = OperatorAssign(p[1], p[3])


def p_statement_bind(p):
    """bind_expression : VARIABLE COLON VARIABLE"""
    p[0] = OperatorBind(p[1], p[3])


def p_expression_conditional_1(p):
    """conditional_expression : logical_expression"""
    p[0] = p[1]


def p_expression_conditional_2(p):
    """conditional_expression : logical_expression CONDITION logical_expression COLON conditional_expression"""
    p[0] = OperatorCondition(p[1], p[3], p[5])


def p_expression_binop(p):
    """logical_expression : logical_expression OR logical_expression
                  | logical_expression AND logical_expression
                  | logical_expression EQ logical_expression
                  | logical_expression NE logical_expression
                  | logical_expression LE logical_expression
                  | logical_expression LT logical_expression
                  | logical_expression GE logical_expression
                  | logical_expression GT logical_expression
                  | logical_expression IN logical_expression
                  | logical_expression NI logical_expression
                  | logical_expression ADD logical_expression
                  | logical_expression SUB logical_expression
                  | logical_expression MUL logical_expression
                  | logical_expression FLOORDIV logical_expression
                  | logical_expression DIV logical_expression
                  | logical_expression MOD logical_expression
    """
    if p[2] == '+':
        p[0] = OperatorAdd(p[1], p[3])
    elif p[2] == '-':
        if isinstance(p[3], (IntegerNode, FloatNode)):
            p[3].leaf = -p[3].leaf
            p[0] = OperatorAdd(p[1], p[3])
        else:
            p[0] = OperatorSub(p[1], p[3])
    elif p[2] == '*':
        p[0] = OperatorMul(p[1], p[3])
    elif p[2] == '/':
        p[0] = OperatorDiv(p[1], p[3])
    elif p[2] == '//':
        p[0] = OperatorFloorDiv(p[1], p[3])
    elif p[2] == '%':
        p[0] = OperatorMod(p[1], p[3])
    elif p[2] == '>':
        p[0] = OperatorGt(p[1], p[3])
    elif p[2] == '>=':
        p[0] = OperatorGe(p[1], p[3])
    elif p[2] == '<':
        p[0] = OperatorLt(p[1], p[3])
    elif p[2] == '<=':
        p[0] = OperatorLe(p[1], p[3])
    elif p[2] == '==':
        p[0] = OperatorEq(p[1], p[3])
    elif p[2] == '!=':
        p[0] = OperatorNe(p[1], p[3])
    elif p[2] == 'in':
        p[0] = OperatorIn(p[1], p[3])
    elif p[2] == 'not in':
        p[0] = OperatorNi(p[1], p[3])
    elif p[2] == 'or':
        p[0] = OperatorOr(p[1], p[3])
    elif p[2] == 'and':
        p[0] = OperatorAnd(p[1], p[3])


def p_statement_attribute(p):
    """logical_expression : logical_expression PERIOD SYMBOL"""
    p[0] = OperatorAttribute(p[1], p[3])


def p_expression_not(p):
    """logical_expression : NOT logical_expression"""
    if isinstance(p[2], ValueNode):
        p[0] = ValueNode(not p[2].leaf)
    else:
        p[0] = OperatorNot(p[2], p[1])


def p_expression_neg(p):
    """logical_expression : SUB logical_expression %prec NEG"""
    if isinstance(p[2], (IntegerNode, FloatNode)):
        p[2].leaf = -p[2].leaf
        p[0] = p[2]
    else:
        p[0] = OperatorNeg(p[2], p[1])


def p_expression_pos(p):
    """logical_expression : ADD logical_expression %prec POS"""
    if isinstance(p[2], (IntegerNode, FloatNode)):
        p[2].leaf = +p[2].leaf
        p[0] = p[2]
    else:
        p[0] = OperatorPos(p[2], p[1])


def p_expression_group(p):
    """logical_expression : LPAREN logical_expression RPAREN"""
    p[0] = ParenNode(p[2])


def p_expression_list(p):
    """logical_expression : LBRACKET logical_comma_expression RBRACKET
                  | LBRACKET empty RBRACKET
    """
    p[0] = ListNode(p[2] or [])


def p_expression_comma_1(p):
    """logical_comma_expression : logical_expression"""
    p[0] = [p[1]]


def p_expression_comma_2(p):
    """logical_comma_expression : logical_comma_expression COMMA logical_expression"""
    p[0] = p[1] + [p[3]]


def p_expression_float(p):
    """logical_expression : FLOAT"""
    p[0] = FloatNode(float(p[1]))


def p_expression_int(p):
    """logical_expression : INT"""
    p[0] = IntegerNode(int(p[1]))


def p_expression_string(p):
    """logical_expression : STRING"""
    p[0] = StringNode(unescape_literal(p[1]))


def p_expression_variable(p):
    """logical_expression : VARIABLE"""
    p[0] = VariableNode(p[1])


def p_expression_filter(p):
    """logical_expression : FILTER"""
    p[0] = FilterNode(*parse_filter(p[1].strip()))


def p_expression_symbol(p):
    """logical_expression : SYMBOL"""
    fields = getattr(p.parser, "fields", None) or {}
    if p[1] in fields:
        p[0] = resolve_symbol(fields[p[1]])
    else:
        p[0] = SymbolNode(p[1])


def p_expression_empty(p):
    """empty :"""
    pass


# Error rule for syntax errors.
def p_error(p):
    if p:
        # print("Syntax error at '%s'" % p.value)
        raise SyntaxError("Syntax error at '%s'" % p.value)
    else:
        # print("Syntax error. Unexpected EOF")
        raise SyntaxError("Syntax error. Unexpected EOF")


parser = yacc.yacc(start="statement")


class Node(object):
    def __init__(self, nodetype, children=None, leaf=None):
        self.nodetype = nodetype
        self.children = children or []
        self.leaf = leaf

    def evaluate(self, context):
        raise NotImplementedError

    @property
    def hash(self):
        return str(self.leaf)

    def __str__(self):
        return self.hash

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.hash)

    def __iter__(self):
        return self.pre_order()

    def pre_order(self):
        s = [self]

        while s:
            r = s.pop()
            yield r
            for child in reversed(r.children):
                s.append(child)

    def post_order(self):
        s = [self]
        v = []

        while s:
            r = s.pop()
            v.append(r)
            for child in r.children:
                s.append(child)
        for node in reversed(v):
            yield node

    def get_nodes_by_type(self, nodetype):
        """
        :param T <- ExpressionNode nodetype:
        :rtype: list[Node]
        """
        nodes = []
        for node in self:
            if isinstance(node, nodetype):
                nodes.append(node)
        return nodes


class ParenNode(Node):
    def __init__(self, node):
        super(ParenNode, self).__init__("paren", [node], "(%s)")
        self.node = node

    def evaluate(self, context):
        return self.node.evaluate(context)

    @property
    def hash(self):
        return self.leaf % self.node.hash


class ValueNode(Node):
    def __init__(self, val, nodetype="val"):
        super(ValueNode, self).__init__(nodetype, [], val)

    @property
    def value(self):
        return self.leaf

    def evaluate(self, context):
        return self.leaf


class IntegerNode(ValueNode):
    def __init__(self, val):
        super(IntegerNode, self).__init__(val, "int")


class FloatNode(ValueNode):
    def __init__(self, val):
        super(FloatNode, self).__init__(val, "float")


class StringNode(ValueNode):
    def __init__(self, val):
        super(StringNode, self).__init__(val, "string")


class ListNode(ValueNode):
    def __init__(self, val):
        super(ListNode, self).__init__(val, "list")

    def evaluate(self, context):
        return [ele.evaluate(context) for ele in self.value]

    @property
    def hash(self):
        truck_size = 5
        if len(self.value) > truck_size:
            left = ",".join([item.hash for item in self.value[truck_size:]])
            left_hash = hashlib.md5(str(left)).hexdigest()
            return "[%s,...%s]" % (",".join([item.hash for item in self.value[:truck_size]]), left_hash[:7])
        else:
            return "[%s]" % ",".join([item.hash for item in self.value])


class VariableNode(ValueNode):
    def __init__(self, val):
        super(VariableNode, self).__init__(val, "var")

    def evaluate(self, context):
        val = context[self.leaf]
        if callable(val):
            val = val(context)
        return val

    @property
    def hash(self):
        return "$%s" % self.leaf


class SymbolNode(ValueNode):
    def __init__(self, val):
        super(SymbolNode, self).__init__(val, "sym")

    def evaluate(self, context):
        return context.get(self.leaf, self.leaf)


class OperatorNode(Node):
    def __init__(self, nodetype, children, leaf):
        super(OperatorNode, self).__init__(nodetype, children, leaf)

    @property
    def operator(self):
        return self.leaf

    def evaluate(self, context):
        raise NotImplementedError


class FilterNode(OperatorNode):
    def __init__(self, token, var, filters):
        super(FilterNode, self).__init__("filter", [var], "|")
        self.token = token
        self.var = var
        self.filters = filters

    def evaluate(self, context):
        val = self.var.evaluate(context)
        if callable(val):
            val = val(context)

        for func, args in self.filters:
            arg_vals = []
            for arg in args:
                if isinstance(arg, Node):
                    arg_vals.append(arg.evaluate(context))
                else:
                    arg_vals.append(arg)
            if callable(func):
                val = func(val, *arg_vals)

        return val

    @property
    def hash(self):
        return "${%s}" % self.token


class UnaryOperator(OperatorNode):
    def __init__(self, var, leaf):
        super(UnaryOperator, self).__init__("unaryop", [var], leaf)

        self.var = var

    def evaluate(self, context):
        raise NotImplementedError

    @property
    def hash(self):
        return "%s%s" % (self.leaf, self.var.hash)


class BinaryOperator(OperatorNode):
    def __init__(self, l_var, r_var, leaf):
        super(BinaryOperator, self).__init__("binop", [l_var, r_var], leaf)

        self.l_var = l_var
        self.r_var = r_var

    def evaluate(self, context):
        raise NotImplementedError

    @property
    def hash(self):
        return "%s%s%s" % (self.l_var.hash, self.leaf, self.r_var.hash)


class OperatorAdd(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="+"):
        super(OperatorAdd, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) + self.r_var.evaluate(context)


class OperatorSub(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="-"):
        super(OperatorSub, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) - self.r_var.evaluate(context)


class OperatorMul(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="*"):
        super(OperatorMul, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) * self.r_var.evaluate(context)


class OperatorDiv(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="/"):
        super(OperatorDiv, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) / self.r_var.evaluate(context)


class OperatorFloorDiv(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="//"):
        super(OperatorFloorDiv, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) // self.r_var.evaluate(context)


class OperatorMod(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="%"):
        super(OperatorMod, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) % self.r_var.evaluate(context)


class OperatorNeg(UnaryOperator):
    def __init__(self, var, leaf="-"):
        super(OperatorNeg, self).__init__(var, leaf)

    def evaluate(self, context):
        return -self.var.evaluate(context)


class OperatorPos(UnaryOperator):
    def __init__(self, var, leaf="+"):
        super(OperatorPos, self).__init__(var, leaf)

    def evaluate(self, context):
        return +self.var.evaluate(context)


class OperatorNot(UnaryOperator):
    def __init__(self, var, leaf="not"):
        super(OperatorNot, self).__init__(var, leaf)

    def evaluate(self, context):
        return not self.var.evaluate(context)

    @property
    def hash(self):
        return "%s %s" % (self.leaf, self.var.hash)


class OperatorAnd(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="and"):
        super(OperatorAnd, self).__init__(l_var, r_var, leaf)

        expanded = []
        if isinstance(l_var, OperatorAnd):
            expanded.extend(l_var.children)
        else:
            expanded.append(l_var)
        if isinstance(r_var, OperatorAnd):
            expanded.extend(r_var.children)
        else:
            expanded.append(r_var)
        self.children = expanded

    def evaluate(self, context):
        # result = self.l_var.evaluate(context) and self.r_var.evaluate(context)
        result = True
        for i in range(len(self.children)):
            result = result and self.children[i].evaluate(context)
            if not result:
                break
        return result

    @property
    def hash(self):
        h = []
        for child in self.children:
            h.append("%s" % child.hash)
        return (" %s " % self.leaf).join(h)


class OperatorOr(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="or"):
        super(OperatorOr, self).__init__(l_var, r_var, leaf)

        expanded = []
        if isinstance(l_var, OperatorOr):
            expanded.extend(l_var.children)
        else:
            expanded.append(l_var)
        if isinstance(r_var, OperatorOr):
            expanded.extend(r_var.children)
        else:
            expanded.append(r_var)
        self.children = expanded

    def evaluate(self, context):
        # result = self.l_var.evaluate(context) or self.r_var.evaluate(context)
        result = False
        for i in range(len(self.children)):
            result = result or self.children[i].evaluate(context)
            if result:
                break
        return result

    @property
    def hash(self):
        h = []
        for child in self.children:
            h.append("%s" % child.hash)
        return (" %s " % self.leaf).join(h)


class OperatorEq(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="=="):
        super(OperatorEq, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) == self.r_var.evaluate(context)


class OperatorNe(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="!="):
        super(OperatorNe, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) != self.r_var.evaluate(context)


class OperatorLt(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="<"):
        super(OperatorLt, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) < self.r_var.evaluate(context)


class OperatorGt(BinaryOperator):
    def __init__(self, l_var, r_var, leaf=">"):
        super(OperatorGt, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) > self.r_var.evaluate(context)


class OperatorLe(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="<="):
        super(OperatorLe, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) <= self.r_var.evaluate(context)


class OperatorGe(BinaryOperator):
    def __init__(self, l_var, r_var, leaf=">="):
        super(OperatorGe, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) >= self.r_var.evaluate(context)


class OperatorIn(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="in"):
        super(OperatorIn, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) in self.r_var.evaluate(context)


class OperatorNi(BinaryOperator):
    def __init__(self, l_var, r_var, leaf="not in"):
        super(OperatorNi, self).__init__(l_var, r_var, leaf)

    def evaluate(self, context):
        return self.l_var.evaluate(context) not in self.r_var.evaluate(context)


class OperatorCondition(OperatorNode):
    def __init__(self, cond_var, true_var, false_var):
        super(OperatorCondition, self).__init__("condop", [cond_var, true_var, false_var], "?:")

        self.cond_var = cond_var
        self.true_var = true_var
        self.false_var = false_var

    def evaluate(self, context):
        return self.true_var.evaluate(context) if self.cond_var.evaluate(context) else self.false_var.evaluate(context)

    @property
    def hash(self):
        return "%s?%s:%s" % (self.cond_var.hash, self.true_var.hash, self.false_var.hash)


class OperatorAssign(OperatorNode):
    def __init__(self, name, expr):
        super(OperatorAssign, self).__init__("assign", [expr], "=")
        self.name = name
        self.expr = expr

    def evaluate(self, context):
        context[self.name] = self.expr.evaluate(context)
        return True

    @property
    def hash(self):
        return "$%s=%s" % (self.name, self.expr.hash)


class OperatorBind(OperatorNode):
    def __init__(self, name, attr):
        super(OperatorBind, self).__init__("bind", [], ":")
        self.name = name
        self.attr = attr

    def evaluate(self, context):
        return True

    @property
    def hash(self):
        return "$%s:$%s" % (self.name, self.attr)


class OperatorAttribute(OperatorNode):
    def __init__(self, obj, attr):
        super(OperatorAttribute, self).__init__("attr", [obj], ".")
        self.obj = obj
        self.attr = attr

    def evaluate(self, context):
        obj = self.obj.evaluate(context)
        if hasattr(obj, self.attr):
            return getattr(obj, self.attr)
        try:
            if self.attr in obj:
                return obj[self.attr]
        except TypeError:
            pass
        return None

    @property
    def hash(self):
        return "%s.%s" % (self.obj.hash, self.attr)


class Expression(object):
    def __init__(self, expr=None, fields=None):
        self.expr = expr or ""
        self.fields = fields or {}

        self.ast = self.compile_ast()

    def compile_ast(self):
        setattr(parser, "fields", self.fields)
        try:
            root = parser.parse(self.expr, lexer=lexer)
        finally:
            delattr(parser, "fields")
        if root and isinstance(root, ParenNode):
            root = root.node
        return root

    def evaluate(self, context):
        if not self.ast:
            return None

        return self.ast.evaluate(context)

    def interpret(self, context):
        if not self.ast:
            return None

        pre_order_node_list = list(self.ast.pre_order())
        stack = []
        while pre_order_node_list:
            node = pre_order_node_list.pop()
            if isinstance(node, ParenNode):
                pass
            elif isinstance(node, OperatorNode):
                if isinstance(node, OperatorBind):
                    stack.append(True)
                elif isinstance(node, OperatorAssign):
                    n1 = stack.pop()
                    context[n1.name] = n1
                    stack.append(True)
                elif isinstance(node, OperatorCondition):
                    n1, n2, n3 = stack.pop(), stack.pop(), stack.pop()
                    stack.append(n2 if n1 else n3)
                elif isinstance(node, OperatorAnd):
                    result = True
                    for i in range(len(node.children)):
                        ele = stack.pop()
                        result = result and ele
                    stack.append(result)
                elif isinstance(node, OperatorOr):
                    result = False
                    for i in range(len(node.children)):
                        ele = stack.pop()
                        result = result or ele
                    stack.append(result)
                elif isinstance(node, OperatorAttribute):
                    obj = stack.pop()
                    try:
                        if hasattr(obj, node.attr):
                            result = getattr(obj, node.attr)
                        elif node.attr in obj:
                            result = obj[node.attr]
                        else:
                            result = None
                    except TypeError:
                        result = None
                    stack.append(result)
                elif isinstance(node, BinaryOperator):
                    operator = node.leaf
                    n1, n2 = stack.pop(), stack.pop()
                    if operator == "+":
                        stack.append(n1 + n2)
                    elif operator == "-":
                        stack.append(n1 - n2)
                    elif operator == "*":
                        stack.append(n1 * n2)
                    elif operator == "/":
                        stack.append(n1 / n2)
                    elif operator == "//":
                        stack.append(n1 // n2)
                    elif operator == "%":
                        stack.append(n1 % n2)
                    elif operator == ">=":
                        stack.append(n1 >= n2)
                    elif operator == ">":
                        stack.append(n1 > n2)
                    elif operator == "<=":
                        stack.append(n1 <= n2)
                    elif operator == "<":
                        stack.append(n1 < n2)
                    elif operator == "==":
                        stack.append(n1 == n2)
                    elif operator == "!=":
                        stack.append(n1 != n2)
                    elif operator == "in":
                        stack.append(n1 in n2)
                    elif operator == "not in":
                        stack.append(n1 not in n2)
                    else:
                        raise NotImplementedError
                elif isinstance(node, UnaryOperator):
                    operator = node.leaf
                    n1 = stack.pop()
                    if operator == "+":
                        stack.append(+n1)
                    elif operator == "-":
                        stack.append(-n1)
                    elif operator == "not":
                        stack.append(not n1)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                stack.append(node.evaluate(context))
        return stack[-1]  # top element in stack

    def get_nodes_by_type(self, nodetype):
        """
        :param T <- ExpressionNode nodetype:
        :rtype: list[Node]
        """
        if not self.ast:
            return []

        return self.ast.get_nodes_by_type(nodetype)
