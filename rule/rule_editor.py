# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals, division, print_function

from .rule_parser import lexer, tokenize_filter


def get_symbol(token, fields):
	fields = fields or {}
	field = fields.get(token) or {}
	display = field.get("name") or token
	description = field.get("desc") or token
	typ = field.get("type")
	val = field.get("value")

	if val is None and typ:
		if typ == "int":
			val = 0
		elif typ == "str":
			val = ""
		elif typ == "float":
			val = 0.0
		elif typ.startswith("list"):
			val = []
	if typ is None and val:
		typ = type(val).__name__

	return {
		"token": token,
		"type": typ,
		"value": val,
		"display": display,
		"title": description,
	}


def update_symbol(token, typ, val, fields):
	fields = fields or {}
	field = fields.setdefault(token, {})
	display = field.get("name") or token
	description = field.get("desc") or token

	if val is None and typ:
		if typ == "int":
			val = 0
		elif typ == "str":
			val = ""
		elif typ == "float":
			val = 0.0
		elif typ.startswith("list"):
			val = []
	if typ is None and val:
		typ = type(val).__name__
	field.update({
		"type": typ,
		"value": val,
	})

	return {
		"token": token,
		"type": typ,
		"value": val,
		"display": display,
		"title": description,
	}


def format_expression(expr, fields=None):
	fields = fields or {}
	lexer.input(expr)
	tokens = list(lexer)

	for token in reversed(tokens):
		t, v, r, c = token.type, token.value, token.lineno - 1, token.lexpos
		if t == 'VARIABLE':
			field = fields.get(v) or {}
			display = field.get("name") or ("$%s" % v)
			description = field.get("desc") or display
			expr = expr[:c] + '<span class="token-var" title="%s">%s</span>' % (description, display) + expr[c + len(v) + 1:]
		elif t == 'SYMBOL':
			field = fields.get(v) or {}
			display = field.get("name") or v
			description = field.get("desc") or display
			expr = expr[:c] + '<button type="button" class="token-symbol" title="%s" data-token="%s">%s</button>' % (description, v, display) + expr[c + len(v):]
		elif t == 'FILTER':
			f, pos = "", 0
			for tok in tokenize_filter(v.strip()):
				if isinstance(tok, tuple):  # filter
					filter_name, filter_args = tok
					i = v.index(filter_name, pos)
					f += v[pos:i] + '<span class="token-func">%s<span>' % filter_name
					pos = i + len(filter_name)

					if filter_args:
						i = v.index(filter_args[0], pos)
						if filter_args[0][0] == '"':  # constant
							f += v[pos:i] + '%s' % filter_args[0]
						else:  # var
							field = fields.get(filter_args[0]) or {}
							display = field.get("name") or filter_args[0]
							description = field.get("desc") or display
							f += v[pos:i] + '<span class="token-var" title="%s">%s</span>' % (description, display)
						pos = i + len(filter_args[0])
				else:  # var
					var_name = tok  # type: str
					field = fields.get(var_name) or {}
					display = field.get("name") or var_name
					description = field.get("desc") or display
					i = v.index(var_name, pos)
					f += v[pos:i] + '<span class="token-var" title="%s">%s</span>' % (description, display)
					pos = i + len(var_name)
			f += v[pos:]
			expr = expr[:c] + '<span class="token-filter">${%s}</span>' % f + expr[c + len(v) + 3:]
	expr = '<span class="token">%s</span>' % expr
	return expr
