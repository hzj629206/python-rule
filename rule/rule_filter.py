# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals, division, print_function

import six
import inspect


class Library(object):
    def __init__(self):
        self.filters = {}

    def filter(self, name=None, filter_func=None, **flags):
        if name is None and filter_func is None:
            # @register.filter()
            def dec(func):
                return self.filter_function(func, **flags)
            return dec
        elif name is not None and filter_func is None:
            if callable(name):
                # @register.filter
                return self.filter_function(name, **flags)
            else:
                # @register.filter('somename') or @register.filter(name='somename')
                def dec(func):
                    return self.filter(name, func, **flags)
                return dec
        elif name is not None and filter_func is not None:
            # register.filter('somename', somefunc)
            self.filters[name] = filter_func
            # handle flags here
            filter_func._filter_name = name
            return filter_func
        else:
            raise ValueError("Unsupported arguments to Library.filter: (%r, %r)" % (name, filter_func))

    def filter_function(self, func, **flags):
        name = getattr(func, "_decorated_function", func).__name__
        return self.filter(name, func, **flags)


register = Library()


def getargspec(func):
    if six.PY2:
        return inspect.getargspec(func)

    sig = inspect.signature(func)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    varkw = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    varkw = varkw[0] if varkw else None
    defaults = [
        p.default for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default is not p.empty
    ] or None
    return args, varargs, varkw, defaults


def args_check(name, func, provided):
    provided = list(provided)
    # First argument, filter input, is implied.
    plen = len(provided) + 1
    # Check to see if a decorator is providing the real function.
    func = getattr(func, '_decorated_function', func)

    args, _, _, defaults = getargspec(func)
    alen = len(args)
    dlen = len(defaults or [])
    # Not enough OR Too many
    if plen < (alen - dlen) or plen > alen:
        raise Exception("%s requires %d arguments, %d provided" % (name, alen - dlen, plen))


def find_filter(filter_name):
    if filter_name in register.filters:
        return register.filters[filter_name]
    else:
        raise Exception("Invalid filter: '%s'" % filter_name)


@register.filter
def echo(value):
    print("echo:", value)
    return value


@register.filter
def debug(value, arg):
    print("debug:", (value, arg))
    return value, arg
