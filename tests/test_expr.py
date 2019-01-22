# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import unittest

from rule.rule_parser import Expression


class TestExpr(unittest.TestCase):
    def test_expr_int(self):
        r = Expression("1").evaluate({})
        self.assertEqual(r, 1)

    def test_expr_string(self):
        r = Expression("\"1\"").evaluate({})
        self.assertEqual(r, "1")

    def test_expr_list(self):
        r = Expression("[\"1\", 2]").evaluate({})
        self.assertEqual(r, ["1", 2])

    def test_expr_op(self):
        r = Expression("[\"1\", 2] == [\"1\", 2]").evaluate({})
        self.assertTrue(r)

    def test_expr_var(self):
        r = Expression("$a > $b").evaluate({
            "a": 2,
            "b": 1,
        })
        self.assertTrue(r)

        r = Expression("$a < $b").evaluate({
            "a": 2,
            "b": 1,
        })
        self.assertFalse(r)

        r = Expression("$a < C").evaluate({
            "a": 2,
            "b": 1,
            "C": 1,
        })
        self.assertFalse(r)

    def test_expr_neg(self):
        expr = Expression("5 - -1")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 6)

        expr = Expression("5 - --1")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 4)

        expr = Expression("5 - +1")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 4)

        expr = Expression("5 - ++1")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 4)

    def test_expr_not(self):
        expr = Expression("not 1")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, False)

        expr = Expression("not 0")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, True)

    def test_expr_minus(self):
        expr = Expression("1 - 1")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 0)

    def test_expr_and(self):
        expr = Expression("1 and 2")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 2)

        expr = Expression("1 and 2 and 3 and 4 and 5")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 5)

        expr = Expression("1 and 2 and 3 and 4 and 5 or 6 and 7 and 8 and 9")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 5)

    def test_expr_or(self):
        expr = Expression("1 or 2")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 1)

        expr = Expression("1 or 2 or 3 or 4 or 5")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 1)

        expr = Expression("1 and 1 or 2 and 2 or 3 and 3 or 4 and 4 or 5 and 5")
        print(list(expr.ast))
        r = expr.evaluate({})
        self.assertEqual(r, 1)

    def test_expr_hash(self):
        expr = Expression("1 or 2")
        self.assertEquals(expr.ast.hash, "1 or 2")

        expr = Expression("1 or 2 or 3 or 4 or 5")
        self.assertEquals(expr.ast.hash, "1 or 2 or 3 or 4 or 5")

        expr = Expression("1 and 1 or 2 and 2 or 3 and 3 or 4 and 4 or 5 and 5")
        self.assertEquals(expr.ast.hash, "1 and 1 or 2 and 2 or 3 and 3 or 4 and 4 or 5 and 5")

        expr = Expression("1 + 1 - 2 > 0")
        self.assertEquals(expr.ast.hash, "1+1+-2>0")

        expr = Expression("$a = 1 + 1 - 2 > 0 ? 111 : 222 or 1")
        self.assertEquals(expr.ast.hash, "$a=1+1+-2>0?111:222 or 1")

        expr = Expression("1 + 1 - (1 + 1)")
        self.assertEquals(expr.ast.hash, "1+1-(1+1)")

        expr = Expression("[1,2,3,4,5,6,7,8,9,10]")
        self.assertEquals(expr.ast.hash, "[1,2,3,4,5,...a3cb039]")

        expr = Expression("${ a | echo }")
        self.assertEquals(expr.ast.hash, "${a | echo}")

    def test_expr_symbol(self):
        expr = Expression("ABC")
        self.assertEquals(expr.evaluate({}), "ABC")

        expr = Expression("ABC", {
            "ABC": {
                "type": "int",
                "value": "1",
            }
        })
        self.assertEquals(expr.evaluate({}), 1)

        expr = Expression("ABC", {
            "ABC": {
                "type": "list[int]",
                "value": ["1", "2", "3"],
            }
        })
        self.assertEquals(expr.evaluate({}), [1, 2, 3])

    def test_expr_assignment(self):
        ctx = {}

        expr = Expression("$a = 1")
        self.assertTrue(expr.evaluate(ctx))
        self.assertEquals(ctx["a"], 1)

        expr = Expression("($a > 1)")
        self.assertFalse(expr.evaluate(ctx))
        self.assertEquals(ctx["a"], 1)

        expr = Expression("$a = 1, $b = 1 + 1, $a == 1 and $b")
        self.assertEquals(expr.evaluate(ctx), 2)
        self.assertEquals(ctx["a"], 1)
        self.assertEquals(ctx["b"], 2)

    def test_expr_bind(self):
        ctx = {}

        expr = Expression("$a : $b")
        self.assertTrue(expr.evaluate(ctx))
        self.assertEquals(ctx, {})

    def test_expr_attr(self):
        ctx = {"a": {}}

        expr = Expression("$a.__dict__")
        self.assertFalse(expr.evaluate(ctx))
        self.assertEquals(ctx["a"], {})

    def test_expr_colon(self):
        class FooClass(object):
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        ctx = {"a": 1, "b": 2, "c": 3, "d": FooClass(x=1)}

        expr = Expression("$h = $a ? $b : 3, $x:$a, $i=3, $d and $d.__dict__")
        self.assertEquals(expr.evaluate(ctx), {"x": 1})
        self.assertEquals(ctx["h"], 2)
        self.assertEquals(ctx["i"], 3)

    def test_expr_filter(self):
        ctx = {"a": 1, "c": 2}

        expr = Expression('${a|echo}')
        self.assertEquals(expr.evaluate(ctx), 1)

        expr = Expression('${a|debug:c}')
        self.assertEquals(expr.evaluate(ctx), (1, 2))

        expr = Expression(r'${a|debug:"x\nyz"}')
        self.assertEquals(expr.evaluate(ctx), (1, "x\nyz"))

        expr = Expression('${ a | echo }')
        self.assertEquals(expr.evaluate(ctx), 1)

        expr = Expression('${ a | debug : c }')
        self.assertEquals(expr.evaluate(ctx), (1, 2))

        expr = Expression(r'${ a | debug : "x\nyz" }')
        self.assertEquals(expr.evaluate(ctx), (1, "x\nyz"))

        expr = Expression('${ a | echo | echo }')
        self.assertEquals(expr.evaluate(ctx), 1)

        expr = Expression('${ a | debug : c | debug : "xxx" | echo }')
        self.assertEquals(expr.evaluate(ctx), ((1, 2), "xxx"))

        expr = Expression(r'${ a | debug : "x\nyz" | debug : "yyy" | echo }')
        self.assertEquals(expr.evaluate(ctx), ((1, "x\nyz"), "yyy"))
