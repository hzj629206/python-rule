# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import unittest

from rule.rule_parser import Expression
from rule.rule_rete import *


class TestRete(unittest.TestCase):
	def test_helper(self):
		lhs = "$a > 1 and $b < 100"
		ast = Expression(lhs).ast
		self.assertEquals(ast.hash, '$a>1 and $b<100')

		lhs_neg = "not ($a > 1 and $b < 100)"
		ast_neg = Expression(lhs_neg).ast
		self.assertEquals(ast_neg.hash, 'not ($a>1 and $b<100)')

		beta_root = BetaNode()

		b1 = BindNode([], beta_root, ast, "aa")
		b2 = Bind(ast, "aa")
		b3 = Bind(ast, "aa")
		self.assertTrue(b1.tmpl.hash == b2.tmpl.hash and b1.bind == b2.to)
		self.assertTrue(b2 == b3)

		f1 = FilterNode([], beta_root, ast)
		f2 = Filter(ast)
		f3 = Filter(ast)
		self.assertTrue(f1.tmpl.hash == f2.tmpl.hash)
		self.assertTrue(f2 == f3)

	def test_condition_vars(self):
		lhs = "$x:$c, $a > 1 and $b < 100"
		ast = Expression(lhs).ast
		self.assertEquals(ast.hash, '$x:$c and $a>1 and $b<100')

		h2 = Has("order", ast)
		self.assertEquals(h2.vars.keys(), ['a', 'b'])

	def test_condition_contain(self):
		lhs = "$x:$c, $a > 1 and $b < 100"
		ast = Expression(lhs).ast
		self.assertEquals(ast.hash, '$x:$c and $a>1 and $b<100')

		h2 = Has("order", ast)
		self.assertEquals(h2.contain('x'), 'c')

	def test_condition_test(self):
		lhs = "$x:$c, $a > 1 and $b < 100"
		ast = Expression(lhs).ast
		self.assertEquals(ast.hash, '$x:$c and $a>1 and $b<100')

		h2 = Has("order", ast)

		network = Network()
		ctx = Session(network)
		wme = WME(ctx, 'order', a=2, b=50)
		self.assertTrue(h2.test(wme))

	def test_ncc(self):
		lhs = "$x:$c, $a > 1 and $b < 100"
		ast = Expression(lhs).ast
		self.assertEquals(ast.hash, '$x:$c and $a>1 and $b<100')

		c0 = Has("order", ast)
		c1 = Ncc(Has("account", ast))
		c2 = Ncc(c0, c1)
		self.assertEquals(c2.number_of_conditions, 2)

	def test_root(self):
		root = RootNode()
		lhs = "$x:$c, $a > 1 and $b < 100"
		ast = Expression(lhs).ast

		c0 = Has("order", ast)

		parent1 = root.build_or_share_type_node(c0.object_type)
		node1 = parent1
		for cond in c0.conditions:
			node1 = node1.build_or_share_alpha_node(cond)
		node1.amem = node1.amem or AlphaMemory()
		self.assertTrue(node1.amem)

		parent2 = root.build_or_share_type_node(c0.object_type)
		node2 = parent2
		for cond in c0.conditions:
			node2 = node2.build_or_share_alpha_node(cond)
		node2.amem = node2.amem or AlphaMemory()
		self.assertEquals(node1.amem, node2.amem)

	def test_add_wme(self):
		root = RootNode()

		c0 = Has("order", Expression("$a > 1").ast)
		c1 = Has("order", Expression("$a > 1 and $b < 100").ast)

		parent1 = root.build_or_share_type_node(c0.object_type)
		node1 = parent1
		for cond in c0.conditions:
			node1 = node1.build_or_share_alpha_node(cond)
		node1.amem = node1.amem or AlphaMemory()

		parent2 = root.build_or_share_type_node(c1.object_type)
		node2 = parent2
		for cond in c1.conditions:
			node2 = node2.build_or_share_alpha_node(cond)
		node2.amem = node2.amem or AlphaMemory()

		network = Network()
		ctx = Session(network)
		wme = WME(ctx, 'order', a=2, b=50)
		root.activate(wme)

		self.assertEquals(len(node1.amem.items), 1)
		self.assertEquals(len(node2.amem.items), 1)

	def test_network_case0(self):
		network = Network()

		c0 = Has("order", Expression("$uid == 1").ast)
		c1 = Has("order", Expression("$order_id == 1").ast)
		p0 = network.add_production(Rule(c0, c1))

		ctx = Session(network)
		w0 = WME(ctx, "order", uid=1, order_id=0)
		w1 = WME(ctx, "order", uid=0, order_id=1)
		network.add_wme(w0)
		self.assertFalse(p0.items)

		network.remove_wme(w0)
		network.add_wme(w1)
		self.assertFalse(p0.items)

		network.add_wme(w0)
		network.add_wme(w1)
		self.assertTrue(p0.items)

	def test_network_case1(self):
		network = Network()

		c0 = Has('wme', Expression('$x:$id, $y:$value, $attribute == "on"').ast)
		c1 = Has('wme', Expression('$z:$value, $id == $y and $attribute == "left-of"').ast)
		c2 = Has('wme', Expression('$id == $z and $attribute == "color" and $value == "red"').ast)
		p0 = network.add_production(Rule(c0, c1, c2))
		self.assertFalse(p0.items)

		am0 = network.build_or_share_alpha_memory(network.alpha_root, c0)
		am1 = network.build_or_share_alpha_memory(network.alpha_root, c1)
		am2 = network.build_or_share_alpha_memory(network.alpha_root, c2)
		dummy_join = am0.successors[0]
		join_on_value_y = am1.successors[0]
		join_on_value_z = am2.successors[0]
		match_c0 = dummy_join.children[0]
		match_c0c1 = join_on_value_y.children[0]
		match_c0c1c2 = join_on_value_z.children[0]

		ctx = Session(network)
		wmes = [
			WME(ctx, 'wme', id='B1', attribute='on', value='B2'),
			WME(ctx, 'wme', id='B1', attribute='on', value='B3'),
			WME(ctx, 'wme', id='B1', attribute='color', value='red'),
			WME(ctx, 'wme', id='B2', attribute='on', value='table'),
			WME(ctx, 'wme', id='B2', attribute='left-of', value='B3'),
			WME(ctx, 'wme', id='B2', attribute='color', value='blue'),
			WME(ctx, 'wme', id='B3', attribute='left-of', value='B4'),
			WME(ctx, 'wme', id='B3', attribute='on', value='table'),
			WME(ctx, 'wme', id='B3', attribute='color', value='red'),
		]
		for wme in wmes:
			network.add_wme(wme)

		self.assertEquals(am0.items, [wmes[0], wmes[1], wmes[3], wmes[7]])
		self.assertEquals(am1.items, [wmes[4], wmes[6]])
		self.assertEquals(am2.items, [wmes[2], wmes[8]])
		self.assertEquals(len(match_c0.items), 4)
		self.assertEquals(len(match_c0c1.items), 2)
		self.assertEquals(len(match_c0c1c2.items), 1)

		t0 = Token(Token(None, None), wmes[0])
		t1 = Token(t0, wmes[4])
		t2 = Token(t1, wmes[8])
		self.assertEquals(match_c0c1c2.items[0], t2)

		network.remove_wme(wmes[0])
		self.assertEquals(am0.items, [wmes[1], wmes[3], wmes[7]])
		self.assertEquals(len(match_c0.items), 3)
		self.assertEquals(len(match_c0c1.items), 1)
		self.assertEquals(len(match_c0c1c2.items), 0)

	def test_dup(self):
		network = Network()

		c0 = Has('wme', Expression('$x:$id, $y:$value, $attribute == "self"').ast)
		c1 = Has('wme', Expression('$x:$id, $attribute == "color" and $value == "red"').ast)
		c2 = Has('wme', Expression('$id == $y and $attribute == "color" and $value == "red"').ast)
		p0 = network.add_production(Rule(c0, c1, c2))
		self.assertFalse(p0.items)

		ctx = Session(network)
		wmes = [
			WME(ctx, 'wme', id='B1', attribute='self', value='B1'),
			WME(ctx, 'wme', id='B1', attribute='color', value='red'),
		]
		for wme in wmes:
			network.add_wme(wme)

		am = network.build_or_share_alpha_memory(network.alpha_root, c2)
		join_on_value_y = am.successors[1]
		match_for_all = join_on_value_y.children[0]

		self.assertEquals(len(match_for_all.items), 1)

	def test_multi_productions(self):
		network = Network()
		c0 = Has('wme', Expression('$x:$id, $y:$value, $attribute == "on"').ast)
		c1 = Has('wme', Expression('$z:$value, $id == $y and $attribute == "left-of"').ast)
		c2 = Has('wme', Expression('$id == $z and $attribute == "color" and $value == "red"').ast)
		c3 = Has('wme', Expression('$id == $z and $attribute == "on" and $value == "table"').ast)
		c4 = Has('wme', Expression('$id == $z and $attribute == "left-of" and $value == "B4"').ast)

		p0 = network.add_production(Rule(c0, c1, c2))
		p1 = network.add_production(Rule(c0, c1, c3, c4))

		ctx = Session(network)
		wmes = [
			WME(ctx, 'wme', id='B1', attribute='on', value='B2'),
			WME(ctx, 'wme', id='B1', attribute='on', value='B3'),
			WME(ctx, 'wme', id='B1', attribute='color', value='red'),
			WME(ctx, 'wme', id='B2', attribute='on', value='table'),
			WME(ctx, 'wme', id='B2', attribute='left-of', value='B3'),
			WME(ctx, 'wme', id='B2', attribute='color', value='blue'),
			WME(ctx, 'wme', id='B3', attribute='left-of', value='B4'),
			WME(ctx, 'wme', id='B3', attribute='on', value='table'),
			WME(ctx, 'wme', id='B3', attribute='color', value='red'),
		]
		for wme in wmes:
			network.add_wme(wme)

		# add product on the fly
		p2 = network.add_production(Rule(c0, c1, c3, c2))

		self.assertEquals(len(p0.items), 1)
		self.assertEquals(len(p1.items), 1)
		self.assertEquals(len(p2.items), 1)
		self.assertEquals(p0.items[0].wmes, [wmes[0], wmes[4], wmes[8]])
		self.assertEquals(p1.items[0].wmes, [wmes[0], wmes[4], wmes[7], wmes[6]])
		self.assertEquals(p2.items[0].wmes, [wmes[0], wmes[4], wmes[7], wmes[8]])

		network.remove_production(p2)
		self.assertEquals(len(p2.items), 0)

	def test_negative_condition(self):
		network = Network()
		c0 = Has('wme', Expression('$x:$id, $y:$value, $attribute == "on"').ast)
		c1 = Has('wme', Expression('$z:$value, $y == $id and $attribute == "left-of"').ast)
		c2 = Neg('wme', Expression('$z == $id and $attribute == "color" and $value == "red"').ast)
		p0 = network.add_production(Rule(c0, c1, c2))

		ctx = Session(network)
		wmes = [
			WME(ctx, 'wme', id='B1', attribute='on', value='B2'),
			WME(ctx, 'wme', id='B1', attribute='on', value='B3'),
			WME(ctx, 'wme', id='B1', attribute='color', value='red'),
			WME(ctx, 'wme', id='B2', attribute='on', value='table'),
			WME(ctx, 'wme', id='B2', attribute='left-of', value='B3'),
			WME(ctx, 'wme', id='B2', attribute='color', value='blue'),
			WME(ctx, 'wme', id='B3', attribute='left-of', value='B4'),
			WME(ctx, 'wme', id='B3', attribute='on', value='table'),
			WME(ctx, 'wme', id='B3', attribute='color', value='red'),
		]
		for wme in wmes:
			network.add_wme(wme)
		self.assertEquals(p0.items[0].wmes, [
			WME(ctx, 'wme', id='B1', attribute='on', value='B3'),
			WME(ctx, 'wme', id='B3', attribute='left-of', value='B4'),
			None
		])

	def test_ncc_condition(self):
		network = Network()
		c0 = Has('wme', Expression('$y:$value, $x:$id, $attribute == "on"').ast)
		c1 = Has('wme', Expression('$z:$value, $id == $y and $attribute == "left-of"').ast)
		c2 = Has('wme', Expression('$id == $z and $attribute == "color" and $value == "red"').ast)
		c3 = Has('wme', Expression('$w:$value, $id == $z and $attribute == "on"').ast)

		p0 = network.add_production(Rule(c0, c1, Ncc(c2, c3)))
		ctx = Session(network)
		wmes = [
			WME(ctx, 'wme', id='B1', attribute='on', value='B2'),
			WME(ctx, 'wme', id='B1', attribute='on', value='B3'),
			WME(ctx, 'wme', id='B1', attribute='color', value='red'),
			WME(ctx, 'wme', id='B2', attribute='on', value='table'),
			WME(ctx, 'wme', id='B2', attribute='left-of', value='B3'),
			WME(ctx, 'wme', id='B2', attribute='color', value='blue'),
			WME(ctx, 'wme', id='B3', attribute='left-of', value='B4'),
			WME(ctx, 'wme', id='B3', attribute='on', value='table'),
		]
		for wme in wmes:
			network.add_wme(wme)
		self.assertEquals(len(p0.items), 2)
		network.add_wme(WME(ctx, 'wme', id='B3', attribute='color', value='red'))
		self.assertEquals(len(p0.items), 1)

	def test_black_white(self):
		network = Network()
		c1 = Has('wme', Expression('$cid:$value, $item:$id, $attribute == "cat"').ast)
		c2 = Has('wme', Expression('$sid:$value, $item == $id and $attribute == "shop"').ast)
		white = Ncc(
			Neg('wme', Expression('$item == $id and $attribute == "cat" and $value == "100"').ast),
			Neg('wme', Expression('$item == $id and $attribute == "cat" and $value == "101"').ast),
			Neg('wme', Expression('$item == $id and $attribute == "cat" and $value == "102"').ast),
		)
		n1 = Neg('wme', Expression('$item == $id and $attribute == "shop" and $value == "1"').ast)
		n2 = Neg('wme', Expression('$item == $id and $attribute == "shop" and $value == "2"').ast)
		n3 = Neg('wme', Expression('$item == $id and $attribute == "shop" and $value == "3"').ast)
		p0 = network.add_production(Rule(c1, c2, white, n1, n2, n3))
		ctx = Session(network)
		wmes = [
			WME(ctx, 'wme', id='item:1', attribute='cat',  value='101'),
			WME(ctx, 'wme', id='item:1', attribute='shop', value='4'),
			WME(ctx, 'wme', id='item:2', attribute='cat',  value='100'),
			WME(ctx, 'wme', id='item:2', attribute='shop', value='1'),
		]
		for wme in wmes:
			network.add_wme(wme)

		self.assertEquals(len(p0.items), 1)
		self.assertEquals(p0.items[0].get_binding('item'), 'item:1')

	def test_filter_compare(self):
		network = Network()
		c0 = Has('wme', Expression('$x:$value, $id == "spu:1" and $attribute == "price"').ast)
		f0 = Filter(Expression('$x>100').ast)
		f1 = Filter(Expression('$x<200').ast)
		f2 = Filter(Expression('$x>200 and $x<400').ast)
		f3 = Filter(Expression('$x>300').ast)

		p0 = network.add_production(Rule(c0, f0, f1))
		p1 = network.add_production(Rule(c0, f2))
		p2 = network.add_production(Rule(c0, f3))

		ctx = Session(network)
		network.add_wme(WME(ctx, 'wme', id='spu:1', attribute='price', value=100))
		network.add_wme(WME(ctx, 'wme', id='spu:1', attribute='price', value=150))
		network.add_wme(WME(ctx, 'wme', id='spu:1', attribute='price', value=300))

		self.assertEquals(len(p0.items), 1)
		token = p0.items.pop()
		self.assertEquals(token.get_binding('x'), 150)

		self.assertEquals(len(p1.items), 1)
		token = p1.items.pop()
		self.assertEquals(token.get_binding('x'), 300)

		self.assertFalse(p2.items)

	def test_bind(self):
		network = Network()
		ctx = Session(network)

		c0 = Has('wme', Expression('$x:$value, $id == "spu:1" and $attribute == "sales"').ast)
		b0 = Bind(Expression('$x - 10').ast, 'num')
		f0 = Filter(Expression('$num > 0').ast)
		p0 = network.add_production(Rule(c0, b0, f0))

		b1 = Bind(Expression('$x - 50').ast, 'num')
		p1 = network.add_production(Rule(c0, b1, f0))

		b2 = Bind(Expression('$x - 60').ast, 'num')
		p2 = network.add_production(Rule(c0, b2, f0))

		network.add_wme(WME(ctx, 'wme', id='spu:1', attribute='sales', value=60))

		self.assertEquals(len(p0.items), 1)
		self.assertEquals(len(p1.items), 1)
		self.assertEquals(len(p2.items), 0)
		t0 = p0.items[0]
		t1 = p1.items[0]
		self.assertEquals(t0.get_binding('num'), 50)
		self.assertEquals(t1.get_binding('num'), 10)
