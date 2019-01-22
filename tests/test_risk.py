# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals, division, print_function

import heapq
import logging
import unittest

from rule.rule_parser import (
	Expression, ValueNode, VariableNode, OperatorOr
)
from rule.rule_rete import (
	Network, Session, Rule, Has, Agenda
)

log = logging.getLogger("main")


RISK_RULE_ACTION_REJECT = 1
RISK_RULE_ACTION_REVIEW = 2
RISK_RULE_ACTION_ACCEPT = 3


class RiskRule(object):
	def __init__(self, rule_id, priority, condition, action, fields=None, object_type="default", meta=None, name=None, desc=None):
		"""
		:param unicode rule_id:
		:param int priority:
		:param unicode condition:
		:param int action:
		:param dict fields:
		:param unicode object_type:
		:param dict meta:
		:param unicode name:
		:param unicode desc:
		"""
		self.rule_id = rule_id
		self.priority = priority
		self.condition = Expression(condition, fields)
		self.action = action
		self.type = object_type
		self.meta = meta or {}
		self.name = name or self.rule_id
		self.desc = desc or self.name
		self.expr = condition

		self.variable_nodes = self.condition.get_nodes_by_type(VariableNode)
		self.variable_names = set([variable_node.leaf for variable_node in self.variable_nodes])

	def has_var(self, name):
		return name in self.variable_names


class RiskAgenda(Agenda):
	def __init__(self, engine):
		self.engine = engine

	def resolve(self, matches):
		default_strategy = []
		for p in matches:
			matched_rules = self.engine.mapping.get(p) or []
			for rule in matched_rules:
				default_strategy.append((rule.priority, rule.action))
		heapq.heapify(default_strategy)
		if default_strategy:
			_, action = heapq.heappop(default_strategy)
			return action
		return None


class RiskEngine(object):
	def __init__(self):
		self.rules = {}
		self.mapping = {}
		self.network = Network()
		self.agenda = RiskAgenda(self)

	def create_session(self, global_scope=None, current_time=None):
		return Session(self.network, self.agenda, global_scope, current_time)

	def compile_rules(self, *args):
		for arg in args:
			self.compile_rule(arg)

	def compile_rule(self, rule):
		"""
		:param RiskRule rule:
		"""
		if not rule.condition.ast:
			_rules = [ValueNode(True)]
		elif isinstance(rule.condition.ast, OperatorOr):
			_rules = rule.condition.ast.children
		else:
			_rules = [rule.condition.ast]
		log.info("parse rule [rule_id: %s] to %d sub rules", rule.rule_id, len(_rules))

		_rules = [
			Rule(Has(rule.type, _rule))
			for _rule in _rules
		]

		for _rule in _rules:
			log.info("add sub rule for [rule_id: %s]: %s", rule.rule_id, list(_rule))

			try:
				p = self.network.add_production(_rule)
			except Expression as ex:
				log.exception("failed to add sub rule for [rule_id: %s]: %s, %s", rule.rule_id, list(_rule), ex)
				continue
			self.mapping.setdefault(p, []).append(rule)

		self.rules[rule.rule_id] = rule


class TestRisk(unittest.TestCase):
	def test_risk_rules(self):
		rules = [
			RiskRule("1", 0, "$a > 1 and $b < MAX", RISK_RULE_ACTION_ACCEPT, {"MAX": {"value": 100}}),
			RiskRule("2", 0, "$a > 1 and $d > MIN", RISK_RULE_ACTION_ACCEPT, {"MIN": {"value": 1}}),
			RiskRule("3", 0, "not ($a == 1 or $b == 2)", RISK_RULE_ACTION_ACCEPT),
			RiskRule("4", 0, "$x == 1", RISK_RULE_ACTION_ACCEPT),
			RiskRule("5", 0, "$x in LIST", RISK_RULE_ACTION_ACCEPT, {"LIST": {"value": [1, 2, 3]}}),
		]
		self.assertEqual(rules[0].type, "default")

		engine = RiskEngine()
		engine.compile_rules(*rules)

		session = engine.create_session()
		session["x"] = 1
		wmes = [
			session.create_wme(a=2, b=99, c=100, d=1),
			session.create_wme(a=100, b=99, c=100, d=2),
		]
		session.execute(*wmes)
		session.resolve()
		print(session.lru, session.matches)
