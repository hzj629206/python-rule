# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals, division, print_function

import copy
import logging
from datetime import datetime
from .rule_parser import (
	Node, VariableNode, OperatorAnd, OperatorBind,
)

log = logging.getLogger("main")


class Variable(object):
	def __init__(self, name):
		"""
		:param unicode name:
		"""
		self.name = name

	def evaluate(self, context):
		"""
		:param Context context:
		"""
		return self.resolve(context, context.session)

	def resolve(self, context, session):
		"""
		:param Context context:
		:param Session session:
		"""
		raise NotImplementedError


class Agenda(object):
	def resolve(self, matches):
		"""
		:param list[PNode] matches:
		"""
		raise NotImplementedError


class Session(object):
	def __init__(self, network, agenda=None, lookup=None, current_time=None):
		"""
		:param Network network:
		:param Agenda agenda:
		:param lookup: context to lookup variable object,
			impl __contains__, __getitem__, get and return instance of Variable
		:param datetime|None current_time:
		"""
		self.network = network
		self.agenda = agenda
		self.lookup = lookup

		self.current_time = current_time or datetime.now()
		self.dict = {}
		self.lru = []
		self.matches = []
		self.wmes = []

	def __contains__(self, item):
		return item in self.dict

	def __getitem__(self, item):
		self.add_lru(item)
		return self.dict[item]

	def __setitem__(self, key, value):
		self.dict[key] = value

	def get(self, key, default=None):
		self.add_lru(key)
		return self.dict.get(key, default)

	def add_lru(self, name):
		if not self.lru or self.lru[-1] != name:
			self.lru.append(name)

	def create_wme(self, object_type="default", **kwargs):
		return WME(self, object_type, **kwargs)

	def execute(self, *args):
		if self.wmes:
			raise Exception("already executed.")

		self.wmes = args  # list of WME
		i, wme = -1, None
		try:
			for i, wme in enumerate(self.wmes):
				if not wme.session:
					wme.session = self
				assert wme.session == self, "session does not match"
				assert wme.type, "object type is not set"
				self.network.add_wme(wme)
		except Exception as ex:
			log.exception("failed to add wme. i = %s, wme = %s, %s", i, wme, ex)
		finally:
			for i in range(i + 1):
				self.network.remove_wme(self.wmes[i])

	def resolve(self):  # by strategy
		# tokens could be removed. i.e. Neg
		self.matches = [p for p in set(self.matches) if p.items]
		if self.agenda:
			return self.agenda.resolve(self.matches)
		return None


class WME(object):
	def __init__(self, session, object_type="default", **kwargs):
		"""
		:param Session session:
		:param unicode object_type:
		"""
		self.session = session
		self.type = object_type
		self.dict = dict(kwargs)
		self.context = Context({}, self, self.session)

		self.amems = []  # the ones containing this WME
		self.tokens = []  # the ones containing this WME
		self.negative_join_result = []  # list of NegativeJoinResult

	def __contains__(self, item):
		if item in self.dict:
			return True
		if self.session.lookup and item in self.session.lookup:
			return True
		return False

	def __getitem__(self, item):
		if item in self.dict:
			self.session.add_lru(item)
			return self.dict[item]
		if self.session.lookup and item in self.session.lookup:
			return self.session.lookup[item].evaluate(self.context)
		log.warning("unknown item in this wme context, item: %s", item)
		raise KeyError("unknown item %s in this wme context" % item)

	def __setitem__(self, key, value):
		if self.session.lookup and key in self.session.lookup:
			self.session.add_lru(key)
			log.warning("overwrite builtin variable value, key: %s, value: %s", key, value)
		self.dict[key] = value

	def __repr__(self):
		return "%s(%s)" % (
			self.type, ",".join("%s=%s" % item for item in self.dict.items())
		)

	def __eq__(self, other):
		return (
			isinstance(other, WME)
			and self.type == other.type
			and self.session == other.session
			and self.dict == other.dict
		)


class Context(object):
	def __init__(self, primary, *args):
		if isinstance(primary, Context):
			self.primary = primary.primary
			self.lookups = primary.lookups + args
		else:
			self.primary = primary
			self.lookups = args

		if isinstance(self.primary, Session):
			self.session = self.primary
		else:
			self.session = next((arg for arg in args if isinstance(arg, Session)), None)

	def __contains__(self, item):
		for lookup in self.lookups:
			if lookup and item in lookup:
				return True
		return item in self.primary

	def __getitem__(self, item):
		for lookup in self.lookups:
			if lookup and item in lookup:
				self.session and self.session.add_lru(item)
				if item in self.primary:
					log.warning(
						"ignored value in primary, item: %s, lookup: %s, primary: %s",
						item, lookup[item], self.primary[item]
					)
				return lookup[item]
		self.session and self.session.add_lru(item)
		return self.primary[item]

	def __setitem__(self, key, value):
		for lookup in self.lookups:
			if lookup and key in lookup:
				log.warning(
					"try to overwrite %s, but the new value will be ignored. old: %s, new: %s",
					key, lookup[key], value
				)
		self.session and self.session.add_lru(key)
		self.primary[key] = value

	def get(self, key, default=None):
		for lookup in self.lookups:
			if lookup and key in lookup:
				self.session and self.session.add_lru(key)
				if key in self.primary:
					log.warning(
						"ignored value in primary, item: %s, lookup: %s, primary: %s",
						key, lookup[key], self.primary[key]
					)
				return lookup[key]
		self.session and self.session.add_lru(key)
		return self.primary.get(key, default)


class Network(object):
	def __init__(self):
		self.alpha_root = RootNode()
		self.beta_root = BetaNode()

	def add_production(self, lhs, **kwargs):
		"""
		:param Rule lhs:
		:rtype: PNode
		"""
		current_node = self.build_or_share_network_for_conditions(self.alpha_root, self.beta_root, lhs, [])
		return self.build_or_share_p(current_node, **kwargs)

	def remove_production(self, node):
		"""
		:param PNode node:
		"""
		self.delete_node_and_any_unused_ancestors(node)

	def add_wme(self, wme):
		"""
		:type wme: WME
		"""
		self.alpha_root.activate(wme)

	@classmethod
	def remove_wme(cls, wme):
		"""
		:type wme: WME
		"""
		for am in wme.amems:
			am.items.remove(wme)
		for t in wme.tokens:
			Token.delete_token_and_descendents(t)
		for jr in wme.negative_join_result:
			jr.owner.join_results.remove(jr)
			if not jr.owner.join_results:
				for child in jr.owner.node.children:
					child.left_activation(jr.owner, None)

	def build_or_share_alpha_memory(self, alpha_parent, condition):
		"""
		:type alpha_parent: RootNode
		:type condition: Has|Neg
		:rtype: AlphaMemory
		"""
		alpha_parent = alpha_parent.build_or_share_type_node(condition.object_type)
		node = alpha_parent
		for cond in condition.conditions:
			node = node.build_or_share_alpha_node(cond)
		if node.amem:
			am = node.amem
		else:
			am = AlphaMemory()
			node.amem = am
		for w in alpha_parent.amem.items:
			if condition.test(w):
				am.activate(w)
		return am

	@classmethod
	def get_join_tests_from_condition(cls, c, earlier_conditions):
		"""
		:type c: Has
		:type earlier_conditions: list of Condition
		:rtype: list of TestAtJoinNode
		"""
		result = []
		for v, field_of_v in c.vars.items():
			for idx, cond in enumerate(earlier_conditions):
				if not isinstance(cond, Has) or isinstance(cond, Neg):  # only Has
					continue
				field_of_v2 = cond.contain(v)
				if not field_of_v2:
					continue
				t = TestAtJoinNode(field_of_v, idx, field_of_v2, v)
				result.append(t)
				# remove join condition
				if field_of_v in c.conditions:
					c.conditions.remove(field_of_v)
		return result

	@classmethod
	def build_or_share_join_node(cls, parent, amem, tests, has):
		"""
		:type has: Has
		:type parent: BetaNode
		:type amem: AlphaMemory
		:type tests: list of TestAtJoinNode
		:rtype: JoinNode
		"""
		for child in parent.children:
			if (
				isinstance(child, JoinNode)
				and child.amem == amem
				and child.tests == tests
				and child.has == has
			):
				return child
		node = JoinNode([], parent, amem, tests, has)
		parent.children.append(node)
		amem.successors.append(node)
		return node

	@classmethod
	def build_or_share_negative_node(cls, parent, amem, tests):
		"""
		:type parent: BetaNode
		:type amem: AlphaMemory
		:type tests: list of TestAtJoinNode
		:rtype: JoinNode
		"""
		for child in parent.children:
			if isinstance(child, NegativeNode) and child.amem == amem and child.tests == tests:
				return child
		node = NegativeNode(parent=parent, amem=amem, tests=tests)
		parent.children.append(node)
		amem.successors.append(node)
		return node

	def build_or_share_beta_memory(self, parent):
		"""
		:type parent: BetaNode
		:rtype: BetaMemory
		"""
		for child in parent.children:
			if isinstance(child, BetaMemory):
				return child
		node = BetaMemory(None, parent)
		# dummy top beta memory
		if parent == self.beta_root:
			node.items.append(Token(None, None))
		parent.children.append(node)
		self.update_new_node_with_matches_from_above(node)
		return node

	def build_or_share_p(self, parent, **kwargs):
		"""
		:type parent: BetaNode
		:rtype: PNode
		"""
		for child in parent.children:
			if isinstance(child, PNode):
				return child
		node = PNode(None, parent, **kwargs)
		parent.children.append(node)
		self.update_new_node_with_matches_from_above(node)
		return node

	def build_or_share_ncc_nodes(self, alpha_parent, beta_parent, ncc, earlier_conditions):
		"""
		:type earlier_conditions: list of Condition
		:type ncc: Ncc
		:type beta_parent: BetaNode
		:type alpha_parent: RootNode
		"""
		bottom_of_sub_network = self.build_or_share_network_for_conditions(alpha_parent, beta_parent, ncc, earlier_conditions)
		for child in beta_parent.children:
			if isinstance(child, NccNode) and child.partner.parent == bottom_of_sub_network:
				return child
		ncc_node = NccNode([], beta_parent)
		ncc_partner = NccPartnerNode([], bottom_of_sub_network)
		beta_parent.children.append(ncc_node)
		bottom_of_sub_network.children.append(ncc_partner)
		ncc_node.partner = ncc_partner
		ncc_partner.ncc_node = ncc_node
		ncc_partner.number_of_conditions = ncc.number_of_conditions
		self.update_new_node_with_matches_from_above(ncc_node)
		self.update_new_node_with_matches_from_above(ncc_partner)
		return ncc_node

	def build_or_share_filter_node(self, parent, f):
		"""
		:type f: Filter
		:type parent: BetaNode
		"""
		for child in parent.children:
			if isinstance(child, FilterNode) and child.tmpl.hash == f.tmpl.hash:
				return child
		node = FilterNode([], parent, f.tmpl)
		parent.children.append(node)
		return node

	def build_or_share_bind_node(self, parent, b):
		"""
		:type b: Bind
		:type parent: BetaNode
		"""
		for child in parent.children:
			if (
				isinstance(child, BindNode)
				and child.tmpl.hash == b.tmpl.hash
				and child.bind == b.to
			):
				return child
		node = BindNode([], parent, b.tmpl, b.to)
		parent.children.append(node)
		return node

	def build_or_share_network_for_conditions(self, alpha_parent, beta_parent, rule, earlier_conditions):
		"""
		:type earlier_conditions: list of Condition
		:type alpha_parent: RootNode
		:type beta_parent: BetaNode
		:type rule: Rule
		"""
		current_node = beta_parent
		conditions_higher_up = earlier_conditions

		for cond in rule:
			if isinstance(cond, Neg):
				tests = self.get_join_tests_from_condition(cond, conditions_higher_up)
				am = self.build_or_share_alpha_memory(alpha_parent, cond)
				current_node = self.build_or_share_negative_node(current_node, am, tests)
			elif isinstance(cond, Has):
				current_node = self.build_or_share_beta_memory(current_node)
				tests = self.get_join_tests_from_condition(cond, conditions_higher_up)
				am = self.build_or_share_alpha_memory(alpha_parent, cond)
				current_node = self.build_or_share_join_node(current_node, am, tests, cond)
			elif isinstance(cond, Ncc):
				current_node = self.build_or_share_ncc_nodes(alpha_parent, current_node, cond, conditions_higher_up)
			elif isinstance(cond, Filter):  # helper condition, must be after Has/Neg/Ncc
				current_node = self.build_or_share_filter_node(current_node, cond)
			elif isinstance(cond, Bind):  # helper condition, must be after Has/Neg/Ncc
				current_node = self.build_or_share_bind_node(current_node, cond)
			else:
				raise NotImplementedError
			conditions_higher_up.append(cond)
		return current_node

	@classmethod
	def update_new_node_with_matches_from_above(cls, new_node):
		"""
		:type new_node: BetaNode
		"""
		parent = new_node.parent
		if isinstance(parent, BetaMemory):
			for token in parent.items:
				new_node.left_activation(token, None)
		elif isinstance(parent, JoinNode):
			saved_list_of_children = parent.children
			parent.children = [new_node]
			for item in parent.amem.items:
				parent.right_activation(item)
			parent.children = saved_list_of_children
		elif isinstance(parent, NegativeNode):
			for token in parent.items:
				if not token.join_results:
					new_node.left_activation(token, None)
		elif isinstance(parent, NccNode):
			for token in parent.items:
				if not token.ncc_results:
					new_node.left_activation(token, None)

	@classmethod
	def delete_node_and_any_unused_ancestors(cls, node):
		"""
		:type node: BetaNode
		"""
		if isinstance(node, JoinNode):
			node.amem.successors.remove(node)
		else:
			for item in node.items:
				Token.delete_token_and_descendents(item)
		node.parent.children.remove(node)
		if not node.parent.children:
			cls.delete_node_and_any_unused_ancestors(node.parent)


class Has(object):
	def __init__(self, object_type, pattern):
		"""
		bind: ObjectType(expression)
		:param unicode object_type:
		:param Node pattern:
		"""
		self.object_type = object_type
		self.pattern = pattern
		self.pattern_hash = pattern.hash

		self._vars = {}
		if isinstance(pattern, OperatorAnd):
			self.conditions = pattern.children
		else:
			self.conditions = [pattern]
		self.conditions = [cond for cond in self.conditions if not isinstance(cond, OperatorBind)]

		for cond in self.conditions:
			_vars = cond.get_nodes_by_type(VariableNode)
			for _var in _vars:
				self._vars[_var.leaf] = cond

		self._bindings = {}
		bindings = pattern.get_nodes_by_type(OperatorBind)
		for binding in bindings:
			if isinstance(binding, OperatorBind):
				self._bindings[binding.name] = (binding.name, binding.attr)

	def __repr__(self):
		return "(%s(%s))" % (self.object_type, self.pattern_hash)

	def __eq__(self, other):
		return (
			self.__class__ == other.__class__
			and self.pattern_hash == other.pattern_hash
		)

	@property
	def vars(self):
		"""
		:rtype: dict
		"""
		return self._vars

	@property
	def bindings(self):
		"""
		:rtype: dict
		"""
		return self._bindings

	def contain(self, v):
		"""
		:type v: unicode
		:rtype: unicode
		"""
		if v in self._bindings:
			return self._bindings[v][1]
		return ""

	def test(self, w):
		"""
		:type w: WME
		"""
		for cond in self.conditions:
			if not cond.evaluate(w.context):
				return False
		return True


class Neg(Has):
	"""
	negative condition
	checks for the non-existence of something in the Working Memory.
	Think of "not" as meaning "there must be none of...".
	"""
	def __repr__(self):
		return "-%s" % super(Neg, self).__repr__()


class Rule(list):
	def __init__(self, *args):
		super(Rule, self).__init__()
		self.extend(args)


class Ncc(Rule):
	"""
	negated conjunctive condition/conjunctive negations
	"""
	def __repr__(self):
		return "-%s" % super(Ncc, self).__repr__()

	@property
	def number_of_conditions(self):
		return len(self)


class Filter(object):
	def __init__(self, tmpl):
		"""
		:param Node tmpl:
		"""
		self.tmpl = tmpl

	def __eq__(self, other):
		return isinstance(other, Filter) and self.tmpl.hash == other.tmpl.hash


class Bind(object):
	def __init__(self, tmpl, to):
		"""
		:param Node tmpl:
		:param unicode to:
		"""
		self.tmpl = tmpl
		self.to = to

	def __eq__(self, other):
		return isinstance(other, Bind) and self.tmpl.hash == other.tmpl.hash and self.to == other.to


class Token(object):
	def __init__(self, parent, wme, node=None, binding=None):
		"""
		:type wme: WME|None
		:type parent: Token|None
		:type binding: dict
		"""
		self.parent = parent
		self.wme = wme
		self.node = node  # points to memory this token is in
		self.children = []  # the ones with parent = this token
		self.join_results = []  # used only on tokens in negative nodes
		self.ncc_results = []
		self.owner = None  # Ncc
		self.binding = binding if binding else {}  # {"$x": "B1"}

		if self.wme:
			self.wme.tokens.append(self)
		if self.parent:
			self.parent.children.append(self)

	def __repr__(self):
		return "<Token %s>" % self.wmes

	def __eq__(self, other):
		return (
			isinstance(other, Token) and
			self.parent == other.parent and self.wme == other.wme
		)

	def is_root(self):
		return not self.parent and not self.wme

	@property
	def wmes(self):
		ret = [self.wme]
		t = self
		while not t.parent.is_root():
			t = t.parent
			ret.insert(0, t.wme)
		return ret

	def get_binding(self, v):
		t = self
		ret = t.binding.get(v)
		while not ret and t.parent:
			t = t.parent
			ret = t.binding.get(v)
		return ret

	def all_binding(self):
		path = [self]
		if path[0].parent:
			path.insert(0, path[0].parent)
		binding = {}
		for t in path:
			binding.update(t.binding)
		return binding

	@classmethod
	def delete_token_and_descendents(cls, token):
		"""
		:type token: Token
		"""
		for child in token.children:
			cls.delete_token_and_descendents(child)
		if not isinstance(token.node, NccPartnerNode):
			token.node.items.remove(token)
		if token.wme:
			token.wme.tokens.remove(token)
		if token.parent:
			token.parent.children.remove(token)
		if isinstance(token.node, NegativeNode):
			for jr in token.join_results:
				jr.wme.negative_join_results.remove(jr)
		elif isinstance(token.node, NccNode):
			for result_tok in token.ncc_results:
				result_tok.wme.tokens.remove(result_tok)
				result_tok.parent.children.remove(result_tok)
		elif isinstance(token.node, NccPartnerNode):
			token.owner.ncc_results.remove(token)
			if not token.owner.ncc_results:
				for child in token.node.ncc_node.children:
					child.left_activation(token.owner, None)


class AlphaMemory(object):
	def __init__(self, items=None, successors=None):
		"""
		:type successors: list of BetaNode
		:type items: list of WME
		"""
		self.items = items if items else []
		self.successors = successors if successors else []

	def activate(self, wme):
		"""
		:type wme: WME
		"""
		if wme in self.items:
			return
		self.items.append(wme)
		wme.amems.append(self)
		for child in reversed(self.successors):
			child.right_activation(wme)


class RootNode(object):
	"""
	The root node is where all objects enter the network.
	From there, it immediately goes to the ObjectTypeNode.
	"""
	def __init__(self, children=None):
		"""
		:param list[ObjectTypeNode]|None children:
		"""
		self.parent = None
		self.children = children or []

		# no alpha memory for root node, object type node is required

		self._type_hashing = dict([(child.object_type, child) for child in self.children])

	def build_or_share_type_node(self, object_type):
		"""
		:param unicode object_type:
		:rtype: ObjectTypeNode
		"""
		if object_type in self._type_hashing:
			return self._type_hashing[object_type]
		else:
			node = ObjectTypeNode(object_type, self)
			self._type_hashing[node.object_type] = node
			self.children.append(node)
			return node

	def activate(self, wme):
		"""
		:param WME wme:
		"""
		if wme.type in self._type_hashing:
			self._type_hashing[wme.type].activate(wme)


class ObjectTypeNode(object):
	"""
	The purpose of the ObjectTypeNode is to make sure the engine doesn't do more work than it needs to.
	For example, say we have 2 objects: Account and Order.
	If the rule engine tried to evaluate every single node against every object,
	it would waste a lot of cycles.
	To make things efficient, the engine should only pass the object to the nodes that match the object type.
	"""
	def __init__(self, object_type, parent, children=None):
		"""
		:param unicode object_type:
		:param RootNode parent:
		:param list[AlphaNode]|None children:
		"""
		self.object_type = object_type
		self.parent = parent
		self.children = children or []
		self.amem = AlphaMemory()

		self._alpha_hashing = dict([(child.condition.hash, child) for child in self.children])

	def build_or_share_alpha_node(self, cond):
		"""
		:param Node cond:
		:rtype: AlphaNode
		"""
		pattern_hash = cond.hash
		if pattern_hash in self._alpha_hashing:
			return self._alpha_hashing[pattern_hash]
		else:
			node = AlphaNode(cond, self)
			self._alpha_hashing[pattern_hash] = node
			self.children.append(node)
			return node

	def activate(self, wme):
		"""
		:param WME wme:
		"""
		if wme.type == self.object_type:
			if self.amem:
				self.amem.activate(wme)
			for child in self.children:
				child.activate(wme)


class AlphaNode(object):
	"""
	AlphaNodes are used to evaluate literal conditions.
	Network performs node sharing.
	When a rule has multiple literal conditions for a single object type,
	they are linked together.
	Many rules repeat the same patterns,
	and node sharing allows us to collapse those patterns
	so that they don't have to be re-evaluated for every single instance.
	"""
	def __init__(self, condition, parent, children=None):
		"""
		:param Node condition:
		:param ObjectTypeNode|AlphaNode parent:
		:param list[AlphaNode]|None children:
		"""
		self.condition = condition
		self.parent = parent
		self.children = children or []
		self.amem = None  # type: AlphaMemory

		self._alpha_hashing = self._alpha_hashing = dict([(child.condition.hash, child) for child in self.children])

	def __repr__(self):
		return "<AlphaNode %s>" % self.condition.hash

	def build_or_share_alpha_node(self, cond):
		"""
		:param Node cond:
		:rtype: AlphaNode
		"""
		pattern_hash = cond.hash
		if pattern_hash in self._alpha_hashing:
			return self._alpha_hashing[pattern_hash]
		else:
			node = AlphaNode(cond, self)
			self._alpha_hashing[pattern_hash] = node
			self.children.append(node)
			return node

	def activate(self, wme):
		"""
		:param WME wme:
		"""
		if self.condition.evaluate(wme.context):
			if self.amem:
				self.amem.activate(wme)
			for child in self.children:
				child.activate(wme)


class BetaNode(object):
	"""
	BetaNodes are used to compare 2 objects, and their fields, to each other.
	The objects may be the same or different types.
	"""
	kind = 'dummy-beta'

	def __init__(self, children=None, parent=None):
		"""
		:param list[BetaNode]|None children:
		:param BetaNode|None parent:
		"""
		self.children = children or []
		self.parent = parent
		self.items = []

	def right_activation(self, wme):
		"""
		:param WME wme:
		"""
		pass


class BetaMemory(BetaNode):
	kind = 'beta-memory'

	def __init__(self, children=None, parent=None, items=None):
		"""
		:type children: list of JoinNode or None
		:type items: list of Token or None
		"""
		super(BetaMemory, self).__init__(children=children, parent=parent)
		self.children = children or []  # make PyCharm happy
		self.items = items if items else []

	def left_activation(self, token, wme, binding=None):
		"""
		:type binding: dict
		:type wme: WME
		:type token: Token
		"""
		new_token = Token(token, wme, node=self, binding=binding)
		self.items.append(new_token)
		for child in self.children:
			child.left_activation(new_token)


class BindNode(BetaNode):
	"""
	helper node, bind a value to a variable.
	"""
	kind = 'bind-node'

	def __init__(self, children, parent, expression, to):
		"""
		:type children:
		:type parent: BetaNode
		:type expression: Node
		:type to: unicode
		"""
		super(BindNode, self).__init__(children=children, parent=parent)
		self.tmpl = expression
		self.bind = to

	def left_activation(self, token, wme, binding=None):
		"""
		:type binding: dict
		:type wme: WME
		:type token: Token
		"""
		all_binding = token.all_binding()
		all_binding.update(binding)
		result = self.tmpl.evaluate(Context(wme.context, binding))
		binding[self.bind] = result
		for child in self.children:
			binding = copy.deepcopy(binding)
			child.left_activation(token, wme, binding)


class FilterNode(BetaNode):
	"""
	helper node, extra condition check.
	"""
	kind = 'filter-node'

	def __init__(self, children, parent, condition):
		"""
		:type children:
		:type parent: BetaNode
		:type condition: Node
		"""
		super(FilterNode, self).__init__(children=children, parent=parent)
		self.tmpl = condition

	def left_activation(self, token, wme, binding=None):
		"""
		:type binding: dict
		:type wme: WME
		:type token: Token
		"""
		all_binding = token.all_binding()
		all_binding.update(binding)
		result = self.tmpl.evaluate(Context(wme.context, binding))
		if bool(result):
			for child in self.children:
				child.left_activation(token, wme, binding)


class JoinNode(BetaNode):
	kind = 'join-node'

	def __init__(self, children, parent, amem, tests, has):
		"""
		:type children: list of BetaNode
		:type parent: BetaNode
		:type amem: AlphaMemory
		:type tests: list of TestAtJoinNode
		:type has: Has
		"""
		super(JoinNode, self).__init__(children=children, parent=parent)
		self.amem = amem
		self.tests = tests
		self.has = has

	def right_activation(self, wme):
		"""
		:type wme: WME
		"""
		for token in self.parent.items:
			if self.perform_join_test(token, wme):
				binding = self.make_binding(wme)
				for child in self.children:
					child.left_activation(token, wme, binding)

	def left_activation(self, token):
		"""
		:type token: Token
		"""
		for wme in self.amem.items:
			if self.perform_join_test(token, wme):
				binding = self.make_binding(wme)
				for child in self.children:
					child.left_activation(token, wme, binding)

	def perform_join_test(self, token, wme):
		"""
		:type token: Token
		:type wme: WME
		"""
		for this_test in self.tests:
			wme2 = token.wmes[this_test.condition_number_of_arg2]
			arg2 = wme2[this_test.field_of_arg2]

			if not this_test.field_of_arg1.evaluate(Context(
				wme.context, {this_test.var_of_arg2: arg2}
			)):
				return False
		return True

	def make_binding(self, wme):
		"""
		:type wme: WME
		"""
		binding = {}
		for name, attr in self.has.bindings.values():
			val = wme[attr]
			binding[name] = val
		return binding


class TestAtJoinNode(object):
	def __init__(self, field_of_arg1, condition_number_of_arg2, field_of_arg2, var_of_arg2):
		"""
		:param Node field_of_arg1:
		:param int condition_number_of_arg2:
		:param unicode field_of_arg2:
		:param unicode var_of_arg2:
		"""
		self.field_of_arg1 = field_of_arg1
		self.condition_number_of_arg2 = condition_number_of_arg2
		self.field_of_arg2 = field_of_arg2
		self.var_of_arg2 = var_of_arg2

	def __repr__(self):
		return "<TestAtJoinNode %s:Condition%s.%s, %s?>" % (
			self.var_of_arg2, self.condition_number_of_arg2, self.field_of_arg2,
			self.field_of_arg1.hash
		)

	def __eq__(self, other):
		return isinstance(other, TestAtJoinNode) and \
			self.field_of_arg1.hash == other.field_of_arg1.hash and \
			self.field_of_arg2 == other.field_of_arg2 and \
			self.condition_number_of_arg2 == other.condition_number_of_arg2 and\
			self.var_of_arg2 == other.var_of_arg2


class NegativeJoinResult(object):
	def __init__(self, owner, wme):
		"""
		:type wme: WME
		:type owner: Token
		"""
		self.owner = owner
		self.wme = wme


class NegativeNode(BetaNode):
	def __init__(self, children=None, parent=None, amem=None, tests=None):
		"""
		:type amem: AlphaMemory
		"""
		super(NegativeNode, self).__init__(children=children, parent=parent)
		self.items = []
		self.amem = amem
		self.tests = tests if tests else []

	def left_activation(self, token, wme, binding=None):
		"""
		:type wme: WME
		:type token: Token
		:type binding: dict
		"""
		new_token = Token(token, wme, self, binding)
		self.items.append(new_token)
		for item in self.amem.items:
			if self.perform_join_test(new_token, item):
				jr = NegativeJoinResult(new_token, item)
				new_token.join_results.append(jr)
				item.negative_join_result.append(jr)
		if not new_token.join_results:
			for child in self.children:
				child.left_activation(new_token, None)

	def right_activation(self, wme):
		"""
		:type wme: WME
		"""
		for t in self.items:
			if self.perform_join_test(t, wme):
				if not t.join_results:
					Token.delete_token_and_descendents(t)
				jr = NegativeJoinResult(t, wme)
				t.join_results.append(jr)
				wme.negative_join_result.append(jr)

	def perform_join_test(self, token, wme):
		"""
		:type token: Token
		:type wme: WME
		"""
		for this_test in self.tests:
			wme2 = token.wmes[this_test.condition_number_of_arg2]
			arg2 = wme2[this_test.field_of_arg2]
			if not this_test.field_of_arg1.evaluate(Context(
				wme.context, {this_test.var_of_arg2: arg2}
			)):
				return False
		return True


class NccNode(BetaNode):
	kind = "ncc"  # negated conjunctive condition/conjunctive negations

	def __init__(self, children=None, parent=None, items=None, partner=None):
		"""
		:type partner: NccPartnerNode
		:type items: list of Token
		"""
		super(NccNode, self).__init__(children=children, parent=parent)
		self.items = items if items else []
		self.partner = partner

	def left_activation(self, token, wme, binding=None):
		"""
		:type wme: WME
		:type token: Token
		:type binding: dict
		"""
		new_token = Token(token, wme, self, binding)
		self.items.append(new_token)
		for result in self.partner.new_result_buffer:
			self.partner.new_result_buffer.remove(result)
			new_token.ncc_results.append(result)
			result.owner = new_token
		if not new_token.ncc_results:
			for child in self.children:
				child.left_activation(new_token, None)


class NccPartnerNode(BetaNode):
	kind = "ncc-partner"

	def __init__(
		self, children=None, parent=None, ncc_node=None,
		number_of_conditions=0, new_result_buffer=None
	):
		"""
		:type new_result_buffer: list of Token
		:type ncc_node: NccNode
		"""
		super(NccPartnerNode, self).__init__(children=children, parent=parent)
		self.ncc_node = ncc_node
		self.number_of_conditions = number_of_conditions
		self.new_result_buffer = new_result_buffer if new_result_buffer else []

	def left_activation(self, token, wme, binding=None):
		"""
		:type wme: WME
		:type token: Token
		:type binding: dict
		"""
		new_result = Token(token, wme, self, binding)
		owners_token = token
		owners_wme = wme
		for i in range(self.number_of_conditions):
			owners_wme = owners_token.wme
			owners_token = owners_token.parent
		for token in self.ncc_node.items:
			if token.parent == owners_token and token.wme == owners_wme:
				token.ncc_results.append(new_result)
				new_result.owner = token
				Token.delete_token_and_descendents(token)
		self.new_result_buffer.append(new_result)


class PNode(BetaNode):
	"""
	Terminal nodes are used to indicate a single rule having matched all its conditions;
	at this point we say the rule has a full match.
	A rule with an 'or' conditional disjunctive connective results in subrule generation for each possible logically branch;
	thus one rule can have multiple terminal nodes.
	"""
	kind = 'p'

	def __init__(self, children=None, parent=None, items=None, **kwargs):
		"""
		:type items: list of Token
		"""
		super(PNode, self).__init__(children=children, parent=parent)
		self.items = items if items else []
		for k, v in kwargs.items():
			setattr(self, k, v)

	def left_activation(self, token, wme, binding=None):
		"""
		:type wme: WME
		:type token: Token
		:type binding: dict
		"""
		new_token = Token(token, wme, node=self, binding=binding)
		self.items.append(new_token)

		log.info("matched all the conditions of a rule, execute rhs.")
		self.execute(new_token)

	def execute(self, token):
		wme = next((_wme for _wme in token.wmes if _wme), None)
		wme.session.matches.append(self)
