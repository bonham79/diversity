"""
Graph and Node classes to abstract steps in subset creation process.
"""
from __future__ import annotations
import heapq
import numpy as np
from typing import Set, List, Tuple, Any


class VertexOrderError(Exception):
	"""A vertex of idx < current state value was passed."""


class Graph:
	"""
	Graph abstraction of maximum-indepenent subset problem. Given boolean
		array of connected nodes, instantiates all nodes in array with information
		regarding their dependencies.
	"""
	def __init__(self, matrix: np.array) -> None:
		"""
		Instantiates graph array. 
		
		This entails intantiating all nodes as indepdent subsets of
		size 1 (themselves) with information regarding their dependencies. Simultaneously,
		it stores all pairwise edges (i,j) in ascending order (i < j) to avoid duplicates.

		Args:
			matrix (np.array): Boolean matrix indicating pairwise edges between
				vertices.
		"""

		self.n = len(matrix)
		self.is_edge, self.not_edge = self._init_edge(matrix)
		self.indep, self.depen, self.explore = self._init_vertex(matrix)

	def _init_edge(self, matrix: np.array) -> Tuple[List[Set]]:
		"""
		Instantiates edges and their complements for graph. 

		For easier lookup of exclusion and inclusion sets. Also imposes an ordering
		on vertexes so that for e = (i,j) i<j. This allows for simplification of
		graph expansions.

		Args:
			matrix (np.array): Boolean matrix indicating pairwise edges between
				vertices.

		Returns:
			Tuple[List[Set], List[Set]] with element [0] indicating nodes connected to
				node idx and element [1] containing the respective complement set. 
		"""

		n = len(matrix)
		is_edge = [set() for _ in range(n)]
		not_edge = [set() for _ in range(n)]
		for val in range(n - 1): # Skip last index since ascending order.
			# Takes bools of connected entries. Reduces to only val onward. This eliminates duplicate edges.
			connected = matrix[val][val:]
			#Take complements.
			not_connected = np.invert(connected)
			# Gets indexes
			connected = connected.nonzero()[0]
			not_connected = not_connected.nonzero()[0]
			# Indexes were shifted from slice. This restores them relative to val.
			connected = connected + val
			not_connected = not_connected + val
			# Convert to set.
			is_edge[val] = set(connected) 
			not_edge[val] = set(not_connected)
		return is_edge, not_edge

	def _init_vertex(self, matrix: np.array) -> Tuple[List[Set]]:
		"""
		Instantiates universal set of vertices for graph.

		In particular, it uses the boolean matrix to specify vertexes
			that are universally independent, dependent, and unknown for the MISP
			problem. This allows sharing of universal information across the graph
			during node exploration. (Also stops needing to move the matrix around.)

		Args:
			matrix (np.array): Boolean matrix indicating pairwise edges between
				vertices.

		Returns:
			Tuple[List[Set], List[Set], List[Set]] with element [0] indicating nodes
				universally independent, element [1] being nodes universally dependent,
				and element [2] being of unknown status.
		"""
		n = len(matrix)
		indep, depen, unk = set(), set(), set()
		for idx in range(n):
			count = sum(matrix[idx])
			if count == 1: # All nodes connected to self.
				indep.add(idx)
			elif count == n: 
				depen.add(idx)
			else:
				unk.add(idx)
		return indep, depen, unk

	def is_independent_subset(self, subset: Set[int]) -> Tuple[Bool, int]:
		"""
		Checks independence of vertices contained in subset.

		Args:
			subset (Set[int]): subset containing vertices in Graph

		Returns:
			Tuple[Bool, int] with bool indicating independence and int
				indicating number of violations.
		"""
		indep = True
		dependencies = set()
		for v1 in subset:
			connected = self.is_edge[v1]
			for v2 in connected:
				if v2 in subset and v2 != v1: # Since value is connected to self.
					indep = False
					dependencies.add((v1,v2))
		return indep, len(dependencies)

class NodeState:
	"""
	Abstraction of state while compiling independent vertices in graph.

	Each NodeState tracks membership in a given subset, along with permitted
		descendants of subset, nodes in dependent relation to subset, and degree
		of dependency.
	"""
	parent: NodeState
	graph: Graph

	def __init__(self, val: int, parent = None, graph = None):
		"""
		Initializes node, either as descendant of a parent node or from an origin graph.

		Using parent or graph, instantiates properties relavant for independent subset exploration.
			
		Args:
			val (int): value of node (which vertex in graph ordering).
			parent (NodeState): Parent node to inherit properties from. Default: None.
			graph (Graph): Graph that is origin of node. Only necessary for root nodes. Default: None.
		"""
		self.val = val
		if parent is not None and graph is None:
			self.graph = parent.graph
			self.subset = parent.subset | {val}
			self.exclude = parent.exclude | self.graph.is_edge[val]
		elif graph is not None and parent is None:
			self.graph = graph
			self.subset = graph.indep | {val}
			self.exclude =  graph.depen | self.graph.is_edge[val]
		else:
			raise ValueError("Node requires exclusively either a Graph or parent NodeState to instantiate.")
		# Maintain degree information.
		self.degree = len(self.exclude)
		self.bound = self.graph.n - self.degree 
		# We keep the children values without initializing for easy lookup.
		self.children = self.graph.not_edge[val] - self.exclude - self.subset


	def get_children(self) -> List:
		"""
		Return possible descendent states from this node.
		"""
		children = []
		for child in self.children:
			children.append(NodeState(child, parent=self))
		return children

	def __str__(self) -> str:
		return f"Node: {self.val} of subset {self.subset} dependencies with {self.exclude}"

	def __repr__(self) -> str:
		return self.val, self.subset

	def __lt__(self, other):
		# Used for ordering in queue. Size then degree to break ties.
		if self.size == other.size:
			return self.degree < other.degree
		return self.size > other.size

	@property
	def size(self):
		return len(self.subset)

class MispEnumQueue:
	"""
	Wrapper class to manage the queue for enumerating MISP nodes.
	"""
	def __init__(self, graph: Graph):
		"""
		Initializes queue for enumeration through MISP solutions for a given graph.

		Args:
			graph (Graph): Graph object to solve for MISP.
		"""
		start_heap = [NodeState(val, graph=graph) for val in graph.explore]
		heapq.heapify(start_heap)
		self.heap = start_heap

	def update(self) -> NodeState:
		node = self.pop()
		children = node.get_children()
		if children:
			[heapq.heappush(self.heap, child) for child in children]
		return node

	def peek(self) -> NodeState:
		return self.heap[0]

	def pop(self) -> NodeState:
		return heapq.heappop(self.heap)

	@property
	def	has_entry(self):
		return bool(self.heap)

# Unit tests
if __name__=="__main__":
    d_matrix = [
        [0,2,4,1],
        [2,0,4,3],
        [4,4,0,3],
        [1,3,3,0],
    ]
    
    print("Testing Graph Class")
    d_matrix = np.array(d_matrix) < 3
    graph = Graph(d_matrix)

    assert graph.n == 4
    print("Edges:", graph.is_edge, "Disjoint:", graph.not_edge)
    print("Independent:", graph.indep, "Dependent:", graph.depen, "Unknown:", graph.explore)

    print("\nTesting NodeState")
    starter = [NodeState(i, graph=graph) for i in range(graph.n)]
    [print(a) for a in starter]

    print("\nTesting Children")
    print(starter[0])
    print([child if child else print("None") for child in starter[0].get_children()])

    print("\nTesting Queue")
    q = MispEnumQueue(graph)
    [print(node, node.degree) for node in q.heap]
    n = q.update()
    print("\nPopped:", n)
    [print(node, node.degree) for node in q.heap]