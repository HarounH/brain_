"""
File to process ward agglomerative into a tree
"""

import os
import pickle
import numpy as np
import torch
import heapq
from collections import defaultdict
import scipy.sparse as scsp
import torch.sparse as tsp


class WardNode():
    def __init__(self, val):
        self.val = val
        self.children = []
        self.parent = None
        self.exist = False

    def __lt__(self, other):
        if self.depth == other.depth:
            return self.val < other.val
        else:
            return self.depth > other.depth


class WardTree():
    def __init__(self, ward):
        if isinstance(ward, str):
            with open(ward, "rb") as f:
                ward = pickle.load(ward)
        self.ward = ward  # Dont need to store this.
        self.n_samples = ward.children_.shape[0] + 1
        self.node_idxs = list(range(2 * self.n_samples))
        self.nodes = {val: WardNode(val) for val in self.node_idxs}

        self.construct_tree()

    def construct_tree(self):
        for i in range(self.ward.children_.shape[0]):
            cidx0, cidx1 = self.ward.children_[i]
            pidx = self.n_samples + i
            assert(len(self.nodes[pidx].children) == 0)
            assert(self.nodes[cidx0].parent is None)
            assert(self.nodes[cidx1].parent is None)

            self.nodes[pidx].exist = True
            self.nodes[cidx0].exist = True
            self.nodes[cidx1].exist = True
            self.nodes[pidx].children.extend([self.nodes[cidx0], self.nodes[cidx1]])
            self.nodes[cidx0].parent = self.nodes[pidx]
            self.nodes[cidx1].parent = self.nodes[pidx]

        self.node_idxs = [k for k, val in self.nodes.items() if val.exist]
        self.nodes = {k: self.nodes[k] for k in self.node_idxs}

        self.roots = [val for k, val in self.nodes.items() if val.parent is None]
        # Write out depths
        level = [val for k, val in self.nodes.items() if val.parent is None]
        current_depth = 0
        while len(level) > 0:
            new_level = []
            for node in level:
                node.depth = current_depth
                new_level.extend(node.children)
            level = new_level
            current_depth += 1

    def get_leaves(self):
        # Linear search is okay - graph is linear in #nodes=#edges
        return [self.nodes[k] for k in self.node_idxs if (len(self.nodes[k].children) == 0)]

    def get_depth(self, d):
        # Linear search is okay really.
        return [self.nodes[k] for k in self.node_idxs if (self.nodes[k].depth == d)]


def get_adjacency_list(nodes_from, nodes_to):
    # adjacency list is from position in nodes_from to position in nodes_to
    to_vals = [node.val for node in nodes_to]
    adj_list = []
    for start_node in nodes_from:
        adjacency = []
        stack = [start_node]
        while len(stack) > 0:
            next_node = stack.pop()
            try:
                to_idx = to_vals.index(next_node.val)
                adjacency.append(to_idx)
            except ValueError:
                pass
            stack.extend(next_node.children)
        adj_list.append(adjacency)
    return adj_list


def go_up_to_reduce(from_nodes, level_size, degree_normalized=True):
    # from_nodes (WardNode list)
    # Goes up from from_nodes in a breadth first fashion until the size of the level is n
    base_val_to_idx = {node.val: i for i, node in enumerate(from_nodes)}

    # INVAR 0: Everything that is in the level,
    #   has adjacencies to the base stored in the variable adjacencies2base
    adjacencies2base = {node.val: [node.val] for node in from_nodes}

    # nodes_pq is the level that will be output once its cut down to the right size.
    # INVAR 1: nodes_pq[i] is at least as deep as nodes_pq[i+1]
    nodes_pq = [nd for nd in from_nodes]
    heapq.heapify(nodes_pq)  # Ensuring INVAR 1
    while(len(nodes_pq) > level_size):
        old_node = heapq.heappop(nodes_pq)
        # We look at old_nodes parent.
        new_node = old_node.parent
        if new_node is not None:
            # print("popped {} to get {}".format(old_node.val, new_node.val))
            # 2 cases - either its in the level or its not.
            # if its in the level, then we should just modify its adjacencies
            if new_node.val in adjacencies2base.keys():
                adjacencies2base[new_node.val].extend(adjacencies2base[old_node.val])
            else:  # Replacing old node by new_node essentially.
                heapq.heappush(nodes_pq, new_node)
                adjacencies2base[new_node.val] = adjacencies2base[old_node.val]

            # base case: old_node was in the original base level.
            if old_node.val in base_val_to_idx:
                adjacencies2base[new_node.val].append(old_node.val)
            adjacencies2base[new_node.val] = list(set(adjacencies2base[new_node.val]))
            del adjacencies2base[old_node.val]  # old_node is no longer in the level
        else:
            continue
    out_nodes = nodes_pq
    in_nodes = from_nodes
    data = []
    rows = []
    cols = []
    for ridx, out_node in enumerate(out_nodes):
        for adj_node_val in adjacencies2base[out_node.val]:
            cidx = base_val_to_idx[adj_node_val]
            data.append(1.0 / len(adjacencies2base[out_node.val]) if degree_normalized else 1.0)
            rows.append(ridx)
            cols.append(cidx)
    adj_matrix = scsp.coo_matrix((data, (rows, cols)), shape=(len(out_nodes), len(in_nodes)))
    return out_nodes, adj_matrix
