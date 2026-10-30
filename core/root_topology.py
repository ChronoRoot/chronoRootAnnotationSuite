"""RSML roots as ordered chains of existing graph edges.

A graph edge is one polyline. A root is a list of signed edge ids: the sign
selects the direction. The same edge id may belong to more than one root.
Edits stay in memory; Export is what writes the RSML file.
"""

import hashlib
import os
from xml.etree import ElementTree as ET

import numpy as np

from core.root_graph_builder import assign_edge_polylines, createGraph, extract_skeleton, graphInit
from core.rsml import createHeader, createTree

class TopologyError(Exception):
    """An edit that the current graph does not allow."""


class Topology:
    def __init__(self, edges, roots, main_id, available, prune=5, main_spine=None):
        self.edges = edges
        self.roots = roots
        self.main_id = main_id
        self.available = available
        self.prune = prune
        spine = main_spine
        if spine is None and main_id in roots:
            spine = roots[main_id]["edges"]
        self.main_spine = list(spine or [])

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def signature(self):
        payload = [
            (edge_id, self.edges[edge_id].u, self.edges[edge_id].v, len(self.edges[edge_id].points))
            for edge_id in sorted(self.edges)
        ]
        return hashlib.sha1(repr(payload).encode()).hexdigest()[:12]

    def edge_ends(self, signed):
        edge = self.edges[abs(signed)]
        if signed > 0:
            return edge.u, edge.v
        return edge.v, edge.u

    def chain_points(self, signed_edges):
        points = []
        for signed in signed_edges:
            edge = self.edges[abs(signed)]
            seq = edge.points if signed > 0 else list(reversed(edge.points))
            if points and seq and points[-1] == seq[0]:
                seq = seq[1:]
            points.extend(seq)
        return points

    def chain_nodes(self, signed_edges):
        if not signed_edges:
            return []
        start, end = self.edge_ends(signed_edges[0])
        nodes = [start]
        for signed in signed_edges:
            _start, end = self.edge_ends(signed)
            nodes.append(end)
        return nodes

    def root_points(self, root_id):
        return self.chain_points(self.roots[root_id]["edges"])

    def order_of(self, root_id):
        if root_id not in self.roots or self.roots[root_id].get("available"):
            return None
        order = 0
        seen = set()
        current = root_id
        while self.roots[current]["parent"] is not None:
            parent = self.roots[current]["parent"]
            if parent in seen or parent not in self.roots:
                return None
            seen.add(current)
            current = parent
            order += 1
        return order if current == self.main_id else None

    def ownership(self, edge_id):
        count = 0
        for root in self.roots.values():
            count += sum(abs(signed) == edge_id for signed in root["edges"])
        for chain in self.available:
            count += sum(abs(signed) == edge_id for signed in chain)
        return count

    def chain_boundaries(self, signed_edges):
        """Points between consecutive edges. The index is where the suffix starts."""
        points = []
        for index in range(1, len(signed_edges)):
            _start, end = self.edge_ends(signed_edges[index - 1])
            points.append((index, end))
        return points

    def boundaries(self, root_id):
        return self.chain_boundaries(self.roots[root_id]["edges"])

    def can_apply(self):
        if self._main_is_open():
            trial = self.clone()
            trial.reconnect_main()
            return trial._ready()
        return self._ready()

    def _ready(self):
        if self.main_id not in self.roots or self.available:
            return False
        owned = set()
        for root in self.roots.values():
            owned.update(abs(signed) for signed in root["edges"])
        return owned == set(self.edges)

    def _main_is_open(self):
        if self.main_id not in self.roots or not self.main_spine:
            return False
        current = self.roots[self.main_id]["edges"]
        spine = self.main_spine
        return not (
            len(current) >= len(spine)
            and current[:len(spine)] == spine
            and _links(self, current)
        )

    def _rewrite_spine(self, signed, replacement):
        updated = []
        for item in self.main_spine:
            if item == signed:
                updated.extend(replacement)
            elif item == -signed:
                updated.extend(-piece for piece in reversed(replacement))
            else:
                updated.append(item)
        return updated

    def reconnect_main(self):
        """Put the original main-root segments back. Copies used by laterals stay."""
        if self.main_id not in self.roots or not self.main_spine or not self._main_is_open():
            return
        self.roots[self.main_id]["edges"] = list(self.main_spine)
        self.roots[self.main_id]["parent"] = None
        self._release_detached(self.main_id)

    def status_text(self):
        parts = []
        if getattr(self, "left_behind", 0):
            parts.append(f"{self.left_behind} skeleton pixels are not on any root.")
        if self.available:
            noun = "segment" if len(self.available) == 1 else "segments"
            parts.append(f"{len(self.available)} {noun} can be connected.")
        if self._main_is_open():
            parts.append("Apply puts the main root back together.")
        if parts:
            text = " ".join(parts)
            if self.available or self._main_is_open():
                text += " Copied segments stay on the main root too."
            return text
        if not self._ready():
            return "Every graph edge must belong to a root before this can be applied."
        return "Ready to apply. Nothing is written until Export."

    def split(self, root_id, boundary_index):
        if root_id == self.main_id:
            raise TopologyError("You cannot split the main root.")
        root = self.roots[root_id]
        if boundary_index <= 0 or boundary_index >= len(root["edges"]):
            raise TopologyError("Choose a join between two segments.")
        suffix = list(root["edges"][boundary_index:])
        root["edges"] = root["edges"][:boundary_index]
        if not root["edges"] or not suffix:
            raise TopologyError("That split does not leave two segments.")
        # The tip side becomes a child of the piece that stays, so a split
        # lateral gains a tertiary instead of an unused segment.
        child_id = (max(self.roots) + 1) if self.roots else 1
        self.roots[child_id] = {"parent": root_id, "edges": suffix, "available": False}
        prefix_nodes = set(self.chain_nodes(root["edges"]))
        for other_id, other in list(self.roots.items()):
            if other_id in (root_id, child_id) or other["parent"] != root_id or not other["edges"]:
                continue
            if self.edge_ends(other["edges"][0])[0] not in prefix_nodes:
                other["parent"] = child_id

    def connect(self, target_id, source_id):
        if target_id == source_id or target_id not in self._chains() or source_id not in self._chains():
            raise TopologyError("Select two different roots or available segments.")
        target = self._chain(target_id)
        source = self._chain(source_id)
        if not target or not source:
            raise TopologyError("A selected chain is empty.")
        # A copied segment may be selected first. The root keeps its direction.
        if isinstance(target_id, str) and not isinstance(source_id, str):
            target_id, source_id = source_id, target_id
            target, source = source, target
        if isinstance(target_id, str) and isinstance(source_id, str):
            return self._join_available(target_id, source_id, target, source)
        if (
            not isinstance(target_id, str)
            and not isinstance(source_id, str)
            and (target_id == self.main_id or source_id == self.main_id)
        ):
            raise TopologyError("Make a copy to be able to connect.")
        owned_by_target = {abs(signed) for signed in target}
        if not isinstance(target_id, str) and source and all(abs(signed) in owned_by_target for signed in source):
            raise TopologyError(
                "That segment is already part of the first root. "
                "Select the lateral first, then this segment, to add the copy to the lateral."
            )
        begin = self.edge_ends(target[0])[0]
        end = self.edge_ends(target[-1])[1]
        source_begin = self.edge_ends(source[0])[0]
        source_end = self.edge_ends(source[-1])[1]
        candidates, valid = self._join_candidates(target, source)
        if len(valid) > 1:
            raise TopologyError(
                "These two meet in more than one direction. Reverse one of them to choose which way to merge."
            )
        if len(valid) == 1:
            merged = valid[0]
            self._write_chain(target_id, merged)
            self._consume(source_id, target_id)
            if not isinstance(target_id, str):
                self._promote_if_lateral(target_id)
            if merged[:len(target)] == list(target):
                return "Added to the end."
            return "Added in front of the beginning. The previous direction was kept."
        if candidates:
            raise TopologyError(
                "That connection turns back along the same root and would make a loop. It was not applied."
            )
        target_nodes = self.chain_nodes(target)
        meeting = source_begin if source_begin in target_nodes[1:-1] else None
        if meeting is None and source_end in target_nodes[1:-1]:
            meeting = source_end
        if meeting is None:
            raise TopologyError(self._untouched_message(target, source))
        if isinstance(target_id, str):
            raise TopologyError(
                "This segment meets the other in the middle. "
                "Select the root as the first click if you want a branch."
            )
        if self._is_ancestor(source_id, target_id):
            raise TopologyError("That connection would make a cycle. It was not applied.")
        oriented = source if source_begin == meeting else _reversed(source)
        self._write_chain(source_id, oriented)
        self._attach(source_id, target_id)
        return f"They meet in the middle, so this became a branch. It begins at {meeting}."

    def split_available(self, index, boundary_index):
        chain = self.available[index]
        if boundary_index <= 0 or boundary_index >= len(chain):
            raise TopologyError("Choose a join between two segments.")
        suffix = chain[boundary_index:]
        self.available[index] = chain[:boundary_index]
        self.available.append(suffix)

    def discard_available(self, index):
        chain = self.available[index]
        if any(self.ownership(abs(signed)) < 2 for signed in chain):
            raise TopologyError("This is the last copy of that segment.")
        del self.available[index]

    def _begins_on_main(self, signed_edges):
        if not signed_edges or self.main_id not in self.roots:
            return False
        begin = self.edge_ends(signed_edges[0])[0]
        return begin in set(self.chain_nodes(self.roots[self.main_id]["edges"]))

    def _promote_if_lateral(self, root_id):
        if root_id in self.roots and root_id != self.main_id and self._begins_on_main(self.roots[root_id]["edges"]):
            self.roots[root_id]["parent"] = self.main_id

    def copy_root(self, root_id):
        """Copy a whole root so a second lateral can use the same segments."""
        edges = list(self.roots[root_id]["edges"])
        if not edges:
            raise TopologyError("Select one root first.")
        if edges in self.available or _reversed(edges) in self.available:
            raise TopologyError("That root is already duplicated.")
        self.available.append(edges)
        return None

    def cut_polyline(self, root_id, position, point):
        """Cut one segment at the vertex nearest to point.

        The two pieces replace that segment on this root only, including the
        main root, which stays one connected chain. Another root that shares
        the segment keeps the original. Returns the index of the shorter piece.
        """
        if root_id not in self.roots:
            raise TopologyError("Click a point on a root.")
        chain = self.roots[root_id]["edges"]
        if position < 0 or position >= len(chain):
            raise TopologyError("Click a point along the lateral segment, then Split.")
        signed = chain[position]
        edge = self.edges[abs(signed)]
        directed = list(edge.points if signed > 0 else reversed(edge.points))
        if len(directed) < 3:
            raise TopologyError("This segment is too short to split.")
        best = min(
            range(1, len(directed) - 1),
            key=lambda index: (directed[index][0] - point[0]) ** 2 + (directed[index][1] - point[1]) ** 2,
        )
        if signed > 0:
            left_points = list(edge.points[:best + 1])
            right_points = list(edge.points[best:])
        else:
            # best is an index along the reversed polyline.
            cut = len(edge.points) - 1 - best
            left_points = list(edge.points[:cut + 1])
            right_points = list(edge.points[cut:])
        if len(left_points) < 2 or len(right_points) < 2:
            raise TopologyError("Click a point along the segment, not an end.")
        first_id = (max(self.edges) + 1) if self.edges else 1
        second_id = first_id + 1
        self.edges[first_id] = _Edge(left_points[0], left_points[-1], left_points, edge.root_type)
        self.edges[second_id] = _Edge(right_points[0], right_points[-1], right_points, edge.root_type)
        if signed > 0:
            replacement = [first_id, second_id]
        else:
            replacement = [-second_id, -first_id]
        chain[position:position + 1] = replacement
        # The saved main chain must follow a split of the main. Leaving the old
        # edge id there makes Keep fixes look up an edge that was just deleted.
        if root_id == self.main_id or (
            self.ownership(abs(signed)) == 0
            and any(abs(item) == abs(signed) for item in self.main_spine)
        ):
            self.main_spine = self._rewrite_spine(signed, replacement)
        if self.ownership(abs(signed)) == 0:
            del self.edges[abs(signed)]
        left_length = polyline_length(self.edges[abs(replacement[0])].points)
        right_length = polyline_length(self.edges[abs(replacement[1])].points)
        return position if left_length <= right_length else position + 1

    def copy_segment(self, root_id, position):
        """Keep this segment on its root and also offer it for another lateral."""
        edges = self.roots[root_id]["edges"]
        if position < 0 or position >= len(edges):
            raise TopologyError("Select one segment first.")
        segment = [edges[position]]
        if segment in self.available or [-segment[0]] in self.available:
            raise TopologyError("That segment is already available to connect.")
        self.available.append(segment)
        return None

    def reuse(self, root_id, position):
        edges = self.roots[root_id]["edges"]
        if position < 0 or position >= len(edges):
            raise TopologyError("Select one segment of the root.")
        self.available.append([edges[position]])

    def remove_copy(self, root_id, position):
        edges = self.roots[root_id]["edges"]
        if position < 0 or position >= len(edges):
            raise TopologyError("Select one segment of the root.")
        edge_id = abs(edges[position])
        if self.ownership(edge_id) < 2:
            raise TopologyError("This is the last copy of that segment.")
        if len(edges) == 1:
            if root_id == self.main_id:
                raise TopologyError("The main root needs at least one segment.")
            self._delete_root(root_id)
            return
        if position not in (0, len(edges) - 1):
            raise TopologyError("Split on either side of an interior segment before removing it.")
        del edges[position]
        self._release_detached(root_id)

    def to_record(self):
        return {
            "prune": self.prune,
            "graph_hash": self.signature(),
            "main": self.main_id,
            "roots": [
                {"id": root_id, "parent": root["parent"], "edges": list(root["edges"])}
                for root_id, root in self.roots.items()
            ],
            "available": [list(chain) for chain in self.available],
            "main_spine": list(self.main_spine),
            "edges": {
                str(edge_id): {
                    "u": [int(edge.u[0]), int(edge.u[1])],
                    "v": [int(edge.v[0]), int(edge.v[1])],
                    "points": [[int(point[0]), int(point[1])] for point in edge.points],
                    "root_type": None if edge.root_type is None else int(edge.root_type),
                }
                for edge_id, edge in self.edges.items()
            },
        }

    @classmethod
    def from_record(cls, edges, record):
        roots = {
            item["id"]: {"parent": item["parent"], "edges": list(item["edges"]), "available": False}
            for item in record["roots"]
        }
        available = [list(chain) for chain in record["available"]]
        needed = {abs(signed) for root in roots.values() for signed in root["edges"]}
        needed.update(abs(signed) for chain in available for signed in chain)
        if not needed.issubset(edges):
            raise ValueError("The topology refers to an edge that is not in this graph.")
        topology = cls(
            edges, roots, record["main"], available, record.get("prune", 5),
            main_spine=record.get("main_spine"),
        )
        return topology

    @classmethod
    def from_saved(cls, edges, record):
        """Restore a kept tracing when its edges are still this skeleton."""
        if not record:
            return None
        try:
            topology = cls.from_record(edges, record)
        except (KeyError, TypeError, ValueError):
            return None
        if topology.signature() != record.get("graph_hash"):
            return None
        return topology

    def _chains(self):
        ids = set(self.roots)
        ids.update(f"a{index}" for index in range(len(self.available)))
        return ids

    def _chain(self, chain_id):
        if isinstance(chain_id, str) and chain_id.startswith("a"):
            return self.available[int(chain_id[1:])]
        return self.roots[chain_id]["edges"]

    def _write_chain(self, chain_id, edges):
        if isinstance(chain_id, str) and chain_id.startswith("a"):
            self.available[int(chain_id[1:])] = edges
        else:
            self.roots[chain_id]["edges"] = edges

    def _join_candidates(self, target, source):
        begin = self.edge_ends(target[0])[0]
        end = self.edge_ends(target[-1])[1]
        source_begin = self.edge_ends(source[0])[0]
        source_end = self.edge_ends(source[-1])[1]
        candidates = []
        if source_begin == end:
            candidates.append(target + source)
        if source_end == end and source_end != source_begin:
            candidates.append(target + _reversed(source))
        if source_end == begin:
            candidates.append(source + target)
        if source_begin == begin and source_begin != source_end:
            candidates.append(_reversed(source) + target)
        valid = []
        seen = set()
        for merged in candidates:
            key = tuple(merged)
            if key in seen:
                continue
            seen.add(key)
            nodes = self.chain_nodes(merged)
            if _links(self, merged) and len(nodes) == len(set(nodes)):
                valid.append(merged)
        return candidates, valid

    def _join_available(self, target_id, source_id, target, source):
        _candidates, valid = self._join_candidates(target, source)
        if len(valid) > 1:
            raise TopologyError(
                "These two meet in more than one direction. Reverse one of them to choose which way to merge."
            )
        if len(valid) != 1:
            if _candidates:
                raise TopologyError(
                    "That connection turns back along the same root and would make a loop. It was not applied."
                )
            raise TopologyError(self._untouched_message(target, source))
        source_index = int(source_id[1:])
        target_index = int(target_id[1:])
        self.available.pop(source_index)
        if source_index < target_index:
            target_index -= 1
        self.available[target_index] = valid[0]
        return "The two copies were joined."

    def _consume(self, source_id, target_id):
        """Append already written onto target. Drop source and reparent its children."""
        if isinstance(source_id, str):
            del self.available[int(source_id[1:])]
            return
        source_parent = self.roots[source_id]["parent"]
        for root_id, root in self.roots.items():
            if root["parent"] != source_id:
                continue
            # The root we extended may itself have been a child of the source.
            root["parent"] = source_parent if root_id == target_id else target_id
        del self.roots[source_id]

    def _attach(self, source_id, target_id):
        if isinstance(source_id, str):
            edges = self.available.pop(int(source_id[1:]))
            root_id = max(self.roots) + 1 if self.roots else 1
            self.roots[root_id] = {"parent": target_id, "edges": edges, "available": False}
            return
        self.roots[source_id]["parent"] = target_id
        self.roots[source_id]["available"] = False

    def _release_detached(self, root_id):
        if root_id not in self.roots:
            return
        nodes = set(self.chain_nodes(self.roots[root_id]["edges"]))
        displaced = []
        for other_id, other in self.roots.items():
            if other["parent"] != root_id or not other["edges"]:
                continue
            child_nodes = set(self.chain_nodes(other["edges"]))
            if self.edge_ends(other["edges"][0])[0] not in nodes and not child_nodes & nodes:
                displaced.append(other_id)
        for other_id in displaced:
            self._release(other_id)

    def _release(self, root_id):
        """Move a root and its descendants into the available pool."""
        children = [rid for rid, root in self.roots.items() if root["parent"] == root_id]
        for child_id in children:
            self._release(child_id)
        if root_id in self.roots:
            self.available.append(self.roots[root_id]["edges"])
            del self.roots[root_id]

    def _delete_root(self, root_id):
        children = [rid for rid, root in self.roots.items() if root["parent"] == root_id]
        for child_id in children:
            self._release(child_id)
        self.roots.pop(root_id, None)

    def node_degree(self, node):
        return sum(edge.u == node or edge.v == node for edge in self.edges.values())

    def can_reverse(self, chain_id):
        """Only a root that begins and ends at a branch, with no free tip."""
        if chain_id not in self._chains():
            return False
        edges = self._chain(chain_id)
        if not edges:
            return False
        begin = self.edge_ends(edges[0])[0]
        end = self.edge_ends(edges[-1])[1]
        return self.node_degree(begin) >= 3 and self.node_degree(end) >= 3

    def reverse_chain(self, chain_id):
        if chain_id not in self._chains():
            raise TopologyError("Select one root or segment to reverse.")
        if not self.can_reverse(chain_id):
            raise TopologyError(
                "This root has a tip, so its direction stays from the branch toward the tip. "
                "Reverse is only for a root that begins and ends at a branching node."
            )
        self._write_chain(chain_id, _reversed(self._chain(chain_id)))
        return f"Reversed. {self.direction_text(chain_id)}"

    def direction_text(self, chain_id):
        edges = self._chain(chain_id)
        if not edges:
            return "This selection is empty."
        begin = self.edge_ends(edges[0])[0]
        end = self.edge_ends(edges[-1])[1]
        return f"Begins at {begin[0]}, {begin[1]} and ends at {end[0]}, {end[1]}."

    def _reject_loop(self, signed_edges):
        nodes = self.chain_nodes(signed_edges)
        if len(nodes) != len(set(nodes)):
            raise TopologyError(
                "That connection turns back along the same root and would make a loop. "
                "It was not applied. Reverse the segment if you need the other end."
            )

    def _untouched_message(self, target, source):
        return (
            "These two do not touch, so they were not connected. "
            "Split the segment of the main root that meets this lateral, then connect that segment."
        )

    def _is_ancestor(self, maybe_ancestor, node_id):
        seen = set()
        current = node_id
        while current in self.roots and current not in seen:
            if current == maybe_ancestor:
                return True
            seen.add(current)
            current = self.roots[current]["parent"]
        return False


def _reversed(signed_edges):
    return [-signed for signed in reversed(signed_edges)]


def _links(topology, signed_edges):
    for left, right in zip(signed_edges, signed_edges[1:]):
        if topology.edge_ends(left)[1] != topology.edge_ends(right)[0]:
            return False
    return True


def _point(value):
    return (int(round(float(value[0]))), int(round(float(value[1]))))


def edges_from_graph(graph):
    edges = {}
    for _u, _v, data in graph.edges(data=True):
        points = [_point(point) for point in data.get("points") or []]
        if len(points) < 2:
            continue
        edges[int(data["edge_id"])] = _Edge(
            points[0], points[-1], points, data.get("root_type"), data.get("orig_root_type"),
        )
    merged, _mapping = _merge_unbranched_edges(edges)
    return merged


def _merge_unbranched_edges(edges):
    """Join edges through a node that is not a branch or a tip.

    A segment then runs from one bifurcation to the next, instead of being
    broken into pixel-long pieces. The map sends each input edge id to the
    signed id of the polyline that contains it.
    """
    edges = dict(edges)
    parts = {edge_id: [(edge_id, 1)] for edge_id in edges}
    next_id = (max(edges) + 1) if edges else 1
    merged_any = False
    skipped = set()
    while True:
        incident = {}
        for edge_id, edge in edges.items():
            if edge.u == edge.v:
                continue
            incident.setdefault(edge.u, []).append(edge_id)
            incident.setdefault(edge.v, []).append(edge_id)
        node = None
        for point, ids in incident.items():
            if point in skipped or len(ids) != 2 or ids[0] == ids[1]:
                continue
            if not _same_tissue(edges[ids[0]], edges[ids[1]]):
                # A class change on a straight root is not a branch. Joining
                # those edges would hand one side of the color change to the other.
                skipped.add(point)
                continue
            node = point
            break
        if node is None:
            break
        left_id, right_id = incident[node]
        left, right = edges[left_id], edges[right_id]
        # +1 when the edge's own start→end direction follows the joined polyline.
        left_sign = 1 if left.v == node else -1
        right_sign = 1 if right.u == node else -1
        points = _join_at(left, right, node)
        if len(points) < 2:
            break
        del edges[left_id]
        del edges[right_id]
        edges[next_id] = _Edge(
            points[0], points[-1], points, _joined_tissue(left, right), left.orig_root_type,
        )
        parts[next_id] = (
            [(old, sign * left_sign) for old, sign in parts.pop(left_id)]
            + [(old, sign * right_sign) for old, sign in parts.pop(right_id)]
        )
        next_id += 1
        merged_any = True
    if not merged_any:
        return edges, {edge_id: edge_id for edge_id in edges}
    ordered = sorted(edges, key=lambda edge_id: (
        edges[edge_id].u[1], edges[edge_id].u[0], edges[edge_id].v[1], edges[edge_id].v[0],
    ))
    new_edges = {}
    mapping = {}
    for index, edge_id in enumerate(ordered, start=1):
        new_edges[index] = edges[edge_id]
        for old, sign in parts[edge_id]:
            mapping[old] = sign * index
    return new_edges, mapping


def _collapse_degree_two(edges, roots, available):
    """Merge again after unused stubs are gone, and point every chain at the new ids."""
    merged, mapping = _merge_unbranched_edges(edges)
    if all(mapping.get(edge_id) == edge_id for edge_id in edges):
        return edges, roots, available

    def remap(chain):
        out = []
        for signed in chain:
            new = mapping[abs(signed)]
            if signed < 0:
                new = -new
            if out and abs(out[-1]) == abs(new):
                continue
            out.append(new)
        return out

    for root in roots.values():
        root["edges"] = remap(root["edges"])
    _drop_nested_copies(roots)
    used = {abs(signed) for root in roots.values() for signed in root["edges"]}
    kept = []
    seen = set()
    for chain in available:
        for signed in remap(chain):
            edge_id = abs(signed)
            if edge_id in used or edge_id in seen or edge_id not in merged:
                continue
            seen.add(edge_id)
            kept.append(edge_id)
    for edge_id in merged:
        if edge_id not in used and edge_id not in seen:
            kept.append(edge_id)
    return merged, roots, [[edge_id] for edge_id in sorted(kept)]


def _drop_nested_copies(roots):
    """Drop a child that repeats edges its parent already contains."""
    changed = True
    while changed:
        changed = False
        for root_id, root in list(roots.items()):
            parent_id = root["parent"]
            if parent_id not in roots:
                continue
            parent_edges = {abs(signed) for signed in roots[parent_id]["edges"]}
            if {abs(signed) for signed in root["edges"]} <= parent_edges:
                for child in roots.values():
                    if child["parent"] == root_id:
                        child["parent"] = parent_id
                del roots[root_id]
                changed = True
                break


def _join_at(left, right, node):
    def toward(edge):
        if edge.v == node:
            return list(edge.points)
        return list(reversed(edge.points))

    def away(edge):
        if edge.u == node:
            return list(edge.points)
        return list(reversed(edge.points))

    head = toward(left)
    tail = away(right)
    if head and tail and head[-1] == tail[0]:
        tail = tail[1:]
    return head + tail


def _tissue(edge):
    """The class painted on the skeleton, not the main-path mark.

    Type 10 means the edge lies on the chosen main path. It is class 1 only
    when the skeleton itself was class 1. A class-2 edge on that path stays
    class 2, so a straight color change is not joined into one segment.
    """
    kind = getattr(edge, "orig_root_type", None)
    if kind is None:
        kind = getattr(edge, "root_type", None)
    if kind in (1, 10):
        return 1
    return kind


def _same_tissue(left, right):
    return _tissue(left) == _tissue(right)


def _joined_tissue(left, right):
    if 10 in (getattr(left, "root_type", None), getattr(right, "root_type", None)):
        return 10
    return getattr(left, "root_type", None)


class _Edge:
    def __init__(self, u, v, points, root_type=None, orig_root_type=None):
        self.u = u
        self.v = v
        self.points = points
        self.root_type = root_type
        self.orig_root_type = root_type if orig_root_type is None else orig_root_type


def _step_index(edge):
    return {point: index for index, point in enumerate(edge.points)}


def _signed_ends(edges, signed):
    edge = edges[abs(signed)]
    if signed > 0:
        return edge.u, edge.v
    return edge.v, edge.u


def _bridge(edges, leave, arrive, forbidden):
    """The one graph edge that joins these nodes, signed from leave to arrive."""
    if leave == arrive:
        return None
    for edge_id, edge in edges.items():
        if edge_id == forbidden or {edge.u, edge.v} != {leave, arrive}:
            continue
        return edge_id if edge.u == leave else -edge_id
    return None


def chains_from_polyline(points, edges, strict=False):
    indexes = {edge_id: _step_index(edge) for edge_id, edge in edges.items()}
    signed = []
    for left, right in zip(points, points[1:]):
        if left == right:
            continue
        found = None
        for edge_id, index in indexes.items():
            if left in index and right in index and abs(index[left] - index[right]) == 1:
                found = edge_id if index[right] > index[left] else -edge_id
                break
        if found is None:
            if strict:
                return None
            continue
        # Going straight back along the segment just taken traces the same
        # pixels twice. A loop is kept once.
        if signed and abs(signed[-1]) == abs(found):
            continue
        if signed and _signed_ends(edges, signed[-1])[1] != _signed_ends(edges, found)[0]:
            link = _bridge(
                edges, _signed_ends(edges, signed[-1])[1], _signed_ends(edges, found)[0], abs(signed[-1]),
            )
            if link is None or (signed and signed[-1] == link):
                if strict:
                    return None
                continue
            signed.append(link)
        signed.append(found)
    signed = _drop_untouched_edges(signed, edges, points)
    if not signed:
        return signed
    return _clip_to_covered_ends(signed, edges, points)


def _drop_untouched_edges(signed, edges, points):
    """Ignore a segment the polyline only touches.

    A lateral that starts on the last pixels of the main root must not take
    that whole segment. The main root still keeps it, because its polyline
    runs along the segment.
    """
    if len(signed) <= 1:
        return signed
    present = set(points)

    def mostly_covered(signed_id):
        edge = edges[abs(signed_id)]
        hit = sum(point in present for point in edge.points)
        return hit * 2 >= len(edge.points)

    covered = [signed_id for signed_id in signed if mostly_covered(signed_id)]
    if not covered:
        return signed
    return _longest_run(edges, covered)


def _covered_end_run(directed, present):
    """A covered prefix or suffix, when the rest of the segment is untouched."""
    covered = [point in present for point in directed]
    if len(covered) < 3 or all(covered) or not any(covered):
        return None
    if covered[0]:
        end = 0
        while end + 1 < len(covered) and covered[end + 1]:
            end += 1
        if end >= 1 and len(covered) - end - 1 >= 2 and not any(covered[end + 1:]):
            return 0, end
    if covered[-1]:
        start = len(covered) - 1
        while start > 0 and covered[start - 1]:
            start -= 1
        if start >= 2 and start <= len(covered) - 2 and not any(covered[:start]):
            return start, len(covered) - 1
    return None


def _clip_to_covered_ends(signed, edges, points):
    """Stop a loaded root at a cut instead of taking the rest of the segment."""
    present = set(points)
    order = {point: index for index, point in enumerate(points)}
    clipped = []
    for signed_id in signed:
        edge = edges[abs(signed_id)]
        directed = list(edge.points if signed_id > 0 else reversed(edge.points))
        run = _covered_end_run(directed, present)
        if run is None:
            clipped.append(signed_id)
            continue
        start, stop = run
        piece = directed[start:stop + 1]
        if len(piece) < 2 or len(piece) == len(directed):
            clipped.append(signed_id)
            continue
        new_id = None
        sign = 1
        for edge_id, existing in edges.items():
            if existing.points == piece:
                new_id = edge_id
                break
            if existing.points == list(reversed(piece)):
                new_id = edge_id
                sign = -1
                break
        if new_id is None:
            new_id = (max(edges) + 1) if edges else 1
            edges[new_id] = _Edge(piece[0], piece[-1], piece, edge.root_type)
        first = order.get(piece[0], 0)
        last = order.get(piece[-1], 0)
        if first > last:
            sign = -sign
        clipped.append(new_id * sign)
    return clipped


def _longest_run(edges, signed_edges):
    best = []
    current = []
    for signed in signed_edges:
        if current and _signed_ends(edges, current[-1])[1] != _signed_ends(edges, signed)[0]:
            if len(current) > len(best):
                best = current
            current = []
        current.append(signed)
    if len(current) > len(best):
        best = current
    return best


def _assign_edges_once(roots, edges):
    """The automatic tracing gives each segment to one root.

    A second copy is what Duplicate produces later, not the initial construction.
    """
    owner = {}
    for root_id in list(roots):
        root = roots.get(root_id)
        if root is None:
            continue
        kept = [signed for signed in root["edges"] if abs(signed) not in owner]
        if kept and not all(
            _signed_ends(edges, left)[1] == _signed_ends(edges, right)[0]
            for left, right in zip(kept, kept[1:])
        ):
            kept = _longest_run(edges, kept)
        for signed in kept:
            owner[abs(signed)] = root_id
        if not kept:
            parent = root["parent"]
            for other in roots.values():
                if other["parent"] == root_id:
                    other["parent"] = parent
            del roots[root_id]
            continue
        root["edges"] = kept


def topology_from_graph(graph, colored, plant_xml=None, prune=5, strict=False):
    assign_edge_polylines(graph, colored)
    edges = edges_from_graph(graph)
    roots = {}
    used = set()
    next_id = 1

    failed = {"value": False}

    def add(element, parent_id):
        nonlocal next_id
        raw = [
            _point((point.get("x"), point.get("y")))
            for point in element.findall("./geometry/polyline/point")
        ]
        chain = chains_from_polyline(raw, edges, strict=strict)
        if chain is None:
            failed["value"] = True
            return
        if not chain and parent_id is None and edges and not strict:
            chain = _main_chain(graph, edges)
        if not chain:
            if strict:
                failed["value"] = True
                return
            for child in element.findall("./root"):
                add(child, parent_id)
            return
        # A junction can be stored as a short root with the rest of the same
        # segment nested inside it. Both map to one graph edge, so the inner
        # copy is the same root and is not added again.
        if parent_id is not None:
            parent_edges = {abs(signed) for signed in roots[parent_id]["edges"]}
            if {abs(signed) for signed in chain} <= parent_edges:
                for child in element.findall("./root"):
                    add(child, parent_id)
                return
        root_id = next_id
        next_id += 1
        roots[root_id] = {"parent": parent_id, "edges": chain, "available": False}
        used.update(abs(signed) for signed in chain)
        if parent_id is None:
            main_box[0] = root_id
        for child in element.findall("./root"):
            add(child, root_id)

    main_box = [None]
    if plant_xml is not None:
        main = plant_xml.find("./root[@label='mainRoot']") or plant_xml.find("./root")
        if main is not None:
            add(main, None)
    if failed["value"]:
        return None
    if main_box[0] is None:
        if strict:
            return None
        chain = _main_chain(graph, edges)
        if not chain:
            raise TopologyError("This plant has no main root to start from.")
        roots[1] = {"parent": None, "edges": chain, "available": False}
        main_box[0] = 1
        used.update(abs(signed) for signed in chain)
    # A saved RSML may use one segment in two roots. Only the automatic
    # tracing is required to give each segment to a single root.
    if not strict:
        _assign_edges_once(roots, edges)
        used = {abs(signed) for root in roots.values() for signed in root["edges"]}
    # A one-pixel edge that no root uses is a skeleton stub, not a segment to edit.
    # Dropping it can leave a degree-2 joint, so the two sides become one segment.
    edges = {
        edge_id: edge for edge_id, edge in edges.items()
        if edge_id in used or not _is_stub(edge)
    }
    available = [[edge_id] for edge_id in sorted(edges) if edge_id not in used]
    edges, roots, available = _collapse_degree_two(edges, roots, available)
    if not strict:
        _assign_edges_once(roots, edges)
    used = {abs(signed) for root in roots.values() for signed in root["edges"]}
    available = [[edge_id] for edge_id in sorted(edges) if edge_id not in used]
    return Topology(edges, roots, main_box[0], available, prune)


def _main_chain(graph, edges):
    ini = next((node for node, data in graph.nodes(data=True) if data.get("type") == "Ini"), None)
    tip = next((node for node, data in graph.nodes(data=True) if data.get("type") == "FTip"), None)
    by_ends = {}
    for edge_id, edge in edges.items():
        by_ends[frozenset((edge.u, edge.v))] = edge_id
    if ini is None or tip is None:
        return [min(edges)] if edges else []
    try:
        import networkx as nx
        path = nx.shortest_path(graph, ini, tip, weight="traversal_cost")
    except Exception:
        return [min(edges)] if edges else []
    chain = []
    for left, right in zip(path, path[1:]):
        edge_id = by_ends.get(frozenset((_point(left), _point(right))))
        if edge_id is None:
            continue
        edge = edges[edge_id]
        signed = edge_id if _point(left) == edge.u else -edge_id
        chain.append(signed)
    return chain


def _shift_point(point, dx, dy):
    return (int(point[0]) + dx, int(point[1]) + dy)


def _shift_graph(graph, dx, dy):
    """Move every node and polyline from crop coordinates into plate coordinates."""
    if dx == 0 and dy == 0 or graph is None or len(graph.nodes) == 0:
        return graph
    import networkx as nx
    mapping = {node: _shift_point(node, dx, dy) for node in list(graph.nodes)}
    shifted = nx.relabel_nodes(graph, mapping, copy=True)
    for _node, data in shifted.nodes(data=True):
        if "pos" in data:
            data["pos"] = _shift_point(data["pos"], dx, dy)
    for _u, _v, data in shifted.edges(data=True):
        if data.get("points"):
            data["points"] = [_shift_point(point, dx, dy) for point in data["points"]]
        for key in ("canonical_start", "canonical_end"):
            if data.get(key) is not None:
                data[key] = _shift_point(data[key], dx, dy)
    return shifted


def _shift_topology(topology, dx, dy):
    if topology is None or (dx == 0 and dy == 0):
        return topology
    for edge in topology.edges.values():
        edge.u = _shift_point(edge.u, dx, dy)
        edge.v = _shift_point(edge.v, dx, dy)
        edge.points = [_shift_point(point, dx, dy) for point in edge.points]
    return topology


def _paste(crop, shape, y, x):
    canvas = np.zeros(shape, dtype=crop.dtype)
    height, width = crop.shape[:2]
    y2 = min(shape[0], y + height)
    x2 = min(shape[1], x + width)
    if y2 > y and x2 > x:
        canvas[y:y2, x:x2] = crop[:y2 - y, :x2 - x]
    return canvas


def _dense_bounds(box, shape, pad=5):
    """The stored dense-segmentation box, plus the pad used while skeletonizing."""
    x, y, width, height = box
    image_h, image_w = shape[:2]
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    x1 = min(image_w, int(x) + int(width) + pad)
    y1 = min(image_h, int(y) + int(height) + pad)
    return x0, y0, x1, y1


def _content_bounds(image, pad=1):
    """Box around every skeleton pixel, including curves that bow past the nodes."""
    height, width = image.shape[:2]
    rows, cols = np.where(image > 0)
    if len(rows) == 0:
        return 0, 0, width, height
    x0 = max(0, int(cols.min()) - pad)
    y0 = max(0, int(rows.min()) - pad)
    x1 = min(width, int(cols.max()) + pad + 1)
    y1 = min(height, int(rows.max()) + pad + 1)
    return x0, y0, x1, y1


def build_graph(
    patch, x_off, y_off, bin_crop, img_shape, prune_iters=5, *,
    classes=(1, 2), cancel=None, initialize=True,
):
    """Trace one plant on its own box, then report points in plate coordinates.

    Phenomics and the root editor use the default: main root and lateral root,
    then the main-root search and the edge polylines. The tracing check passes
    the classes that are checked in its panel and leaves initialize off, because
    that panel runs the main-root search again when the user moves the start
    or the tip. A plate change sets cancel["stop"] so this returns instead of
    finishing the plant the user already left.
    """
    if cancel is not None and cancel.get("stop"):
        return None
    img_h, img_w = img_shape
    plant_h, plant_w = bin_crop.shape[:2]
    pad = 5
    top = min(pad, int(y_off))
    left = min(pad, int(x_off))
    bottom = min(pad, img_h - (int(y_off) + plant_h))
    right = min(pad, img_w - (int(x_off) + plant_w))
    origin_x = int(x_off) - left
    origin_y = int(y_off) - top

    root_bin = np.zeros((top + plant_h + bottom, left + plant_w + right), dtype=np.uint8)
    class_skel = np.zeros_like(root_bin)
    for class_id in classes:
        active = (patch == class_id) & (bin_crop > 0)
        rows, cols = np.where(active)
        if len(rows):
            root_bin[rows + top, cols + left] = 1
            class_skel[rows + top, cols + left] = class_id

    if cancel is not None and cancel.get("stop"):
        return None
    skel, _branches, endpoints, is_valid = extract_skeleton(root_bin, prune_iters)
    if not is_valid or (cancel is not None and cancel.get("stop")):
        if cancel is not None and cancel.get("stop"):
            return None
        n_ends = 0 if endpoints is None else len(endpoints)
        n_pixels = int((skel > 0).sum()) if skel is not None else 0
        if n_pixels == 0:
            raise TopologyError("This plant has no skeleton to trace.")
        raise TopologyError(
            f"This plant was not traced. The skeleton has {n_pixels} pixels and {n_ends} free tip. "
            "A main root is a connected path between two ends. A closed loop does not give two ends, so the graph is not built."
        )
    rows, cols = np.where(skel > 0)
    base = [int(cols[np.argmin(rows)]), int(rows[np.argmin(rows)])]
    graph, _actual, colored = createGraph(skel.copy(), class_skel, base, endpoints, _branches)
    if graph is None or len(graph.nodes) == 0 or (cancel is not None and cancel.get("stop")):
        return None
    # The search rewrites the main path as root type 10. The tracing check
    # still needs the original class on each edge, so it skips this step.
    if initialize:
        try:
            graph = graphInit(graph, main_root_class=1)
        except Exception:
            return None
        assign_edge_polylines(graph, colored)
    graph = _shift_graph(graph, origin_x, origin_y)
    return (
        graph,
        _paste(colored, (img_h, img_w), origin_y, origin_x),
        _paste(skel, (img_h, img_w), origin_y, origin_x),
    )


def automatic_topology(
    graph, colored, full_skel, image_name, uid, genotype, prune=5, box=None,
):
    """Build the automatic tracing inside the plant's dense-segmentation box.

    ``box`` is ``(x, y, width, height)`` in plate pixels, the same rectangle
    drawn in the review tab. Without it, the box is taken from the skeleton.
    """
    plant_xml = None
    if box is None:
        x0, y0, x1, y1 = _content_bounds(colored if np.any(colored) else full_skel)
    else:
        x0, y0, x1, y1 = _dense_bounds(box, full_skel.shape)
    local_graph = _shift_graph(graph, -x0, -y0)
    local_skel = full_skel[y0:y1, x0:x1]
    local_colored = colored[y0:y1, x0:x1]
    try:
        tree, _count = createTree(
            {"fileKey": f"UID_{uid}_{genotype}", "sequenceLabel": "Static_Extraction",
             "Plant": f"Plant_{uid}_{genotype}"},
            0, [image_name or "plate.png"], local_graph, local_skel, local_colored,
        )
        plant_xml = tree.find(".//plant")
    except Exception:
        plant_xml = None
    topology = topology_from_graph(local_graph, local_colored, plant_xml, prune)
    topology = _shift_topology(topology, x0, y0)
    _mark_left_behind(topology, full_skel)
    return topology


def _mark_left_behind(topology, full_skel):
    """Count skeleton pixels that no root received.

    A cycle whose two ends are the same place used to be stored as a one-pixel
    step, so the rest of that segment belonged to nobody. Those pixels must be
    on a root. If any remain, the tracing says so.
    """
    if topology is None or full_skel is None:
        return
    used = set()
    for root_id in topology.roots:
        used.update(map(tuple, topology.root_points(root_id)))
    for chain in topology.available:
        used.update(map(tuple, topology.chain_points(chain)))
    rows, cols = np.where(full_skel > 0)
    missing = sum((int(x), int(y)) not in used for y, x in zip(rows, cols))
    topology.left_behind = missing


def _points_on_skeleton(edges, full_skel):
    height, width = full_skel.shape[:2]
    for edge in edges.values():
        for x, y in edge.points:
            if y < 0 or x < 0 or y >= height or x >= width or full_skel[y, x] == 0:
                return False
    return True


def _restore_stored_edges(record, full_skel):
    """Rebuild a kept tracing from its own segments.

    A split replaces one skeleton segment with two new ones. Those ids are not
    in the graph built again at save time, but their pixels are still on the
    skeleton. A changed mask leaves the pixels, so the automatic tracing is used.
    """
    stored = record.get("edges") if isinstance(record, dict) else None
    if not stored:
        return None
    edges = {}
    for key, item in stored.items():
        points = [tuple(int(value) for value in point) for point in item.get("points") or []]
        if len(points) < 2:
            return None
        edges[int(key)] = _Edge(
            tuple(int(value) for value in item["u"]),
            tuple(int(value) for value in item["v"]),
            points,
            item.get("root_type"),
        )
    if full_skel is not None and not _points_on_skeleton(edges, full_skel):
        return None
    try:
        topology = Topology.from_record(edges, record)
    except (KeyError, TypeError, ValueError):
        return None
    if topology.signature() != record.get("graph_hash"):
        return None
    return topology


def restore_saved(
    graph, colored, full_skel, image_name, uid, genotype, prune, record, plant_xml,
    box=None,
):
    """Match a kept tracing to the edges the editor actually uses.

    Those edges are the skeleton after tiny unused stubs are dropped and the
    pieces between branches are joined, so the raw graph ids will not match.
    """
    restored = _restore_stored_edges(record, full_skel)
    if restored is not None:
        return restored
    if plant_xml is not None:
        imported = import_plant(graph, colored, plant_xml, prune)
        if imported is not None:
            restored = Topology.from_saved(imported.edges, record)
            if restored is not None:
                return restored
    try:
        automatic = automatic_topology(
            graph, colored, full_skel, image_name, uid, genotype, prune, box,
        )
    except Exception:
        return None
    if automatic is None:
        return None
    return Topology.from_saved(automatic.edges, record)


def import_plant(graph, colored, plant_xml, prune=5):
    """Map one exported plant onto this graph.

    None when a polyline does not follow the current edges. Graph edges the
    file does not mention stay in Available segments.
    """
    try:
        topology = topology_from_graph(graph, colored, plant_xml, prune, strict=True)
    except TopologyError:
        return None
    return topology


def _is_stub(edge):
    return len(edge.points) <= 2 and polyline_length(edge.points) < 1.5


def polyline_length(points):
    if len(points) < 2:
        return 0.0
    array = np.asarray(points, dtype=float)
    return float(np.sum(np.hypot(np.diff(array[:, 0]), np.diff(array[:, 1]))))


def measurement_geometry(topology):
    main_points = topology.root_points(topology.main_id)
    lateral_length = 0.0
    individual = []
    order1 = []
    for root_id, root in topology.roots.items():
        if root_id == topology.main_id or root.get("available"):
            continue
        points = topology.root_points(root_id)
        length = polyline_length(points)
        lateral_length += length
        individual.append(length)
        if root["parent"] == topology.main_id and len(points) >= 2:
            order1.append([[point[0], point[1]] for point in points])
    return order1, polyline_length(main_points), lateral_length, main_points, individual


def plant_element(topology, uid, genotype):
    plant = ET.Element("plant", {
        "id": "1",
        "label": f"Plant_{int(uid)}_{genotype}",
        "uid": str(int(uid)),
    })
    counters = {}

    def append(parent_element, root_id, order):
        points = topology.root_points(root_id)
        if order == 0:
            attrib = {"id": "p", "label": "mainRoot"}
        else:
            index = counters.get(order, 0)
            counters[order] = index + 1
            label = f"lat_o{order}_{index}"
            attrib = {"id": label, "label": label}
        element = ET.SubElement(parent_element, "root", attrib)
        polyline = ET.SubElement(ET.SubElement(element, "geometry"), "polyline")
        for point in points:
            ET.SubElement(polyline, "point", {"x": str(point[0]), "y": str(point[1])})
        children = [rid for rid, root in topology.roots.items() if root["parent"] == root_id]
        children.sort(key=lambda rid: _attach_index(topology, root_id, rid))
        for child_id in children:
            append(element, child_id, order + 1)

    append(plant, topology.main_id, 0)
    return plant


def _attach_index(topology, parent_id, child_id):
    nodes = topology.chain_nodes(topology.roots[parent_id]["edges"])
    start = topology.edge_ends(topology.roots[child_id]["edges"][0])[0]
    return nodes.index(start) if start in nodes else 10 ** 9


def set_rsml_pixel_scale(rsml_root, cm_per_px):
    """ImageJ reads unit=cm and resolution as pixels per centimetre."""
    if cm_per_px is None or float(cm_per_px) <= 0:
        return
    metadata = rsml_root.find("metadata")
    if metadata is None:
        metadata = ET.Element("metadata")
        rsml_root.insert(0, metadata)
    pixels_per_cm = 1.0 / float(cm_per_px)

    def put(tag, text):
        element = metadata.find(tag)
        if element is None:
            element = ET.SubElement(metadata, tag)
        element.text = text

    put("unit", "cm")
    put("resolution", f"{pixels_per_cm:.8g}")


def read_rsml_plants(path):
    if not path or not os.path.isfile(path):
        return {}
    plants = {}
    for plant in ET.parse(path).findall("./scene/plant"):
        label = plant.get("label") or ""
        uid = plant.get("uid")
        if uid and str(uid).isdigit():
            plants[int(uid)] = plant
        elif label.startswith("Plant_"):
            head = label[6:].split("_", 1)[0]
            if head.isdigit():
                plants[int(head)] = plant
    return plants


def header_for(image_name, uid, genotype):
    tree = createHeader(
        {"fileKey": f"UID_{uid}_{genotype}", "sequenceLabel": "Static_Extraction",
         "Plant": f"Plant_{uid}_{genotype}"},
        0, [image_name or "plate.png"],
    )
    return tree.find("./metadata")
