"""Checks for the in-memory RSML edge editor. Run: python -m core.test_root_topology"""

import hashlib
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import networkx as nx

from core.analyzer_engine import export_rsml_and_json, extract_plate_metrics
from core.model import PlantImageModel
from core.root_topology import (
    Topology,
    TopologyError,
    _Edge,
    automatic_topology,
    build_graph,
    chains_from_polyline,
    import_plant,
    measurement_geometry,
    plant_element,
    polyline_length,
    read_rsml_plants,
    restore_saved,
)

CROPS = (
    "/home/IPS2/rgaggio/net/NetScratch/FUNRNA/ScanPlatesRootPhenotyping/Laura/"
    "Chile 2026/Low phosphate/14082026 EXP 5 6/crops"
)
TARGET = "14082026_control_Col00704_001_crop"
CM_PER_PX = 0.0085
IMAGEJ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ImageJ")


def _edge(start, end):
    points = [start]
    x, y = start
    while (x, y) != end:
        x += (end[0] > x) - (end[0] < x)
        y += (end[1] > y) - (end[1] < y)
        points.append((x, y))
    return _Edge(start, end, points)


def _sample():
    edges = {
        1: _edge((0, 0), (0, 10)),
        2: _edge((0, 10), (0, 20)),
        3: _edge((0, 10), (10, 10)),
    }
    roots = {1: {"parent": None, "edges": [1, 2], "available": False}}
    return Topology(edges, roots, 1, [[3]])


def _shape(topology):
    def walk(root_id):
        children = [rid for rid, root in topology.roots.items() if root["parent"] == root_id]
        children.sort(key=lambda rid: tuple(topology.roots[rid]["edges"]))
        return (tuple(topology.roots[root_id]["edges"]), [walk(rid) for rid in children])

    return walk(topology.main_id)


def _contiguous(topology, signed_edges):
    for left, right in zip(signed_edges, signed_edges[1:]):
        if topology.edge_ends(left)[1] != topology.edge_ends(right)[0]:
            return False
    return True


def _assert_structure(topology):
    assert topology.main_id in topology.roots
    assert _contiguous(topology, topology.roots[topology.main_id]["edges"])
    owned = set()
    for root_id, root in topology.roots.items():
        assert root["edges"], root_id
        assert _contiguous(topology, root["edges"])
        for signed in root["edges"]:
            edge = topology.edges[abs(signed)]
            assert edge.points[0] == edge.u and edge.points[-1] == edge.v
            assert len(edge.points) >= 2
            owned.add(abs(signed))
    for chain in topology.available:
        assert _contiguous(topology, chain)
        owned.update(abs(signed) for signed in chain)
    assert owned == set(topology.edges)


def _load(base):
    model = PlantImageModel()
    model.load_task(CROPS, base)
    return model


def _topology(model, uid, base):
    patch, x_off, y_off = model._get_class_patch(uid)
    y2 = y_off + patch.shape[0]
    x2 = x_off + patch.shape[1]
    built = build_graph(
        patch, x_off, y_off, model.masks[uid][y_off:y2, x_off:x2],
        model.raw_image.shape[:2], 5,
    )
    assert built is not None, uid
    graph, colored, full_skel = built
    meta = model.plants_meta.get(uid) or {}
    topology = automatic_topology(
        graph, colored, full_skel, base + ".png", uid, meta.get("genotype") or "Unknown", 5,
    )
    return graph, colored, topology


def _plants(model):
    rows = []
    for uid in sorted(model.masks):
        meta = model.plants_meta.get(uid) or {}
        if meta.get("ignore") or not meta.get("germinated", True):
            continue
        rows.append({
            "uid": uid,
            "genotype": meta.get("genotype") or "Unknown",
            "plant_num": meta.get("plant_num") or str(uid),
            "germinated": True,
            "ignore": False,
        })
    return rows


def test_split_connect_reuse_and_remove():
    topology = _sample()
    topology.copy_segment(1, 1)
    assert topology.roots[1]["edges"] == [1, 2]
    assert topology.main_spine == [1, 2]
    assert [2] in topology.available
    source = f"a{topology.available.index([2])}"
    try:
        topology.connect(1, source)
        raise AssertionError("a segment already in the main root was connected to it")
    except TopologyError as exc:
        assert "already part" in str(exc)
    topology.connect(1, "a0")
    topology.discard_available(topology.available.index([2]))
    child = next(
        rid for rid, root in topology.roots.items()
        if root["parent"] == 1 and abs(root["edges"][0]) == 3
    )
    assert topology.roots[child]["edges"] == [3]
    assert topology.can_apply()
    topology.reuse(1, 0)
    assert topology.ownership(1) == 2
    assert not topology.can_apply()
    topology.discard_available(0)
    assert topology.ownership(1) == 1
    try:
        topology.remove_copy(1, 0)
        raise AssertionError("the last copy was removed")
    except TopologyError:
        pass
    branched = Topology(
        {1: _edge((0, 0), (0, 10)), 2: _edge((0, 10), (0, 20))},
        {
            1: {"parent": None, "edges": [1], "available": False},
            2: {"parent": 1, "edges": [2], "available": False},
        },
        1, [],
    )
    try:
        branched.connect(2, 1)
        raise AssertionError("a cycle was accepted")
    except TopologyError:
        pass


def test_shared_trunk_can_be_copied_and_connected_either_way():
    edges = {
        1: _edge((0, 0), (0, 10)),
        2: _edge((0, 10), (0, 20)),
        3: _edge((0, 20), (10, 20)),
        4: _edge((0, 20), (0, 30)),
    }
    roots = {
        1: {"parent": None, "edges": [1], "available": False},
        2: {"parent": 1, "edges": [2], "available": False},
        3: {"parent": 2, "edges": [3], "available": False},
        4: {"parent": 2, "edges": [4], "available": False},
    }
    topology = Topology(edges, roots, 1, [])
    assert topology.copy_root(2) is None
    assert topology.available == [[2]]
    topology.connect(4, "a0")
    assert topology.roots[2]["edges"] == [2]
    assert topology.roots[4]["edges"] == [2, 4]
    assert topology.roots[3]["edges"] == [3]
    assert _contiguous(topology, topology.roots[4]["edges"])
    assert topology.copy_segment(2, 0) is None
    assert topology.available == [[2]]


def test_tertiary_then_lateral_merges_in_the_only_direction():
    edges = {
        1: _edge((0, 0), (0, 10)),
        2: _edge((0, 10), (0, 20)),
        3: _edge((0, 20), (10, 20)),
        4: _edge((0, 20), (0, 30)),
    }
    roots = {
        1: {"parent": None, "edges": [1], "available": False},
        2: {"parent": 1, "edges": [2], "available": False},
        3: {"parent": 2, "edges": [3], "available": False},
        4: {"parent": 2, "edges": [4], "available": False},
    }
    topology = Topology(edges, roots, 1, [])
    message = topology.connect(3, 2)
    assert "beginning" in message.lower() or "end" in message.lower()
    assert topology.roots[3]["edges"] == [2, 3]
    assert topology.roots[3]["parent"] == 1
    assert 2 not in topology.roots
    assert topology.roots[4]["parent"] == 3
    assert _contiguous(topology, topology.roots[3]["edges"])


def test_split_segment_extends_the_lateral_without_a_loop():
    edges = {
        1: _edge((0, 0), (0, 10)),
        2: _edge((0, 10), (0, 20)),
        3: _edge((0, 20), (12, 20)),
    }
    roots = {
        1: {"parent": None, "edges": [1, 2], "available": False},
        2: {"parent": 1, "edges": [3], "available": False},
    }
    topology = Topology(edges, roots, 1, [])
    old_end = topology.root_points(2)[-1]
    topology.copy_segment(1, 1)
    message = topology.connect(2, "a0")
    assert "beginning" in message.lower()
    assert topology.roots[1]["edges"] == [1, 2]
    assert topology.roots[2]["edges"] == [2, 3]
    points = topology.root_points(2)
    assert points[0] == (0, 10)
    assert points[-1] == old_end
    assert _contiguous(topology, topology.roots[2]["edges"])
    try:
        topology.reverse_chain(2)
        raise AssertionError("a root that ends at a tip was reversed")
    except TopologyError as exc:
        assert "tip" in str(exc).lower()
    try:
        topology.connect(2, 1)
        raise AssertionError("the main root was pulled into the lateral")
    except TopologyError as exc:
        assert "copy" in str(exc).lower()
    try:
        topology.connect(1, 2)
        raise AssertionError("a lateral was connected straight onto the main root")
    except TopologyError as exc:
        assert "copy" in str(exc).lower()


def test_split_lateral_becomes_a_child():
    edges = {
        1: _edge((0, 0), (0, 10)),
        2: _edge((0, 10), (0, 20)),
        3: _edge((0, 20), (0, 30)),
    }
    roots = {
        1: {"parent": None, "edges": [1], "available": False},
        2: {"parent": 1, "edges": [2, 3], "available": False},
    }
    topology = Topology(edges, roots, 1, [])
    topology.split(2, 1)
    assert topology.roots[2]["edges"] == [2]
    child = next(rid for rid, root in topology.roots.items() if root["parent"] == 2)
    assert topology.roots[child]["edges"] == [3]
    assert topology.order_of(child) == 2
    assert topology.available == []


def test_laterals_that_do_not_touch_stay_apart():
    edges = {
        1: _edge((0, 0), (0, 10)),
        2: _edge((0, 10), (0, 20)),
        3: _edge((0, 10), (10, 10)),
        4: _edge((0, 20), (10, 20)),
    }
    roots = {
        1: {"parent": None, "edges": [1, 2], "available": False},
        2: {"parent": 1, "edges": [3], "available": False},
        3: {"parent": 1, "edges": [4], "available": False},
    }
    topology = Topology(edges, roots, 1, [])
    try:
        topology.connect(2, 3)
        raise AssertionError("laterals that do not touch were joined")
    except TopologyError as exc:
        assert "do not touch" in str(exc).lower()
    assert topology.roots[2]["edges"] == [3]
    assert topology.roots[3]["edges"] == [4]
    assert topology.roots[1]["edges"] == [1, 2]


def test_main_root_cannot_be_split_and_a_lateral_can_be_cut():
    edges = {
        1: _edge((0, 0), (0, 10)),
        2: _edge((0, 10), (0, 30)),
    }
    roots = {
        1: {"parent": None, "edges": [1, 2], "available": False},
        2: {"parent": 1, "edges": [2], "available": False},
    }
    topology = Topology(edges, roots, 1, [])
    try:
        topology.split(1, 1)
        raise AssertionError("the main root was split at an intersection")
    except TopologyError as exc:
        assert "cannot split the main root" in str(exc).lower()
    shorter = topology.cut_polyline(1, 0, (0, 6))
    assert len(topology.roots[1]["edges"]) == 3
    assert topology.main_spine == topology.roots[1]["edges"]
    assert not topology._main_is_open()
    topology.reconnect_main()
    assert _contiguous(topology, topology.roots[1]["edges"])
    assert topology.root_points(1)[0] == (0, 0)
    assert topology.root_points(1)[-1] == (0, 30)
    assert topology.roots[2]["edges"] == [2]
    topology.copy_segment(1, shorter)
    assert topology.available
    fresh = Topology(
        {1: _edge((0, 0), (0, 10)), 2: _edge((0, 10), (0, 30))},
        {
            1: {"parent": None, "edges": [1, 2], "available": False},
            2: {"parent": 1, "edges": [2], "available": False},
        },
        1, [],
    )
    original = list(fresh.edges[2].points)
    shorter = fresh.cut_polyline(2, 0, (0, 26))
    assert fresh.roots[1]["edges"] == [1, 2]
    assert fresh.main_spine == [1, 2]
    fresh.reconnect_main()
    assert fresh.roots[1]["edges"] == [1, 2]
    assert fresh.edges[2].points == original
    assert len(fresh.roots[2]["edges"]) == 2
    assert _contiguous(fresh, fresh.roots[2]["edges"])
    assert fresh.root_points(2)[0] == (0, 10)
    assert fresh.root_points(2)[-1] == (0, 30)
    piece = fresh.edges[abs(fresh.roots[2]["edges"][shorter])]
    assert polyline_length(piece.points) <= polyline_length(
        fresh.edges[abs(fresh.roots[2]["edges"][1 - shorter])].points
    )
    created = fresh.copy_segment(2, shorter)
    assert created is None
    assert fresh.available[-1] == [fresh.roots[2]["edges"][shorter]]


def test_split_edges_restore_onto_the_same_skeleton():
    import numpy as np
    topology = Topology(
        {1: _edge((0, 0), (0, 10)), 2: _edge((0, 10), (0, 20))},
        {1: {"parent": None, "edges": [1, 2], "available": False}},
        1, [],
    )
    topology.cut_polyline(1, 0, (0, 6))
    record = topology.to_record()
    assert "edges" in record
    assert 1 not in {int(edge_id) for edge_id in record["edges"]}
    skeleton = np.zeros((30, 5), dtype=np.uint8)
    for item in record["edges"].values():
        for x, y in item["points"]:
            skeleton[y, x] = 1
    restored = restore_saved(None, None, skeleton, "plate.png", 1, "Unknown", 5, record, None)
    assert restored is not None
    assert restored.roots[1]["edges"] == topology.roots[1]["edges"]
    assert restored.signature() == topology.signature()
    restored.reconnect_main()
    blank = np.zeros_like(skeleton)
    assert _restore_misses_blank(record, blank)


def _restore_misses_blank(record, skeleton):
    from core.root_topology import _restore_stored_edges
    return _restore_stored_edges(record, skeleton) is None


def test_a_cut_lateral_stops_at_the_cut_when_loaded():
    full = _edge((0, 0), (0, 47))
    edges = {10: full, 4: _edge((0, 0), (8, 0))}
    covered = full.points[:36]
    polyline = list(reversed(edges[4].points)) + covered[1:]
    chain = chains_from_polyline(polyline, edges, strict=True)
    assert chain is not None
    piece = edges[abs(chain[-1])]
    assert piece.points[0] == (0, 0)
    assert piece.points[-1] == (0, 35)
    assert edges[10].points[-1] == (0, 47)
    assert chains_from_polyline(full.points, {10: full}, strict=True) == [10]


def test_touching_the_end_does_not_copy_the_segment():
    edges = {
        1: _edge((0, 0), (0, 8)),
        2: _edge((0, 8), (6, 8)),
    }
    nick = [(0, 7), (0, 8)] + edges[2].points[1:]
    assert chains_from_polyline(nick, edges) == [2]
    assert chains_from_polyline(edges[1].points, edges) == [1]


def _assert_no_shared_edge(topology):
    seen = set()
    for root in topology.roots.values():
        for signed in root["edges"]:
            assert abs(signed) not in seen
            seen.add(abs(signed))


def _assert_plant2_crossing_uses_a_copied_link(topology):
    """The short link between the two crossing nodes is duplicated, then each pair is joined."""
    link = (768, 529), (770, 531)
    yellow = next(
        root_id for root_id, root in topology.roots.items()
        if len(root["edges"]) == 1
        and {topology.edges[abs(root["edges"][0])].u, topology.edges[abs(root["edges"][0])].v} == set(link)
    )
    main_edges = {abs(signed) for signed in topology.roots[topology.main_id]["edges"]}
    assert abs(topology.roots[yellow]["edges"][0]) not in main_edges

    def end_of(root_id):
        return topology.edge_ends(topology.roots[root_id]["edges"][-1])[1]

    def start_of(root_id):
        return topology.edge_ends(topology.roots[root_id]["edges"][0])[0]

    laterals = [
        root_id for root_id, root in topology.roots.items()
        if root["parent"] == topology.main_id and root_id != topology.main_id
    ]
    upper = next(root_id for root_id in laterals if end_of(root_id) == (768, 529))
    lower = next(root_id for root_id in laterals if end_of(root_id) == (770, 531))
    down = next(
        root_id for root_id, root in topology.roots.items()
        if root["parent"] == lower and start_of(root_id) == (770, 531)
    )
    left = next(
        root_id for root_id, root in topology.roots.items()
        if root["parent"] == upper and start_of(root_id) == (768, 529) and root_id != yellow
    )
    link_edge = abs(topology.roots[yellow]["edges"][0])
    topology.copy_root(yellow)
    topology.connect(lower, yellow)
    topology.connect(lower, left)
    topology.connect(upper, "a0")
    topology.connect(upper, down)
    _assert_structure(topology)
    assert end_of(lower) == (744, 572)
    assert end_of(upper) == (749, 706)
    seen = {}
    for root in topology.roots.values():
        for signed in root["edges"]:
            seen[abs(signed)] = seen.get(abs(signed), 0) + 1
    shared = [edge_id for edge_id, count in seen.items() if count > 1]
    assert shared == [link_edge]
    assert link_edge not in main_edges


def test_connect_rejects_a_gap():
    topology = _sample()
    topology.edges[4] = _edge((50, 50), (60, 50))
    topology.available.append([4])
    try:
        topology.connect(1, "a1")
        raise AssertionError("a segment that does not meet the root was joined")
    except TopologyError:
        pass


def test_target_plate_edges_round_trip_and_do_not_touch_the_file():
    if not os.path.isdir(CROPS):
        print("skip Chile plates; the crops folder is not on this machine")
        return
    source = os.path.join(CROPS, TARGET + "_Topology.rsml")
    before = hashlib.sha256(open(source, "rb").read()).digest() if os.path.isfile(source) else None
    model = _load(TARGET)
    assert sorted(model.masks) == [1, 2, 3, 4, 5, 6, 7, 8]
    plants = _plants(model)
    results, _warnings, _imported = extract_plate_metrics(model, plants, CM_PER_PX, 5, topology_path=None)
    assert sorted(results) == [1, 2, 3, 4, 5, 6, 7, 8]

    folder = tempfile.mkdtemp()
    export_rsml_and_json(folder, TARGET, {"scale_cm_px": CM_PER_PX}, results)
    exported = os.path.join(folder, TARGET + "_Topology.rsml")
    saved = read_rsml_plants(exported)
    assert sorted(saved) == [1, 2, 3, 4, 5, 6, 7, 8]
    metadata = ET.parse(exported).getroot().find("metadata")
    assert metadata.findtext("unit") == "cm"
    assert abs(float(metadata.findtext("resolution")) - 1.0 / CM_PER_PX) < 1e-4

    for uid in plants:
        uid = uid["uid"]
        _graph, colored, topology = _topology(model, uid, TARGET)
        again = _topology(model, uid, TARGET)[2]
        assert topology.signature() == again.signature()
        assert _shape(topology) == _shape(again)
        _assert_structure(topology)
        imported = import_plant(_graph, colored, plant_element(topology, uid, "Unknown"), 5)
        assert imported is not None
        assert _shape(imported) == _shape(topology)
        assert measurement_geometry(imported)[1:3] == measurement_geometry(topology)[1:3]
        reloaded = import_plant(_graph, colored, saved[uid], 5)
        assert reloaded is not None, uid
        assert _shape(reloaded) == _shape(topology)
        assert abs(results[uid]["mr_length_mm"] - measurement_geometry(topology)[1] * CM_PER_PX * 10) < 1e-6

    model.topology_overrides[3] = results[3]  # not a topology record; reset must still leave the file alone
    model.topology_overrides.pop(3, None)
    model.topology_decided.add(3)
    model.topology_decided.discard(3)
    if before is not None:
        after = hashlib.sha256(open(source, "rb").read()).digest()
        assert before == after
    return exported


def test_representative_plates():
    if not os.path.isdir(CROPS):
        return
    simple = _topology(_load("14082026_control_N597_001_crop"), 6, "14082026_control_N597_001_crop")[2]
    _assert_structure(simple)
    assert set(simple.order_of(rid) for rid in simple.roots) <= {0, 1}
    assert simple.can_apply()

    false_tertiary = _topology(
        _load("14082026_control_Col00704_002_crop"), 2, "14082026_control_Col00704_002_crop",
    )[2]
    _assert_structure(false_tertiary)
    assert all(false_tertiary.order_of(rid) < 2 for rid in false_tertiary.roots)

    quaternary = _topology(
        _load("14082026_control_N597_002_crop"), 3, "14082026_control_N597_002_crop",
    )[2]
    _assert_structure(quaternary)
    assert any(quaternary.order_of(rid) >= 3 for rid in quaternary.roots)

    target = _load(TARGET)
    for uid in (2, 7):
        topology = _topology(target, uid, TARGET)[2]
        _assert_structure(topology)
        assert topology.available == []
        assert topology.can_apply()
        for edge in topology.edges.values():
            assert topology.node_degree(edge.u) != 2
            assert topology.node_degree(edge.v) != 2
    graph, colored, plant2 = _topology(target, 2, TARGET)
    _assert_no_shared_edge(plant2)
    _assert_plant2_crossing_uses_a_copied_link(plant2)
    reloaded = import_plant(graph, colored, plant_element(plant2, 2, "Unknown"), 5)
    assert reloaded is not None
    assert _shape(reloaded) == _shape(plant2)

    graph, _colored, fused = _topology(
        _load("14082026_control_N832_002_crop"), 2, "14082026_control_N832_002_crop",
    )
    _assert_structure(fused)
    assert len(graph.edges) - len(graph.nodes) + nx.number_connected_components(graph) > 0
    fused.reuse(fused.main_id, 0)
    assert fused.ownership(abs(fused.roots[fused.main_id]["edges"][0])) == 2


def test_all_crop_plates():
    if not os.path.isdir(CROPS):
        return
    bases = sorted(
        name[:-5] for name in os.listdir(CROPS)
        if name.endswith(".json") and not name.endswith("_Metrics.json")
    )
    assert len(bases) == 30, len(bases)
    checked = 0
    for base in bases:
        model = _load(base)
        for row in _plants(model):
            _graph, _colored, topology = _topology(model, row["uid"], base)
            _assert_structure(topology)
            checked += 1
    assert checked >= 200, checked


def _java_report(rsml_path):
    java = os.path.join(IMAGEJ, "jre", "bin", "java")
    jar = os.path.join(IMAGEJ, "RSML_reader_v2.jar")
    if not os.path.isfile(java) or not os.path.isfile(jar):
        return None
    source = r'''
import java.util.prefs.Preferences;
public class RsmlLengths {
    public static void main(String[] args) {
        SR.prefs = Preferences.userRoot().node("ImageJ/SmartRoot");
        RootModel model = new RootModel(args[0]);
        System.out.println("NPRIMARY\t" + model.getNPRoot());
        System.out.println("NROOT\t" + model.getNRoot());
        System.out.println("TOTAL\t" + model.getTotalRLength());
        System.out.println("PIXEL_SIZE\t" + model.pixelSize);
        for (int i = 0; i < model.rootList.size(); i++) {
            Root root = model.rootList.get(i);
            System.out.println("ROOT\t" + root.getRootID() + "\t" + root.getRootLength());
        }
    }
}
'''
    folder = tempfile.mkdtemp()
    src = os.path.join(folder, "RsmlLengths.java")
    with open(src, "w", encoding="utf-8") as handle:
        handle.write(source)
    classpath = os.pathsep.join([os.path.join(IMAGEJ, "ij.jar"), jar])
    subprocess.check_call(
        ["javac", "--release", "8", "-cp", classpath, "-d", folder, src],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    output = subprocess.check_output(
        [java, "-Djava.awt.headless=true", "-cp", os.pathsep.join([folder, classpath]),
         "RsmlLengths", rsml_path],
        stderr=subprocess.DEVNULL, text=True,
    )
    report = {"roots": {}}
    for line in output.splitlines():
        if "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        if key == "ROOT":
            name, length = value.split("\t")
            report.setdefault("order", []).append((name, float(length)))
        elif key == "PIXEL_SIZE":
            report["pixel_size"] = float(value)
        else:
            report[key] = float(value)
    return report


def _xml_lengths(path):
    rows = []
    plants = ET.parse(path).findall("./scene/plant")

    def walk(element):
        points = [
            (float(point.get("x")), float(point.get("y")))
            for point in element.findall("./geometry/polyline/point")
        ]
        rows.append((element.get("label"), polyline_length(points)))
        for child in element.findall("./root"):
            walk(child)

    for plant in plants:
        for root in plant.findall("./root"):
            walk(root)
    return plants, rows


def test_imagej_reads_the_exported_scale():
    if not os.path.isdir(CROPS):
        return
    exported = test_target_plate_edges_round_trip_and_do_not_touch_the_file()
    if exported is None:
        return
    report = _java_report(exported)
    if report is None:
        raise AssertionError("ImageJ RSML reader is not available")
    plants, rows = _xml_lengths(exported)
    assert int(report["NPRIMARY"]) == len(plants)
    assert int(report["NROOT"]) == len(rows)
    assert abs(report["pixel_size"] - CM_PER_PX) < 1e-5
    assert len(report["order"]) == len(rows)
    paired = list(zip(report["order"], rows))
    # getRootLength stays in file pixels. getTotalRLength is those pixels times pixelSize.
    pixel_total = 0.0
    for (imagej_name, imagej_length), (our_name, pixels) in paired:
        assert imagej_name == our_name, (imagej_name, our_name)
        assert abs(imagej_length - pixels) < 0.05, (our_name, imagej_length, pixels)
        pixel_total += pixels
    assert abs(report["TOTAL"] - pixel_total * report["pixel_size"]) < 0.05


def main():
    test_split_connect_reuse_and_remove()
    test_shared_trunk_can_be_copied_and_connected_either_way()
    test_tertiary_then_lateral_merges_in_the_only_direction()
    test_split_segment_extends_the_lateral_without_a_loop()
    test_split_lateral_becomes_a_child()
    test_laterals_that_do_not_touch_stay_apart()
    test_main_root_cannot_be_split_and_a_lateral_can_be_cut()
    test_split_edges_restore_onto_the_same_skeleton()
    test_a_cut_lateral_stops_at_the_cut_when_loaded()
    test_touching_the_end_does_not_copy_the_segment()
    test_connect_rejects_a_gap()
    test_target_plate_edges_round_trip_and_do_not_touch_the_file()
    test_representative_plates()
    test_all_crop_plates()
    test_imagej_reads_the_exported_scale()
    print("root topology checks passed")


if __name__ == "__main__":
    main()
