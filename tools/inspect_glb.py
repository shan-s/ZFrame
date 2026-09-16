"""Render a .glb from fixed viewpoints, for inspection and fair comparison.

    blender --background --python tools/inspect_glb.py -- --glb out.glb --out shots/

Two jobs:

1. **AGENTS.md §7.** "A change is done when it runs end to end on a real video and
   the output is inspected -- not when it compiles." This makes looking cheap
   enough that it actually happens.

2. **SPIKE.md §5's gallery.** The pass/fail test is whether the result is visibly
   better "to someone who is not you", against Polycam, KIRI and Luma. That
   comparison is only honest if every model is shot from the same angles under
   the same lighting, framed by its own bounding box rather than by a camera
   position that happens to flatter one of them. Hence fixed azimuths and
   bbox-relative framing: run it on the incumbents' exports too.

Reports the geometry numbers alongside, so a model that looks fine but violates
the export contract does not slip through on appearance.
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

# Fixed azimuths, one elevation. Same for every model compared.
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
ELEVATION_DEG = 20.0
RES = 900


def parse_args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = {"glb": None, "out": "/tmp/shots", "views": 4}
    for key in ("glb", "out", "views"):
        flag = f"--{key}"
        if flag in argv:
            args[key] = argv[argv.index(flag) + 1]
    args["views"] = int(args["views"])
    assert args["glb"], "--glb is required"
    return args


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def studio_lighting() -> None:
    """Neutral and identical for every model -- lighting must not favour one."""
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.1
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.06, 1)
    bpy.context.scene.world = world

    for name, rot, energy in (
        ("key", (math.radians(50), 0, math.radians(40)), 3.5),
        ("fill", (math.radians(60), 0, math.radians(200)), 1.4),
    ):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, type="SUN"))
        light.data.energy = energy
        light.rotation_euler = rot
        bpy.context.collection.objects.link(light)


def import_glb(path: str):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    assert meshes, f"No mesh found in {path}"
    return meshes


def geometry_report(meshes) -> dict:
    """Measure the export contract instead of trusting the render."""
    import mathutils

    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    tris = 0

    for obj in meshes:
        obj.data.calc_loop_triangles()
        tris += len(obj.data.loop_triangles)
        for corner in obj.bound_box:
            world = obj.matrix_world @ mathutils.Vector(corner)
            lo = Vector((min(lo[i], world[i]) for i in range(3)))
            hi = Vector((max(hi[i], world[i]) for i in range(3)))

    size = hi - lo
    centre = (hi + lo) / 2
    # CONVENTION: the .glb is Y-up (glTF spec), but Blender's importer converts to
    # Blender's Z-up. These are BLENDER world coordinates, so "up" here is Z.
    # Reading Blender's Y as the base is how a correct export looks broken.
    return {
        "objects": len(meshes),
        "triangles": tris,
        "bbox_min": [round(v, 5) for v in lo],
        "bbox_max": [round(v, 5) for v in hi],
        "size": [round(v, 5) for v in size],
        "up_axis": "blender Z (= +Y in the .glb)",
        "base_up": round(lo.z, 5),          # should be ~0 per AGENTS.md §5
        "height": round(size.z, 5),
        "centred_horizontal": [round(centre.x, 5), round(centre.y, 5)],
    }


def main() -> None:
    args = parse_args()
    out_dir = args["out"]
    os.makedirs(out_dir, exist_ok=True)

    clear_scene()
    meshes = import_glb(args["glb"])
    studio_lighting()

    report = geometry_report(meshes)
    print(json.dumps(report, indent=2))

    lo = Vector(report["bbox_min"])
    hi = Vector(report["bbox_max"])
    centre = (hi + lo) / 2
    radius = max((hi - lo).length, 1e-4)

    target = bpy.data.objects.new("target", None)
    target.location = centre
    bpy.context.collection.objects.link(target)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    track = cam.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = RES
    scene.render.film_transparent = False
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = (
        "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    )

    distance = radius * 1.6
    elevation = math.radians(ELEVATION_DEG)
    azimuths = AZIMUTHS[:: max(1, len(AZIMUTHS) // args["views"])][: args["views"]]

    written = []
    for azimuth in azimuths:
        a = math.radians(azimuth)
        # Y-up orbit: horizontal circle in XZ, lifted along Y.
        cam.location = centre + Vector((
            distance * math.cos(elevation) * math.sin(a),
            distance * math.sin(elevation),
            distance * math.cos(elevation) * math.cos(a),
        ))
        path = os.path.join(out_dir, f"view_{azimuth:03d}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        written.append(path)

    with open(os.path.join(out_dir, "geometry.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nWrote {len(written)} views to {out_dir}")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
