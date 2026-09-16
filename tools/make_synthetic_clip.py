"""Render a synthetic orbit clip with GROUND-TRUTH camera poses.

    blender --background --python tools/make_synthetic_clip.py -- --out /tmp/synthetic

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR
----------------------------------------
This exists so the pipeline can be debugged without burning the real test
captures or GPU budget on plumbing bugs. It proves frames -> VGGT -> gsplat ->
TSDF -> .glb runs end to end, and because the true camera poses are known it can
say whether VGGT's poses are actually right rather than merely plausible.

It proves NOTHING about the thesis in SPIKE.md. It has none of the gloss, thin
structure, low texture or motion blur that the spike exists to test. A pass here
is a green light to run real footage, not evidence about Polycam, KIRI or Luma.

Blender is used only as a local asset generator. It is not a product dependency
and does not appear in LICENSES.md.
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

FRAMES = 96
RES_X, RES_Y = 1280, 720
ORBIT_RADIUS = 7.0
ORBIT_HEIGHT = 2.6


def parse_args() -> str:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = "/tmp/synthetic"
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    return out


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.textures):
        for item in list(block):
            block.remove(item)


def textured_material(name: str, scale: float, colour_a, colour_b):
    """Procedural noise. Texture matters: a flat surface gives VGGT nothing."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")

    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 8.0
    ramp.color_ramp.elements[0].color = (*colour_a, 1.0)
    ramp.color_ramp.elements[1].color = (*colour_b, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.65

    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def build_scene():
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.data.materials.append(textured_material("ground_mat", 12.0, (0.18, 0.17, 0.16), (0.42, 0.40, 0.37)))

    bpy.ops.mesh.primitive_monkey_add(size=2.6, location=(0, 0, 1.55))
    subject = bpy.context.active_object
    subject.name = "subject"
    bpy.ops.object.modifier_add(type="SUBSURF")
    subject.modifiers["Subdivision"].levels = 2
    subject.modifiers["Subdivision"].render_levels = 2
    bpy.ops.object.shade_smooth()
    subject.data.materials.append(textured_material("subject_mat", 26.0, (0.55, 0.22, 0.12), (0.85, 0.66, 0.35)))

    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", type="SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(35))
    bpy.context.collection.objects.link(sun)

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    bpy.context.scene.world = world

    return subject


def main() -> None:
    out_dir = parse_args()
    os.makedirs(out_dir, exist_ok=True)

    clear_scene()
    subject = build_scene()

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 40.0
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    track = cam.constraints.new(type="TRACK_TO")
    track.target = subject
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = RES_X, RES_Y
    scene.render.fps = 24
    scene.frame_start, scene.frame_end = 1, FRAMES
    # EEVEE was renamed in Blender 4.2; ask the enum rather than guess by version.
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = (
        "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    )

    # Slight height variation so the orbit is not a perfect degenerate circle --
    # a coplanar camera path is a weak conditioning case for pose estimation.
    for f in range(1, FRAMES + 1):
        t = (f - 1) / FRAMES
        angle = t * 2 * math.pi
        cam.location = Vector((
            ORBIT_RADIUS * math.cos(angle),
            ORBIT_RADIUS * math.sin(angle),
            ORBIT_HEIGHT + 0.8 * math.sin(angle * 2),
        ))
        cam.keyframe_insert(data_path="location", frame=f)

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = os.path.join(out_dir, "synthetic")
    bpy.ops.render.render(animation=True)

    # Ground truth. Blender cameras look down -Z with +Y up (OpenGL); COLMAP uses
    # +Z forward, -Y down. The comparison script does the conversion -- the
    # convention is recorded here rather than silently baked in.
    poses = []
    for f in range(1, FRAMES + 1):
        scene.frame_set(f)
        poses.append({
            "frame": f,
            "camera_to_world": [list(row) for row in cam.matrix_world],
        })

    with open(os.path.join(out_dir, "ground_truth.json"), "w") as fh:
        json.dump({
            "convention": "blender: camera looks -Z, up +Y, right +X (OpenGL-style)",
            "resolution": [RES_X, RES_Y],
            "sensor_width_mm": cam_data.sensor_width,
            "focal_length_mm": cam_data.lens,
            "focal_length_px": cam_data.lens / cam_data.sensor_width * RES_X,
            "frames": FRAMES,
            "poses": poses,
        }, fh, indent=2)

    print(f"\nWrote {out_dir}/synthetic.mp4 and ground_truth.json ({FRAMES} frames)")


if __name__ == "__main__":
    main()
