"""Splat -> depth render -> Open3D TSDF -> cleaned .glb — SPIKE.md §3.4.

AGENTS.md §1 is explicit that this is the ONLY licence-clean mesh path. Every
well-documented alternative (2DGS, GOF, SuGaR) carries the Inria non-commercial
licence. gsplat ships `examples/simple_trainer_2dgs.py` inside its own Apache-2.0
repo, which makes 2DGS look safe and is the obvious reach for a mesh. It is not.
Do not substitute it.

WHAT THIS DELIVERS (agreed subset of the AGENTS.md §5 export contract)
---------------------------------------------------------------------
  delivered : background/floor removed, no floaters, upright on the support
              plane, origin at the object's BASE, centred in X/Z, decimated
  deferred  : PBR map separation, LODs, true real-world metric scale
              -- all recorded as debt in docs/SPIKE-LOG.md

Scale in particular is not an oversight: VGGT's output is scale-arbitrary and
bare phone video carries no reference, so "metres" would be a fabrication.
AGENTS.md §6 forbids presenting generated geometry as measurement.

Runs inside image B (no VGGT, per AGENTS.md §4).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/opt/gsplat/examples")

# glTF is Y-up, so the mesh is made Y-up here.
#
# VERIFIED, not assumed: trimesh's .glb exporter writes vertex positions through
# UNCHANGED with an identity node transform -- it does NOT rotate Z-up to Y-up.
# Checked by exporting a box 4 units tall in Z and reading the POSITION accessor
# back out of the .glb: the 4.0 stayed in slot 2. So whatever axis is "up" here
# is the axis that is "up" in the file, and it must be Y.
#
# Beware when checking this in Blender: Blender's glTF IMPORTER does convert,
# Y-up -> Blender's Z-up. A correct model therefore stands up along Blender's Z,
# and reading Blender's Y as "up" makes a correct export look broken.
UP_AXIS = 1


def load_splats(ckpt_path: str, device: str = "cuda") -> dict:
    """Load a simple_trainer checkpoint into activated rasterisation parameters."""
    import torch

    raw = torch.load(ckpt_path, map_location=device)
    splats = raw["splats"] if "splats" in raw else raw

    sh0, shN = splats["sh0"], splats["shN"]
    colors = torch.cat([sh0, shN], dim=1)          # (N, K, 3) SH coefficients
    sh_degree = int(round((colors.shape[1]) ** 0.5)) - 1

    return {
        "means": splats["means"].to(device),
        "quats": torch.nn.functional.normalize(splats["quats"].to(device), dim=-1),
        "scales": torch.exp(splats["scales"].to(device)),
        "opacities": torch.sigmoid(splats["opacities"].to(device)),
        "colors": colors.to(device),
        "sh_degree": sh_degree,
    }


def render_depth_and_colour(splats: dict, viewmat, K, width: int, height: int):
    """One camera -> (rgb HxWx3 uint8, depth HxW float32).

    render_mode="RGB+ED" gives expected depth in the 4th channel. That is the
    channel the whole mesh path hangs off.
    """
    import numpy as np
    import torch
    from gsplat.rendering import rasterization

    with torch.no_grad():
        out, alpha, _ = rasterization(
            splats["means"], splats["quats"], splats["scales"],
            splats["opacities"], splats["colors"],
            viewmat[None], K[None], width, height,
            sh_degree=splats["sh_degree"],
            render_mode="RGB+ED",
        )

    rgb = out[0, ..., :3].clamp(0, 1).cpu().numpy()
    depth = out[0, ..., 3].cpu().numpy().astype(np.float32)
    # Where nothing was rendered, depth is meaningless -- zero it so TSDF skips it.
    depth[alpha[0, ..., 0].cpu().numpy() < 0.5] = 0.0
    return (rgb * 255).astype(np.uint8), depth


def subject_region(camtoworlds):
    """Where the cameras are all looking, and how big that region is.

    Deriving TSDF resolution from the FULL scene extent is wrong whenever the
    background dominates -- a wide floor or a far wall inflates the extent, the
    voxels grow with it, and the actual subject is fused at a handful of voxels
    or lost entirely. That is not a synthetic-scene artifact: a phone capture of
    an object on a table has exactly the same shape.

    A capture orbits its subject, so the subject is where the optical axes
    converge. Solving for that point (least-squares intersection of the view
    rays) and sizing the region from the camera distances puts the resolution
    where the object is.
    """
    import numpy as np

    centres = camtoworlds[:, :3, 3]
    # OpenCV convention: the camera looks down its +Z axis.
    forwards = camtoworlds[:, :3, 2]
    forwards = forwards / np.linalg.norm(forwards, axis=1, keepdims=True)

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centres, forwards):
        P = np.eye(3) - np.outer(d, d)     # projector onto the plane normal to d
        A += P
        b += P @ c
    centre = np.linalg.solve(A, b) if np.linalg.matrix_rank(A) == 3 else centres.mean(0)

    distances = np.linalg.norm(centres - centre, axis=1)
    # Loose upper bound only. It is bounded properly by measure_subject_radius(),
    # which uses rendered depth; the orbit radius alone says nothing about how
    # big the thing in the middle is.
    radius = float(np.percentile(distances, 10) * 0.9)

    return centre, radius, float(np.median(distances))


def measure_subject_radius(splats, parser, centre, fallback, samples: int = 8):
    """Bound the subject by looking at how far away its front surface is.

    Every camera in an orbit points AT the subject, so the depth in the middle of
    each frame is the distance to the subject's near face. The gap between that
    and the camera's distance to the convergence point is the subject's radius
    towards that camera; taking the median over an orbit bounds the whole object.

    The orbit radius cannot substitute for this. It measures how far the
    photographer stood back, so using it sizes the region to the room rather than
    the object, and the background is then inside the sphere and gets fused.
    """
    import numpy as np
    import torch

    n = len(parser.camtoworlds)
    step = max(1, n // samples)
    radii = []

    for i in range(0, n, step):
        camtoworld = parser.camtoworlds[i]
        cam_id = parser.camera_ids[i]
        K = parser.Ks_dict[cam_id]
        width, height = parser.imsize_dict[cam_id]

        viewmat = torch.from_numpy(np.linalg.inv(camtoworld)).float().cuda()
        _, depth = render_depth_and_colour(
            splats, viewmat, torch.from_numpy(K).float().cuda(), width, height
        )

        # Central 12% box: the subject, not the floor at the frame edges.
        cy, cx = height // 2, width // 2
        hy, hx = max(1, int(height * 0.06)), max(1, int(width * 0.06))
        patch = depth[cy - hy:cy + hy, cx - hx:cx + hx]
        patch = patch[patch > 0]
        if not len(patch):
            continue

        front = float(np.median(patch))
        cam_distance = float(np.linalg.norm(camtoworld[:3, 3] - centre))
        if cam_distance > front:
            radii.append(cam_distance - front)

    if not radii:
        return fallback

    # x1.6 so the far side and the base are comfortably inside the sphere.
    radius = float(np.median(radii) * 1.6)
    return float(min(max(radius, fallback * 0.02), fallback))


def fit_support_plane(points, centre, radius):
    """Fit the ground plane to the SPARSE cloud, not to the subject mesh.

    Once fusion is masked to the subject sphere the ground is almost entirely
    gone, so RANSAC on the resulting mesh has no real plane left to find and
    returns a strong-looking fit through the object itself — after which
    "remove everything below the plane" removes most of the subject.

    The sparse cloud still contains the whole floor, which is exactly where a
    ground plane should be measured. Points far from the subject are excluded so
    a distant wall cannot out-vote the surface the object is standing on.
    """
    import numpy as np
    import open3d as o3d

    pts = np.asarray(points)
    near = pts[np.linalg.norm(pts - np.asarray(centre), axis=1) < radius * 6.0]
    if len(near) < 200:
        near = pts

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(near))
    plane, inliers = cloud.segment_plane(
        distance_threshold=radius * 0.05, ransac_n=3, num_iterations=3000
    )
    a, b, c, d = plane
    normal = np.array([a, b, c], dtype=float)
    scale = np.linalg.norm(normal)
    normal, d = normal / scale, d / scale

    # Point the normal upward: the subject sits on the positive side.
    if float(np.asarray(centre) @ normal + d) < 0:
        normal, d = -normal, -d

    return {
        "normal": normal,
        "d": float(d),
        "inlier_fraction": round(len(inliers) / max(1, len(near)), 4),
        "height_above_plane": round(float(np.asarray(centre) @ normal + d), 5),
    }


def mask_to_subject(depth, K, camtoworld, centre, radius):
    """Zero depth wherever the unprojected point falls outside the subject sphere.

    Depth truncation alone cannot do this. A splat renders the empty background
    with Gaussians too, and its EXPECTED depth there is a weighted average over
    low-opacity blobs, which comes back as a NEAR value rather than as "nothing".
    Fused across an orbit, those readings agree with each other and TSDF turns
    them into a solid shell wrapped around the subject -- a crumpled sheet that
    survives every 2D cleanup because, in 3D, it is a real consistent surface.

    Unprojecting and testing against the subject sphere removes it at the source.
    It also does the first half of AGENTS.md §5's "background removed, subject
    isolated" for free, since the room is outside the sphere by construction.
    """
    import numpy as np

    height, width = depth.shape
    v, u = np.nonzero(depth > 0)
    d = depth[v, u]

    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    cam = np.stack([x, y, d, np.ones_like(d)], axis=0)       # OpenCV: looks +Z
    world = (camtoworld @ cam)[:3].T

    outside = np.linalg.norm(world - np.asarray(centre), axis=1) > radius
    masked = depth.copy()
    masked[v[outside], u[outside]] = 0.0
    return masked


def fuse_tsdf(data_dir: str, ckpt_path: str, max_views: int | None = None):
    """TSDF-fuse depth renders at the training camera poses, over the subject region."""
    import numpy as np
    import open3d as o3d
    import torch
    from datasets.colmap import Parser

    # normalize=True matches simple_trainer's default (normalize_world_space),
    # so these poses are in the same frame as the trained splat. Getting this
    # wrong produces a plausible-looking mesh in the wrong place.
    parser = Parser(data_dir=data_dir, factor=1, normalize=True, test_every=1)
    splats = load_splats(ckpt_path)

    # VGGT output is scale-arbitrary, so a fixed voxel size is meaningless.
    # Size everything off the SUBJECT region, not the whole scene (see above).
    centre, orbit_bound, cam_distance = subject_region(parser.camtoworlds)
    radius = measure_subject_radius(splats, parser, centre, orbit_bound)
    print(f"subject radius {radius:.4f} (orbit upper bound was {orbit_bound:.4f})")
    extent = 2.0 * radius
    # 512 across the subject over-resolves a splat's expected-depth noise into
    # holes and speckle. The limit here is depth quality, not voxel count.
    voxel_length = extent / 320.0
    sdf_trunc = voxel_length * 6.0
    # Stop fusing beyond the subject, so distant floor and walls never enter the
    # volume in the first place -- far cheaper than carving them out afterwards.
    depth_trunc = float(cam_distance + radius)
    scene_extent = float(np.linalg.norm(parser.points.max(0) - parser.points.min(0)))
    print(f"subject region: centre {np.round(centre, 3)}, radius {radius:.3f} "
          f"(full scene extent {scene_extent:.3f}) -> voxel {voxel_length:.5f}")

    support = fit_support_plane(parser.points, centre, radius)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    indices = range(len(parser.camtoworlds))
    if max_views:
        step = max(1, len(parser.camtoworlds) // max_views)
        indices = range(0, len(parser.camtoworlds), step)

    n = 0
    for i in indices:
        camtoworld = parser.camtoworlds[i]
        # Parser.camera_ids is per-image (colmap.py:251). Ks_dict/imsize_dict hold
        # the UNDISTORTED intrinsics the trainer actually used, so renders match.
        cam_id = parser.camera_ids[i]
        K = parser.Ks_dict[cam_id]
        width, height = parser.imsize_dict[cam_id]

        viewmat = torch.from_numpy(np.linalg.inv(camtoworld)).float().cuda()
        K_t = torch.from_numpy(K).float().cuda()

        rgb, depth = render_depth_and_colour(splats, viewmat, K_t, width, height)
        depth = mask_to_subject(depth, K, camtoworld, centre, radius)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(depth),
            depth_scale=1.0,            # depth is already in world units
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(camtoworld))
        n += 1

    mesh = volume.extract_triangle_mesh()

    # Clip to the subject region: anything outside it is background by definition.
    box = o3d.geometry.AxisAlignedBoundingBox(centre - radius, centre + radius)
    mesh = mesh.crop(box)
    mesh.compute_vertex_normals()
    print(f"TSDF: fused {n} views, extent {extent:.3f}, voxel {voxel_length:.5f} "
          f"-> {len(mesh.vertices)} verts / {len(mesh.triangles)} tris")
    return mesh, extent, voxel_length, centre, support


def remove_floaters(mesh, near=None):
    """Keep one connected component (AGENTS.md §5: no floaters).

    `near` selects the component closest to that point; without it, the largest.

    Largest is the WRONG rule for this pipeline and cost a long debugging detour.
    A capture of an object on a surface reconstructs the surface as one big sheet
    and the object as a smaller island on top of it, so "largest" reliably keeps
    the floor and throws away the subject — and it does so silently, producing a
    clean, single-component, contract-passing mesh of a patch of ground.

    The capture already tells us which component matters: every camera was
    pointed at the subject. Passing that convergence point picks it directly.
    """
    import numpy as np

    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) <= 1:
        return mesh, 0

    if near is None:
        keep = int(np.argmax(counts))
    else:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        best, best_d = 0, float("inf")
        for label in range(len(counts)):
            member = triangles[labels == label]
            if not len(member):
                continue
            pts = vertices[np.unique(member)]
            d = float(np.linalg.norm(pts - np.asarray(near), axis=1).min())
            # Ignore specks: a stray triangle nearer the centre must not win.
            if d < best_d and counts[label] >= max(50, 0.001 * counts.sum()):
                best, best_d = label, d
        keep = best

    mesh.remove_triangles_by_mask(labels != keep)
    mesh.remove_unreferenced_vertices()
    return mesh, len(counts) - 1


def align_to_support_plane(mesh, extent: float, voxel_length: float,
                           centre, support: dict):
    """Upright, floor removed, origin at the object's base, centred horizontally.

    AGENTS.md §5: orientation comes from the dominant support plane, never from
    wherever the capture happened to start. The plane itself is measured on the
    sparse cloud by fit_support_plane(); here it is only applied.
    """
    import numpy as np

    normal, d = support["normal"], support["d"]

    vertices = np.asarray(mesh.vertices)
    margin = max(voxel_length * 2, extent * 0.002)
    signed = vertices @ normal + d

    remove = signed < margin
    if remove.all():
        raise RuntimeError("support-plane removal would delete every vertex")
    mesh.remove_vertices_by_mask(remove)
    mesh.remove_unreferenced_vertices()

    # Select on proximity to the subject, never on size: the floor remnant is
    # reliably the bigger piece.
    mesh, dropped = remove_floaters(mesh, near=centre)
    if not len(mesh.vertices):
        raise RuntimeError("floor removal deleted the whole mesh")

    # Rotate the plane normal onto +Y (glTF up; see the UP_AXIS note).
    target = np.zeros(3)
    target[UP_AXIS] = 1.0
    v = np.cross(normal, target)
    s_, c_ = np.linalg.norm(v), float(np.dot(normal, target))
    if s_ < 1e-8:
        R = np.eye(3) if c_ > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_) / (s_ ** 2))
    mesh.rotate(R, center=(0, 0, 0))

    # Origin at the BASE, centred horizontally (AGENTS.md §5 — not bbox centre).
    vertices = np.asarray(mesh.vertices)
    offset = vertices.mean(axis=0)
    offset[UP_AXIS] = vertices[:, UP_AXIS].min()
    mesh.translate(-offset)

    return mesh, {
        "plane_inlier_fraction": support["inlier_fraction"],
        "subject_height_above_plane": support["height_above_plane"],
        "floor_margin": round(float(margin), 6),
        "vertices_removed_as_floor": int(remove.sum()),
        "components_removed": int(dropped),
    }


def decimate(mesh, target_triangles: int = 200_000):
    if len(mesh.triangles) <= target_triangles:
        return mesh
    return mesh.simplify_quadric_decimation(target_triangles)


def finalise(mesh, target_triangles: int = 200_000):
    """Decimate, then re-clean and re-seat the mesh.

    Order matters. Quadric decimation can pinch a connected surface apart, so a
    mesh that was a single component going in comes out as hundreds, and removing
    those pieces afterwards shifts the bounding box — which would silently break
    the base-at-origin and centred guarantees if the mesh were not re-seated.
    """
    import numpy as np

    mesh = decimate(mesh, target_triangles)

    vertices = np.asarray(mesh.vertices)
    centroid = vertices.mean(axis=0)
    mesh, dropped = remove_floaters(mesh, near=centroid)

    vertices = np.asarray(mesh.vertices)
    offset = vertices.mean(axis=0)
    offset[UP_AXIS] = vertices[:, UP_AXIS].min()
    mesh.translate(-offset)

    return mesh, dropped


def export_glb(mesh, out_path: str) -> str:
    """Open3D mesh -> .glb via trimesh."""
    import numpy as np
    import trimesh

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colours = np.asarray(mesh.vertex_colors)

    tm = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=(colours * 255).astype(np.uint8) if len(colours) else None,
        process=False,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tm.export(out_path)
    return out_path


def contract_report(mesh) -> dict:
    """Measure the delivered subset of AGENTS.md §5. Numbers, not hope.

    Deliberately reports rather than asserts: a violation here is a finding for
    SPIKE-LOG.md, and SPIKE.md §5 treats a poor mesh as 'Partial', not a stop.
    """
    import numpy as np

    vertices = np.asarray(mesh.vertices)
    base = float(vertices[:, UP_AXIS].min())
    centroid = vertices.mean(axis=0)
    horizontal = [i for i in range(3) if i != UP_AXIS]
    size = float((vertices.max(0) - vertices.min(0)).max())

    labels, counts, _ = mesh.cluster_connected_triangles()

    return {
        # Measured on the pre-export axes; trimesh maps UP_AXIS=Z to glTF +Y.
        "base_at_origin_preexport": round(base, 6),
        "centred_horizontal": [round(float(centroid[i]), 6) for i in horizontal],
        "components": len(np.asarray(counts)),
        "triangles": len(mesh.triangles),
        "vertices": len(vertices),
        "watertight": bool(mesh.is_watertight()),
        "largest_dimension_scene_units": round(size, 4),
        "scale_is_metric": False,  # AGENTS.md §6: never present this as measurement
    }
