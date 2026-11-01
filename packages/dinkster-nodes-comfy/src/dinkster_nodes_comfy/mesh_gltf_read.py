"""glTF scene and material parsing over caller-supplied resource bytes."""

from __future__ import annotations

import base64
import importlib
import json
import struct
import urllib.parse
from collections.abc import Callable
from io import BytesIO
from typing import Any

np = importlib.import_module("numpy")
Warn = Callable[[str, str], None]
ReadResource = Callable[[str], bytes]

_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_SUPPORTED_REQUIRED = {
    "EXT_mesh_gpu_instancing",
    "EXT_texture_webp",
    "KHR_materials_emissive_strength",
    "KHR_materials_unlit",
}


def parse_container(data: bytes) -> tuple[Any, bytes | None]:
    if data[:4] != b"glTF":
        return json.loads(data), None
    if len(data) < 12:
        raise ValueError("GLB file truncated (missing 12-byte header)")
    _, version, _ = struct.unpack_from("<4sII", data, 0)
    if version != 2:
        raise ValueError(f"unsupported GLB container version {version}")
    json_chunk = bin_chunk = None
    offset = 12
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + length]
        offset += length
        if kind == 0x4E4F534A and json_chunk is None:
            json_chunk = chunk
        elif kind == 0x004E4942 and bin_chunk is None:
            bin_chunk = chunk
    if json_chunk is None:
        raise ValueError("GLB file has no JSON chunk")
    return json.loads(json_chunk), bin_chunk


def _read_uri(uri: str, resource: ReadResource) -> bytes:
    if uri.startswith("data:"):
        header, _, payload = uri.partition(",")
        return (
            base64.b64decode(payload)
            if ";base64" in header
            else urllib.parse.unquote_to_bytes(payload)
        )
    scheme = urllib.parse.urlparse(uri).scheme
    if scheme:
        raise ValueError(
            f"glTF references URI scheme {scheme!r}; only relative file paths are allowed"
        )
    return resource(urllib.parse.unquote(uri))


def _view_data(gltf: Any, buffers: list[bytes], index: int) -> bytes:
    view = gltf["bufferViews"][index]
    buffer = buffers[int(view.get("buffer", 0))]
    start = view.get("byteOffset", 0)
    return buffer[start : start + view["byteLength"]]


def read_accessor(gltf: Any, buffers: list[bytes], index: int) -> tuple[Any, bool]:
    accessor = gltf["accessors"][index]
    count = accessor["count"]
    components = _TYPE_SIZES[accessor["type"]]
    dtype = np.dtype(_COMPONENT_DTYPES[accessor["componentType"]])
    element = dtype.itemsize * components
    if "bufferView" in accessor:
        view = gltf["bufferViews"][accessor["bufferView"]]
        buffer = buffers[int(view.get("buffer", 0))]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride") or element
        if stride == element:
            values = (
                np.frombuffer(buffer, dtype, count * components, start)
                .reshape(count, components)
                .copy()
            )
        else:
            raw = np.frombuffer(buffer, np.uint8, stride * (count - 1) + element, start)
            rows = np.lib.stride_tricks.as_strided(raw, (count, element), (stride, 1))
            values = np.ascontiguousarray(rows).view(dtype).reshape(count, components)
    else:
        values = np.zeros((count, components), dtype)
    sparse = accessor.get("sparse")
    if sparse:
        size = sparse["count"]
        indices, replacements = sparse["indices"], sparse["values"]
        index_dtype = np.dtype(_COMPONENT_DTYPES[indices["componentType"]])
        view = gltf["bufferViews"][indices["bufferView"]]
        index_values = np.frombuffer(
            buffers[view.get("buffer", 0)],
            index_dtype,
            size,
            view.get("byteOffset", 0) + indices.get("byteOffset", 0),
        ).astype(np.int64)
        view = gltf["bufferViews"][replacements["bufferView"]]
        replacement_values = np.frombuffer(
            buffers[view.get("buffer", 0)],
            dtype,
            size * components,
            view.get("byteOffset", 0) + replacements.get("byteOffset", 0),
        ).reshape(size, components)
        values[index_values] = replacement_values
    return values, bool(accessor.get("normalized"))


def to_float(values: Any, normalized: bool) -> Any:
    if values.dtype == np.float32:
        return values
    out = values.astype(np.float32)
    if not normalized:
        return out
    if values.dtype == np.uint8:
        return out / 255.0
    if values.dtype == np.uint16:
        return out / 65535.0
    if values.dtype == np.int8:
        return np.maximum(out / 127.0, -1.0)
    if values.dtype == np.int16:
        return np.maximum(out / 32767.0, -1.0)
    return out


def _quaternion(x: float, y: float, z: float, w: float) -> Any:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _local_matrix(node: Any) -> Any:
    if "matrix" in node:
        return np.array(node["matrix"], np.float32).reshape(4, 4).T
    matrix = np.eye(4, dtype=np.float32)
    rotation = _quaternion(*node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    matrix[:3, :3] = rotation * np.array(node.get("scale", (1.0, 1.0, 1.0)), np.float32)
    matrix[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return matrix


def _instance_matrices(gltf: Any, buffers: list[bytes], node: Any, warn: Warn) -> Any:
    attributes = node.get("extensions", {}).get("EXT_mesh_gpu_instancing", {}).get("attributes", {})
    if not attributes:
        return None
    counts = {gltf["accessors"][accessor]["count"] for accessor in attributes.values()}
    if len(counts) > 1:
        raise ValueError(
            "EXT_mesh_gpu_instancing attribute accessors disagree on instance count: "
            f"{sorted(counts)}"
        )
    if not any(key in attributes for key in ("TRANSLATION", "ROTATION", "SCALE")):
        warn(
            "instancing-custom",
            "EXT_mesh_gpu_instancing has only custom attributes; importing a single copy",
        )
        return None
    translation = (
        to_float(*read_accessor(gltf, buffers, attributes["TRANSLATION"]))
        if "TRANSLATION" in attributes
        else None
    )
    rotation = (
        to_float(*read_accessor(gltf, buffers, attributes["ROTATION"]))
        if "ROTATION" in attributes
        else None
    )
    scale = (
        to_float(*read_accessor(gltf, buffers, attributes["SCALE"]))
        if "SCALE" in attributes
        else None
    )
    matrices: list[Any] = []
    for index in range(counts.pop()):
        matrix = np.eye(4, dtype=np.float32)
        rotate = (
            _quaternion(*rotation[index]) if rotation is not None else np.eye(3, dtype=np.float32)
        )
        matrix[:3, :3] = rotate * (scale[index] if scale is not None else 1.0)
        if translation is not None:
            matrix[:3, 3] = translation[index]
        matrices.append(matrix)
    return matrices


def _mesh_nodes(gltf: Any, buffers: list[bytes], warn: Warn) -> Any:
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes")
    if scenes:
        roots = scenes[gltf.get("scene", 0)].get("nodes", [])
    elif nodes:
        children = {child for node in nodes for child in node.get("children", [])}
        roots = [index for index in range(len(nodes)) if index not in children]
    else:
        for index in range(len(gltf.get("meshes", []))):
            yield {"mesh": index}, np.eye(4, dtype=np.float32)
        return
    seen: set[int] = set()
    stack: list[Any] = [(index, np.eye(4, dtype=np.float32)) for index in roots]
    while stack:
        index, parent = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        node = nodes[index]
        world = parent @ _local_matrix(node)
        if "mesh" in node:
            instances = _instance_matrices(gltf, buffers, node, warn)
            if instances is None:
                yield node, world
            else:
                warn(
                    "instancing",
                    f"EXT_mesh_gpu_instancing: expanding {len(instances)} instances "
                    "into merged geometry",
                )
                for matrix in instances:
                    yield node, world @ matrix
        for child in node.get("children", []):
            stack.append((child, world))


def _triangles(indices: Any, mode: int) -> Any:
    if mode == 4:
        if len(indices) % 3:
            raise ValueError("TRIANGLES primitive index count must be divisible by 3")
        return indices.reshape(-1, 3)
    if len(indices) < 3:
        return np.zeros((0, 3), np.int64)
    if mode == 6:
        first = np.full(len(indices) - 2, indices[0], dtype=np.int64)
        return np.stack([first, indices[1:-1], indices[2:]], axis=1)
    triangles = np.stack([indices[:-2], indices[1:-1], indices[2:]], axis=1)
    triangles[1::2] = triangles[1::2, ::-1]
    return triangles


def _attribute(
    gltf: Any, buffers: list[bytes], accessor: int, count: int, warn: Warn, label: str
) -> Any:
    values = to_float(*read_accessor(gltf, buffers, accessor))
    if values.shape[0] < count:
        warn(
            f"count:{label}",
            f"{label} has {values.shape[0]} entries for {count} vertices; attribute dropped",
        )
        return None
    return values[:count]


def load_scene_geometry(gltf: Any, buffers: list[bytes], warn: Warn) -> list[dict[str, Any]]:
    unsupported = set(gltf.get("extensionsRequired", [])) - _SUPPORTED_REQUIRED
    if unsupported:
        raise ValueError(
            f"glTF requires extensions this loader does not support: {sorted(unsupported)}"
        )
    primitives: list[dict[str, Any]] = []
    for node, world in _mesh_nodes(gltf, buffers, warn):
        if "skin" in node:
            warn("skin", "skinned mesh: joints/weights ignored, geometry imported in bind pose")
        linear = world[:3, :3]
        determinant = float(np.linalg.det(linear))
        normal_matrix = np.linalg.inv(linear).T if abs(determinant) > 1e-12 else linear
        flip = determinant < 0.0
        for primitive in gltf["meshes"][node["mesh"]].get("primitives", []):
            mode = primitive.get("mode", 4)
            if mode not in (4, 5, 6):
                warn(f"mode{mode}", f"skipping non-triangle primitive (mode {mode})")
                continue
            attributes = primitive.get("attributes", {})
            if "POSITION" not in attributes:
                continue
            positions = to_float(*read_accessor(gltf, buffers, attributes["POSITION"]))[:, :3]
            count = positions.shape[0]
            if count == 0:
                continue
            positions = positions @ linear.T + world[:3, 3]
            indices = (
                read_accessor(gltf, buffers, primitive["indices"])[0].reshape(-1).astype(np.int64)
                if "indices" in primitive
                else np.arange(count, dtype=np.int64)
            )
            faces = _triangles(indices, mode)
            if faces.shape[0] == 0:
                continue
            if faces.min() < 0 or faces.max() >= count:
                raise ValueError("primitive contains a face index outside its POSITION accessor")
            if flip:
                faces = np.ascontiguousarray(faces[:, ::-1])
            if primitive.get("targets"):
                warn("morph", "morph targets ignored; base geometry imported")
            out: dict[str, Any] = {
                "positions": np.ascontiguousarray(positions, np.float32),
                "faces": faces,
                "uvs": None,
                "colors": None,
                "normals": None,
                "tangents": None,
                "material": primitive.get("material"),
            }
            if "TEXCOORD_0" in attributes:
                uv = _attribute(gltf, buffers, attributes["TEXCOORD_0"], count, warn, "TEXCOORD_0")
                out["uvs"] = uv[:, :2] if uv is not None else None
            if "COLOR_0" in attributes:
                values, normalized = read_accessor(gltf, buffers, attributes["COLOR_0"])
                values = to_float(values, normalized or values.dtype != np.float32)
                if values.shape[0] >= count:
                    out["colors"] = np.clip(values[:count], 0.0, 1.0)
            if "NORMAL" in attributes:
                normals = _attribute(gltf, buffers, attributes["NORMAL"], count, warn, "NORMAL")
                if normals is not None:
                    normals = normals[:, :3] @ normal_matrix.T
                    out["normals"] = np.ascontiguousarray(
                        normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12),
                        np.float32,
                    )
            if "TANGENT" in attributes:
                tangents = _attribute(gltf, buffers, attributes["TANGENT"], count, warn, "TANGENT")
                if tangents is not None and tangents.shape[1] == 4:
                    xyz = tangents[:, :3] @ linear.T
                    xyz /= np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12)
                    handedness = tangents[:, 3:4] * (-1.0 if flip else 1.0)
                    out["tangents"] = np.ascontiguousarray(
                        np.concatenate([xyz, handedness], axis=1), np.float32
                    )
            primitives.append(out)
    return primitives


def _texture(
    gltf: Any, buffers: list[bytes], resource: ReadResource, info: Any, warn: Warn, label: str
) -> Any:
    if info is None:
        return None
    if info.get("texCoord", 0) != 0:
        warn(
            f"texcoord:{label}",
            f"{label} uses TEXCOORD_{info['texCoord']}; MESH only carries TEXCOORD_0",
        )
    if "KHR_texture_transform" in info.get("extensions", {}):
        warn("textransform", "KHR_texture_transform ignored; UVs used as-is")
    texture = gltf["textures"][info["index"]]
    source = texture.get("source")
    if source is None:
        webp = texture.get("extensions", {}).get("EXT_texture_webp")
        source = webp.get("source") if webp is not None else None
    if source is None:
        warn(
            f"compressed:{label}",
            f"{label}: compressed texture (basisu/ktx2) without fallback; skipped",
        )
        return None
    image = gltf["images"][source]
    if "bufferView" in image:
        raw = _view_data(gltf, buffers, image["bufferView"])
    elif "uri" in image:
        raw = _read_uri(image["uri"], resource)
    else:
        return None
    image_module = importlib.import_module("PIL.Image")
    with image_module.open(BytesIO(bytes(raw))) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0


def extract_material(
    gltf: Any, buffers: list[bytes], resource: ReadResource, index: Any, warn: Warn
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "texture": None,
        "metallic_roughness": None,
        "normal_map": None,
        "emissive": None,
        "occlusion_in_mr": False,
        "unlit": False,
        "material": None,
    }
    if index is None:
        return result
    material = gltf["materials"][index]
    extensions = material.get("extensions", {})
    result["unlit"] = "KHR_materials_unlit" in extensions
    pbr = material.get("pbrMetallicRoughness", {})
    result["texture"] = _texture(
        gltf, buffers, resource, pbr.get("baseColorTexture"), warn, "baseColorTexture"
    )
    mr_info = pbr.get("metallicRoughnessTexture")
    result["metallic_roughness"] = _texture(
        gltf, buffers, resource, mr_info, warn, "metallicRoughnessTexture"
    )
    normal_info = material.get("normalTexture")
    result["normal_map"] = _texture(gltf, buffers, resource, normal_info, warn, "normalTexture")
    result["emissive"] = _texture(
        gltf, buffers, resource, material.get("emissiveTexture"), warn, "emissiveTexture"
    )
    occlusion = material.get("occlusionTexture")
    if occlusion is not None:
        if mr_info is not None and occlusion["index"] == mr_info["index"]:
            result["occlusion_in_mr"] = True
        else:
            warn(
                "occlusion",
                "standalone occlusionTexture not representable in MESH (ORM packing only); skipped",
            )
    overrides: dict[str, Any] = {}
    base_color = pbr.get("baseColorFactor")
    if base_color is not None and list(base_color) != [1.0, 1.0, 1.0, 1.0]:
        overrides["base_color_factor"] = [float(value) for value in base_color]
    overrides["metallic_factor"] = float(pbr.get("metallicFactor", 1.0))
    overrides["roughness_factor"] = float(pbr.get("roughnessFactor", 1.0))
    overrides["double_sided"] = bool(material.get("doubleSided", False))
    if normal_info is not None and normal_info.get("scale", 1.0) != 1.0:
        overrides["normal_scale"] = float(normal_info["scale"])
    if occlusion is not None and occlusion.get("strength", 1.0) != 1.0:
        overrides["occlusion_strength"] = float(occlusion["strength"])
    emissive = material.get("emissiveFactor", (0.0, 0.0, 0.0))
    if any(value > 0.0 for value in emissive):
        overrides["emissive_factor"] = [float(value) for value in emissive]
    strength = extensions.get("KHR_materials_emissive_strength", {}).get("emissiveStrength")
    if strength is not None:
        overrides["emissive_strength"] = float(strength)
    result["material"] = overrides
    return result


def load_gltf(
    data: bytes, resource: ReadResource, warn: Warn
) -> tuple[Any, list[bytes], list[dict[str, Any]]]:
    gltf, binary = parse_container(data)
    version = str(gltf.get("asset", {}).get("version", ""))
    if version.partition(".")[0] != "2":
        raise ValueError(
            f"unsupported glTF asset version {version or 'unknown'}; only glTF 2.x is supported"
        )
    buffers: list[bytes] = []
    for buffer in gltf.get("buffers", []):
        if "uri" in buffer:
            buffers.append(_read_uri(buffer["uri"], resource))
        elif binary is not None:
            buffers.append(binary)
        else:
            raise ValueError("glTF buffer has no uri and there is no GLB BIN chunk")
    return gltf, buffers, load_scene_geometry(gltf, buffers, warn)
