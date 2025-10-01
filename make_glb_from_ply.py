# make_glb_from_ply.py
import blenderproc as bproc
from blenderproc.python.utility.Utility import Utility, stdout_redirected
import bpy, sys, os

argv = sys.argv[sys.argv.index("--")+1:]  # args after --
inp  = os.path.abspath(argv[0])
outp = os.path.abspath(argv[1])
radius = float(argv[2]) if len(argv) > 2 else 0.03

# Fresh scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import PLY (expects per-vertex colors; Blender names it "Col")
bpy.ops.import_mesh.ply(filepath=inp)
pc = bpy.context.selected_objects[0]
pc.name = "PointCloud"

# Build small sphere to instance
bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12, radius=radius)
ball = bpy.context.active_object
bpy.ops.object.shade_smooth(use_auto_smooth=True)

# Material that uses vertex colors
mat = bpy.data.materials.new("VC_Mat")
mat.use_nodes = True
nt = mat.node_tree
for n in nt.nodes: nt.nodes.remove(n)
n_out = nt.nodes.new("ShaderNodeOutputMaterial")
n_bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
n_attr = nt.nodes.new("ShaderNodeAttribute"); n_attr.attribute_name = "Col"
nt.links.new(n_attr.outputs["Color"], n_bsdf.inputs["Base Color"])
n_bsdf.inputs["Specular"].default_value = 0.1
n_bsdf.inputs["Roughness"].default_value = 0.35
nt.links.new(n_bsdf.outputs["BSDF"], n_out.inputs["Surface"])
ball.data.materials.append(mat)

# Geometry Nodes: instance ball on points, then realize
gn = pc.modifiers.new("GN_Instance", type='NODES')
ng = bpy.data.node_groups.new("GN_InstanceOnPoints", 'GeometryNodeTree')
gn.node_group = ng

ng.inputs.new('NodeSocketGeometry', 'Geometry')
ng.outputs.new('NodeSocketGeometry', 'Geometry')

n_in  = ng.nodes.new('NodeGroupInput')
n_out = ng.nodes.new('NodeGroupOutput')
n_iop = ng.nodes.new('GeometryNodeInstanceOnPoints')
n_rz  = ng.nodes.new('GeometryNodeRealizeInstances')
n_obj = ng.nodes.new('GeometryNodeObjectInfo'); n_obj.inputs['As Instance'].default_value = True
n_obj.inputs['Object'].default_value = ball

ng.links.new(n_in.outputs['Geometry'],  n_iop.inputs['Points'])
ng.links.new(n_obj.outputs['Geometry'], n_iop.inputs['Instance'])
ng.links.new(n_iop.outputs['Instances'], n_rz.inputs['Geometry'])
ng.links.new(n_rz.outputs['Geometry'], n_out.inputs['Geometry'])

# Apply GN (bake to real mesh) & remove template sphere
bpy.context.view_layer.objects.active = pc
bpy.ops.object.modifier_apply(modifier=gn.name)
bpy.data.objects.remove(ball, do_unlink=True)

# Export GLB
bpy.ops.export_scene.gltf(
    filepath=outp,
    export_format='GLB',
    export_yup=True,
    export_apply=True,
    export_texcoords=False,
    export_normals=True,
    export_colors=True,
    export_materials='EXPORT',
    use_selection=False
)
print("Exported GLB:", outp)
