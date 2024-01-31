import os
import bpy
from datetime import datetime
from tree_gen_package.gui import TreeGen, TreeGenConvertToMesh


def register_utils():
    bpy.utils.register_class(TreeGen)
    bpy.utils.register_class(TreeGenConvertToMesh)


def clean_scene():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()


def call_tree_gen():
    bpy.ops.object.tree_gen()
    bpy.ops.object.tree_gen_convert_to_mesh()


def add_materials():
    bpy.ops.object.select_all(action='DESELECT')

    # Assume you have an empty object selected
    selected_object = bpy.context.active_object

    wood_material = bpy.data.materials.new(name="Wood")
    leaves_material = bpy.data.materials.new(name="Leaves")

    if selected_object.type == 'EMPTY':
        sub_group_models = selected_object.children
        for child in sub_group_models:
            if child.name == "Trunk" or child.name.startswith("Branches"):
                if child.data.materials:
                    child.data.materials[0] = wood_material
                else:
                    child.data.materials.append(wood_material)

            elif child.name.startswith("Leaves"):
                if child.data.materials:
                    child.data.materials[0] = leaves_material
                else:
                    child.data.materials.append(leaves_material)

    wood_material.diffuse_color = (0.173, 0.076, 0.027, 1)
    leaves_material.diffuse_color = (0.049, 0.173, 0.048, 1)


def export_obj_model(export_path):
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    bpy.ops.object.select_all(action='SELECT')
    file_path = export_path + "tree_model.obj"
    bpy.ops.export_scene.obj(
        filepath=file_path,
        check_existing=True,
        use_selection=True,
        use_materials=True,
    )


def generate_tree(export_path):
    register_utils()
    clean_scene()
    call_tree_gen()
    add_materials()
    export_obj_model(export_path)


if __name__ == "__main__":
    path = "outputs\\" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + "\\"
    generate_tree(path)
