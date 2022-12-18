# modify from commit below:
# 989a552de3d1fcd1f178fe873713b884e192dd61
# 2022/10/15 4:04:08
# xy_grid.py

# Author: bbc_mc

# ver 1.0.0
# Add functions to:
#   - add UI radio button
#     - allow to choose process order, from X or Y. (see UI)
#   - add UI checkbox
#     - allow choose to restore checkpoint after process or not.
#   - allow other name of Checkpoint for grid legends (use #  i.e. "sd-v1.4 # SD14". only "SD14" shown in image. )

# ver 1.1.0
# Add functions to:
#   - save info(prompt, seed, Model hash ...) as PNG chunk

# ver 1.2.0
# Add functions to:
#  - add UI checkbox
#    - show only favorite AxisOption, and hide other

# ver 1.3.0
# Bug fix
#  - catch-up change of processing.py
# For debug
#  - add definition of FLG_WIP and FLG_DEBUG
#  - add dprint as debug print function. (show nothing when FLG_DEBUG = False)
# Add Functions
#  - add regex for int/float values. as "Seed",
#    - seed and step and range
#      "   123 ( 4 ) [ 5 ] "  => "123, 127, 131, 135, 139"
#      " - 123 ( 4 ) [ 5 ] "  => "123, 119, 115, 111, 107"
#    - seed and range
#      "   123 [ 5 ] "        => "123, 124, 125, 126, 127"
#      " - 123 [ 5 ] "        => "123, 122, 121, 120, 119"
#    - step and range
#      refer default UI "seed" value. <seed>
#      "   ( 4 ) [ 5 ] "      => "<seed>, <>+4, <>+8, <>+12, <>+16"
#      " - ( 4 ) [ 5 ] "      => "<seed>, <>-4, <>-8, <>-12, <>-16"
#    - range
#      refer default UI "seed" value. <seed>
#      "   [ 5 ] "            => "<seed>, +1, +2, +3, +4"
#      " - [ 5 ] "            => "<seed>, -1, -2, -3, -4"

# ver 1.3.1
# Add checkbox for "save info(prompt, seed, Model hash ...) as PNG chunk"

# ver 1.4.0
# Rebase xy_grid.py on commit 10923f9
# Add new AxisOption "Checkpoint Dropdown"
#   - allow to choose checkpoint by dropdown
# Add selection Radio for grid Legends

# ver 1.4.1
# Bug fix: cant find checkpoint name when selected on "Checkpoint Dropdown" include sub-directory path

# ver 1.4.2
# Rebase xy_grid.py on commit ce049c4
# Bug fix: checkpoint list on "Checkpoint Dropdown" dropdowns is not reloaded

from collections import namedtuple
from copy import copy
from itertools import permutations, chain
import random
import csv
from io import StringIO
from PIL import Image
import numpy as np
import os

import modules.scripts as scripts
import gradio as gr

from modules import images, sd_samplers
from modules.hypernetworks import hypernetwork
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import re


# Setting values
VERSION = "1.4.2"
TITLE_HEADER = "X/Y Plus"
FILE_HEADER = "xy_plus"
FAVORITE_AXISOPTION_NAMES = ["Nothing", "Seed", "Steps", "CFG Scale", "Sampler", "Prompt S/R", "Checkpoint name", "Hypernetwork", "Checkpoint Dropdown"]

# flgs
FLG_WIP = False
FLG_DEBUG = False

# generated definitions
TITLE = TITLE_HEADER + "-" + VERSION + "{}".format("-dev" if FLG_WIP else "")  # "1.2.1-dev"
GRID_FILENAME_HEADER = FILE_HEADER + "{}".format("_dev" if FLG_WIP else "")  # "xy_plus_dev"
SEED_MAX = 4294967294


import inspect

def dprint(str, detailed=False):
    if FLG_DEBUG:
        print(f"DEBUG[{TITLE}] " + str)
        if detailed:
            # show prev function
            print(inspect.stack()[3])


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun


def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")

    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)


def apply_order(p, x, xs):
    token_order = []

    # Initally grab the tokens from the prompt, so they can be replaced in order of earliest seen
    for token in x:
        token_order.append((p.prompt.find(token), token))

    token_order.sort(key=lambda t: t[0])

    prompt_parts = []

    # Split the prompt up, taking out the tokens
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]

    # Rebuild the prompt with the tokens in the order we want
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def apply_sampler(p, x, xs):
    sampler_name = sd_samplers.samplers_map.get(x.lower(), None)
    if sampler_name is None:
        raise RuntimeError(f"Unknown sampler: {x}")

    p.sampler_name = sampler_name


def confirm_samplers(p, xs):
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            raise RuntimeError(f"Unknown sampler: {x}")


def apply_checkpoint(p, x, xs):
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    modules.sd_models.reload_model_weights(shared.sd_model, info)
    p.sd_model = shared.sd_model


def confirm_checkpoints(p, xs):
    for x in xs:
        if '#' in x:
            x = x.split('#')[0].strip()
        if modules.sd_models.get_closet_checkpoint_match(x) is None:
            raise RuntimeError(f"Unknown checkpoint: {x}")


def apply_hypernetwork(p, x, xs):
    if x.lower() in ["", "none"]:
        name = None
    else:
        name = hypernetwork.find_closest_hypernetwork_name(x)
        if not name:
            raise RuntimeError(f"Unknown hypernetwork: {x}")
    hypernetwork.load_hypernetwork(name)


def apply_hypernetwork_strength(p, x, xs):
    hypernetwork.apply_strength(x)


def confirm_hypernetworks(p, xs):
    for x in xs:
        if x.lower() in ["", "none"]:
            continue
        if not hypernetwork.find_closest_hypernetwork_name(x):
            raise RuntimeError(f"Unknown hypernetwork: {x}")


def apply_clip_skip(p, x, xs):
    opts.data["CLIP_stop_at_last_layers"] = x


def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 8)

    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return x


def format_value_join_list(p, opt, x):
    return ", ".join(x)


def do_nothing(p, x, xs):
    pass


def format_nothing(p, opt, x):
    return ""


def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x

AxisOption = namedtuple("AxisOption", ["label", "type", "apply", "format_value", "confirm"])
AxisOptionImg2Img = namedtuple("AxisOptionImg2Img", ["label", "type", "apply", "format_value", "confirm"])


axis_options = [
    AxisOption("Nothing", str, do_nothing, format_nothing, None),
    AxisOption("Seed", int, apply_field("seed"), format_value_add_label, None),
    AxisOption("Var. seed", int, apply_field("subseed"), format_value_add_label, None),
    AxisOption("Var. strength", float, apply_field("subseed_strength"), format_value_add_label, None),
    AxisOption("Steps", int, apply_field("steps"), format_value_add_label, None),
    AxisOption("CFG Scale", float, apply_field("cfg_scale"), format_value_add_label, None),
    AxisOption("Prompt S/R", str, apply_prompt, format_value, None),
    AxisOption("Prompt order", str_permutations, apply_order, format_value_join_list, None),
    AxisOption("Sampler", str, apply_sampler, format_value, confirm_samplers),
    AxisOption("Checkpoint name", str, apply_checkpoint, format_value, confirm_checkpoints),
    AxisOption("Hypernetwork", str, apply_hypernetwork, format_value, confirm_hypernetworks),
    AxisOption("Hypernet str.", float, apply_hypernetwork_strength, format_value_add_label, None),
    AxisOption("Sigma Churn", float, apply_field("s_churn"), format_value_add_label, None),
    AxisOption("Sigma min", float, apply_field("s_tmin"), format_value_add_label, None),
    AxisOption("Sigma max", float, apply_field("s_tmax"), format_value_add_label, None),
    AxisOption("Sigma noise", float, apply_field("s_noise"), format_value_add_label, None),
    AxisOption("Eta", float, apply_field("eta"), format_value_add_label, None),
    AxisOption("Clip skip", int, apply_clip_skip, format_value_add_label, None),
    AxisOption("Denoising", float, apply_field("denoising_strength"), format_value_add_label, None),
    AxisOption("Cond. Image Mask Weight", float, apply_field("inpainting_mask_weight"), format_value_add_label, None),
]

axis_options += [
    AxisOption("Checkpoint Dropdown", str, apply_checkpoint, format_value, confirm_checkpoints)
]

def draw_xy_grid(p, xs, ys, x_labels, y_labels, cell, draw_legend, include_lone_images, start_from):

    hor_texts = [[] for x in x_labels]
    ver_texts = [[] for y in y_labels]
    if draw_legend == 1 or draw_legend == 2:
        hor_texts = [[images.GridAnnotation(x)] for x in x_labels]
    if draw_legend == 1 or draw_legend == 3:
        ver_texts = [[images.GridAnnotation(y)] for y in y_labels]

    # Temporary list of all the images that are generated to be populated into the grid.
    # Will be filled with empty images for any individual step that fails to process properly
    image_cache = []

    processed_result = None
    cell_mode = "P"
    cell_size = (1,1)

    state.job_count = len(xs) * len(ys) * p.n_iter

    inner = xs if start_from == 0 else ys
    outer = ys if start_from == 0 else xs

    for outer_index, outer_val in enumerate(outer):
        for inner_index, inner_val in enumerate(inner):
            state.job = f"{inner_index + outer_index * len(inner) + 1} out of {len(inner) * len(outer)}"

            # check if interruped
            if state.interrupted:
                dprint(f"found interrupte button pushed.")
                return processed_result
            elif state.skipped:
                dprint(f"found skip button pushed.")
                dprint(f"skip current image.")
                continue

            if start_from == 0:
                processed:Processed = cell(inner_val, outer_val)
            else:
                processed:Processed = cell(outer_val, inner_val)

            try:
                # this dereference will throw an exception if the image was not processed
                # (this happens in cases such as if the user stops the process from the UI)
                processed_image = processed.images[0]

                if processed_result is None:
                    # Use our first valid processed result as a template container to hold our full results
                    processed_result = copy(processed)
                    cell_mode = processed_image.mode
                    cell_size = processed_image.size
                    processed_result.images = [Image.new(cell_mode, cell_size)]

                image_cache.append(processed_image)
                if include_lone_images:
                    processed_result.images.append(processed_image)
                    processed_result.all_prompts.append(processed.prompt)
                    processed_result.all_seeds.append(processed.seed)
                    processed_result.infotexts.append(processed.infotexts[0])
            except:
                image_cache.append(Image.new(cell_mode, cell_size))

    if not processed_result:
        print("Unexpected error: draw_xy_grid failed to return even a single processed image")
        return Processed(p, [])

    #re-order
    if start_from != 0:
        _image_cache = image_cache
        image_cache = []
        for inner_index, inner_val in enumerate(inner):
            for outer_index, outer_val in enumerate(outer):
                _index = outer_index * len(inner) + inner_index
                image_cache.append(_image_cache[_index])

    grid = images.image_grid(image_cache, rows=len(ys))
    if draw_legend != 0:
        grid = images.draw_grid_annotations(grid, cell_size[0], cell_size[1], hor_texts, ver_texts)

    processed_result.images[0] = grid

    return processed_result


class SharedSettingsStackHelper(object):
    def __enter__(self):
        self.CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
        self.hypernetwork = opts.sd_hypernetwork
        self.model = shared.sd_model

    def __exit__(self, exc_type, exc_value, tb):
        modules.sd_models.reload_model_weights(self.model)

        hypernetwork.load_hypernetwork(self.hypernetwork)
        hypernetwork.apply_strength()

        opts.data["CLIP_stop_at_last_layers"] = self.CLIP_stop_at_last_layers


re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*\])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*\])?\s*")

# SSC
# " - 123 ( 3 ) [ 4 ] " => 123, 126, 129, 132
re_range_seed_step_count = re.compile(r"\s*([+-]?)\s*(\d+)\s*\(\s*(\d+|\s*?)\s*\)\s*\[\s*(\d+)\s*\]\s*")

# SEC
# " - 123 [ 4 ] " => 123, 124, 125, 126
re_range_seed_count = re.compile(r"\s*([+-]?)\s*(\d+)\s*\[\s*(\d+)\s*\]\s*")

# STC
# " - ( 3 ) [ 4 ] " => (seed), +3, +6, +9
re_range_step_count = re.compile(r"\s*([+-]?)\s*\(\s*(\d+|\s*?)\s*\)\s*\[\s*(\d+)\s*\]\s*")

# OLY
# basic count
# " - [ 3 ] " => (seed), -1, -2
re_range_only = re.compile(r"\s*([+-]?)\s*\[\s*(\d+)\s*\]\s*")


class Script(scripts.Script):

    def title(self):
        return TITLE

    def ui(self, is_img2img):
        xyp_current_axis_options = [x for x in axis_options if type(x) == AxisOption or type(x) == AxisOptionImg2Img and is_img2img]

        with gr.Row():
            xyp_start_from = gr.Radio(label='Start from Axis', choices=['X', 'Y'], value='X', type="index")

        with gr.Row():
            xyp_x_type = gr.Dropdown(label="X type", choices=[x.label for x in xyp_current_axis_options], value=xyp_current_axis_options[1].label, type="index", elem_id="xyp_x_type")
            xyp_x_values = gr.Textbox(label="X values", lines=1)

        with gr.Row():
            xyp_y_type = gr.Dropdown(label="Y type", choices=[x.label for x in xyp_current_axis_options], value=xyp_current_axis_options[0].label, type="index", elem_id="xyp_y_type")
            xyp_y_values = gr.Textbox(label="Y values", lines=1)

        # "Checkpoint Dropdown"
        with gr.Row(visible=False) as row_dd_ckpt:
            with gr.Column(scale=1):
                gr.HTML("<p style='max-width: 14em;'>* change 'type' to clear/reload dropdowns.</p>")
            with gr.Column(scale=3):
                dd_ckpt01 = gr.Dropdown(label="ckpt_01", choices=modules.sd_models.checkpoint_tiles(), visible=False)
                dd_ckpt02 = gr.Dropdown(label="ckpt_02", choices=modules.sd_models.checkpoint_tiles(), visible=False)
                dd_ckpt03 = gr.Dropdown(label="ckpt_03", choices=modules.sd_models.checkpoint_tiles(), visible=False)
                dd_ckpt04 = gr.Dropdown(label="ckpt_04", choices=modules.sd_models.checkpoint_tiles(), visible=False)
                dd_ckpt05 = gr.Dropdown(label="ckpt_05", choices=modules.sd_models.checkpoint_tiles(), visible=False)
                dd_ckpt06 = gr.Dropdown(label="ckpt_06", choices=modules.sd_models.checkpoint_tiles(), visible=False)
                dd_ckpt07 = gr.Dropdown(label="ckpt_07", choices=modules.sd_models.checkpoint_tiles(), visible=False)
                dd_ckpt08 = gr.Dropdown(label="ckpt_08", choices=modules.sd_models.checkpoint_tiles(), visible=False)
                dd_ckpt09 = gr.Dropdown(label="ckpt_09", choices=modules.sd_models.checkpoint_tiles(), visible=False)
                dd_ckpt10 = gr.Dropdown(label="ckpt_10", choices=modules.sd_models.checkpoint_tiles(), visible=False)
                dd_ckpt_list = [dd_ckpt01,dd_ckpt02,dd_ckpt03,dd_ckpt04,dd_ckpt05,dd_ckpt06,dd_ckpt07,dd_ckpt08,dd_ckpt09,dd_ckpt10]

        # Default xy_grid checkbox
        with gr.Row():
            xyp_include_lone_images = gr.Checkbox(label='Include Separate Images', value=True)
            xyp_no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=False)

        with gr.Row():
            xyp_draw_legends = gr.Radio(label="Draw Legends", choices=["None", "Both", "X", "Y"], type="index", value="Both")

        # Additional check box below.
        with gr.Row():
            xyp_restore_checkpoint = gr.Checkbox(label='Restore Checkpoint after process', value=True)
            xyp_show_only_favorite_axis_option = gr.Checkbox(label="Show only favorite Axis Option", value=False)
            xyp_save_info_in_grid = gr.Checkbox(label="Save PNGinfo to grid", value=True)

        #
        # Event
        #
        def on_change_xy_type(op_type, sub_type):
            op_type = axis_options[op_type]
            sub_type = axis_options[sub_type]
            if op_type.label == "Checkpoint Dropdown" and (sub_type.label == "Checkpoint Dropdown" or sub_type.label == "Checkpoint name"):
                return [gr.update(value="Nothing"), gr.update()] + [gr.update(visible=True)] + [gr.update()] * len(dd_ckpt_list)
            elif op_type.label != "Checkpoint Dropdown" and sub_type.label == "Checkpoint Dropdown":
                return [gr.update(), gr.update()] + [gr.update(visible=True)] + [gr.update()] * len(dd_ckpt_list)
            elif op_type.label == "Checkpoint Dropdown" and sub_type.label != "Checkpoint Dropdown":
                # すなおに描画
                return [gr.update(), gr.update()] + [gr.update(visible=True)] + [gr.update(value="", visible=True, choices=modules.sd_models.checkpoint_tiles())] + [gr.update(value="") for x in range(len(dd_ckpt_list)-1)]
            else:
                return [gr.update(), gr.update()] + [gr.update(visible=False)] + [ gr.update(value="", visible=False) for x in range(len(dd_ckpt_list)) ]

        xyp_x_type.change(
            fn=on_change_xy_type,
            inputs=[xyp_x_type, xyp_y_type],
            outputs=[xyp_x_type, xyp_x_values] + [row_dd_ckpt] + dd_ckpt_list
            )
        xyp_y_type.change(
            fn=on_change_xy_type,
            inputs=[xyp_y_type, xyp_x_type],
            outputs=[xyp_y_type, xyp_y_values] + [row_dd_ckpt] + dd_ckpt_list
            )

        def on_change_dd_ckpt(
            dd_ckpt01,dd_ckpt02,dd_ckpt03,dd_ckpt04,dd_ckpt05,dd_ckpt06,dd_ckpt07,dd_ckpt08,dd_ckpt09,dd_ckpt10,
            xyp_x_type, xyp_x_values, xyp_y_type, xyp_y_values
        ):
            dd_ckpt_list = [dd_ckpt01,dd_ckpt02,dd_ckpt03,dd_ckpt04,dd_ckpt05,dd_ckpt06,dd_ckpt07,dd_ckpt08,dd_ckpt09,dd_ckpt10]
            # find which dd selected, current row_index
            row_index = -1
            _output_txt = ""
            for index, _dd_ckpt in enumerate(dd_ckpt_list):
                print(f"dd_ckpt[{index}]: [{_dd_ckpt}]")
                if _dd_ckpt == "" or _dd_ckpt == None or _dd_ckpt == {}:
                    row_index = index
                    break
                else:
                    _model_info = modules.sd_models.get_closet_checkpoint_match(_dd_ckpt)
                    if _model_info:
                        _dd_ckpt = os.path.basename(_model_info.filename)
                        if _output_txt != "":
                            _output_txt += "\n" + _dd_ckpt
                        else:
                            _output_txt = _dd_ckpt
            # make update object
            if axis_options[xyp_x_type].label == "Checkpoint Dropdown":
                _ret = [gr.update(value=_output_txt), gr.update()]
            elif axis_options[xyp_y_type].label == "Checkpoint Dropdown":
                _ret = [gr.update(), gr.update(value=_output_txt)]
            else:
                _ret = [gr.update(), gr.update()]
            # make dd update
            if row_index == -1:
                _dd_ret = [ gr.update() for x in range(len(dd_ckpt_list)) ] + _ret
            else:
                _dd_ret = [ gr.update() for x in range(row_index) ] + [ gr.update(value="", visible=True, choices=modules.sd_models.checkpoint_tiles())] + [ gr.update() for x in range(len(dd_ckpt_list) - row_index - 1)] + _ret
            return _dd_ret
        dd_ckpt01.change(
            fn=on_change_dd_ckpt,
            inputs=dd_ckpt_list + [xyp_x_type, xyp_x_values] + [xyp_y_type, xyp_y_values],
            outputs=dd_ckpt_list + [xyp_x_values, xyp_y_values]
            )
        dd_ckpt02.change(fn=on_change_dd_ckpt, inputs=dd_ckpt_list + [xyp_x_type, xyp_x_values] + [xyp_y_type, xyp_y_values], outputs=dd_ckpt_list + [xyp_x_values, xyp_y_values])
        dd_ckpt03.change(fn=on_change_dd_ckpt, inputs=dd_ckpt_list + [xyp_x_type, xyp_x_values] + [xyp_y_type, xyp_y_values], outputs=dd_ckpt_list + [xyp_x_values, xyp_y_values])
        dd_ckpt04.change(fn=on_change_dd_ckpt, inputs=dd_ckpt_list + [xyp_x_type, xyp_x_values] + [xyp_y_type, xyp_y_values], outputs=dd_ckpt_list + [xyp_x_values, xyp_y_values])
        dd_ckpt05.change(fn=on_change_dd_ckpt, inputs=dd_ckpt_list + [xyp_x_type, xyp_x_values] + [xyp_y_type, xyp_y_values], outputs=dd_ckpt_list + [xyp_x_values, xyp_y_values])
        dd_ckpt06.change(fn=on_change_dd_ckpt, inputs=dd_ckpt_list + [xyp_x_type, xyp_x_values] + [xyp_y_type, xyp_y_values], outputs=dd_ckpt_list + [xyp_x_values, xyp_y_values])
        dd_ckpt07.change(fn=on_change_dd_ckpt, inputs=dd_ckpt_list + [xyp_x_type, xyp_x_values] + [xyp_y_type, xyp_y_values], outputs=dd_ckpt_list + [xyp_x_values, xyp_y_values])
        dd_ckpt08.change(fn=on_change_dd_ckpt, inputs=dd_ckpt_list + [xyp_x_type, xyp_x_values] + [xyp_y_type, xyp_y_values], outputs=dd_ckpt_list + [xyp_x_values, xyp_y_values])
        dd_ckpt09.change(fn=on_change_dd_ckpt, inputs=dd_ckpt_list + [xyp_x_type, xyp_x_values] + [xyp_y_type, xyp_y_values], outputs=dd_ckpt_list + [xyp_x_values, xyp_y_values])
        dd_ckpt10.change(fn=on_change_dd_ckpt, inputs=dd_ckpt_list + [xyp_x_type, xyp_x_values] + [xyp_y_type, xyp_y_values], outputs=dd_ckpt_list + [xyp_x_values, xyp_y_values])

        def select_axis_list(is_selected):
            _current_axis_options = [x for x in axis_options if type(x) == AxisOption or type(x) == AxisOptionImg2Img and is_img2img]
            if is_selected:
                _current_axis_options = [x for x in _current_axis_options if x.label in FAVORITE_AXISOPTION_NAMES]
            labels = [x.label for x in _current_axis_options]
            return gr.update(choices=labels, value=xyp_x_type.value if xyp_x_type.value in labels else labels[1]), gr.update(choices=labels, value=xyp_y_type.value if xyp_y_type.value in labels else labels[0]),

        xyp_show_only_favorite_axis_option.change(
            fn=select_axis_list,
            inputs=[xyp_show_only_favorite_axis_option],
            outputs=[xyp_x_type, xyp_y_type]
        )

        return [xyp_x_type, xyp_x_values, xyp_y_type, xyp_y_values, xyp_draw_legends, xyp_include_lone_images, xyp_no_fixed_seeds, xyp_start_from, xyp_restore_checkpoint, xyp_show_only_favorite_axis_option, xyp_save_info_in_grid]


    def run(self, p, x_type, x_values, y_type, y_values, draw_legend, include_lone_images, no_fixed_seeds, start_from, restore_checkpoint, show_only_favorite_axis_option, save_info_in_grid):
        if not no_fixed_seeds:
            modules.processing.fix_seed(p)

        if not opts.return_grid:
            # why? if "Show grid in results for web" disabled, why batch_size go down to 1?
            p.batch_size = 1

        def process_axis(opt, vals):
            if opt.label == 'Nothing':
                return [0]

            if opt.label == 'Sampler' and vals.strip() == '*':
                if p.enable_hr is True:
                    vals = ','.join([x.name for x in modules.sd_samplers.samplers_for_img2img])
                else:
                    vals = ','.join([x.name for x in modules.sd_samplers.all_samplers])

            valslist = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(vals)))]

            if opt.type == int:
                valslist_ext = []

                for val in valslist:
                    m = re_range.fullmatch(val)
                    mc = re_range_count.fullmatch(val)
                    mr_ssc = re_range_seed_step_count.fullmatch(val)
                    mr_sec = re_range_seed_count.fullmatch(val)
                    mr_stc = re_range_step_count.fullmatch(val)
                    mr_oly = re_range_only.fullmatch(val)
                    if m is not None:
                        start = int(m.group(1))
                        end = int(m.group(2))+1
                        step = int(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += list(range(start, end, step))
                    elif mc is not None:
                        start = int(mc.group(1))
                        end   = int(mc.group(2))
                        num   = int(mc.group(3)) if mc.group(3) is not None else 1

                        valslist_ext += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
                    elif mr_ssc is not None:  # SSC
                        direction = -1 if mr_ssc.group(1) == "-" else 1
                        start = int(mr_ssc.group(2))
                        step = mr_ssc.group(3)
                        if step == "":
                            step = p.batch_size if p.batch_size else 1
                        else:
                            step = int(step)
                        num = int(mr_ssc.group(4))
                        dprint(f"mr_ssc start:{start} step:{step} num:{num} direction:{direction}")
                        if opt.label == "Seed" and start == -1 and no_fixed_seeds:
                            valslist_ext += [-1 for x in range(num)]
                        else:
                            end = int(start + step * num * direction)
                            if end < 0:
                                direction *= -1
                                end = int(start + step * num * direction)
                            dprint(f"mr_ssc val:{val}, start:{start} end:{end} step:{step} num:{num} direction:{direction}")
                            valslist_ext += [ int(start + step * x * direction) for x in range(num) ]
                    elif mr_sec is not None:  # SEC
                        direction = -1 if mr_sec.group(1) == "-" else 1
                        start = int(mr_sec.group(2))
                        num = int(mr_sec.group(3))
                        dprint(f"mr_sec start:{start} direction:{direction}")
                        if opt.label == "Seed" and start == -1 and no_fixed_seeds:
                            valslist_ext += [-1 for x in range(num)]
                        else:
                            if opt.label == "Seed":
                                start = int(random.randrange(SEED_MAX)) if start is None or start == '' or start == -1 else start
                            step = 1
                            end = int(start + step * num * direction)
                            if end < 0:
                                direction *= -1
                                end = int(start + step * num * direction)
                            dprint(f"mr_sec val:{val}, start:{start} end:{end} step:{step} num:{num} direction:{direction}")
                            valslist_ext += [ int(start + step * x * direction) for x in range(num) ]
                    elif mr_stc is not None:  # STC
                        direction = -1 if mr_stc.group(1) == "-" else 1
                        step = mr_stc.group(2)
                        if step == "":
                            step = p.batch_size if p.batch_size else 1
                        else:
                            step = int(step)
                        num = int(mr_stc.group(3))
                        dprint(f"mr_stc step:{step} num:{num} direction:{direction}")
                        if opt.label == "Seed" and p.seed == -1 and no_fixed_seeds:
                            valslist_ext += [-1 for x in range(num)]
                            dprint(f"mr_stc seed=-1 because of no_fixed_seeds")
                        else:
                            if opt.label == "Seed":
                                start = int(random.randrange(SEED_MAX)) if p.seed is None or p.seed == '' or p.seed == -1 else int(p.seed)
                            else:
                                start = 0
                            end = int(start + step * num * direction)
                            if end < 0:
                                direction *= -1
                                end = int(start + step * num * direction)
                            dprint(f"mr_stc val:{val}, start:{start} end:{end} step:{step} num:{num} direction:{direction}")
                            valslist_ext += [ int(start + step * x * direction) for x in range(num) ]
                    elif mr_oly is not None:
                        direction = -1 if mr_oly.group(1) == "-" else 1
                        num = int(mr_oly.group(2))
                        dprint(f"mr_oly num:{num} direction:{direction}")
                        if opt.label == "Seed" and p.seed == -1 and no_fixed_seeds:
                            valslist_ext += [-1 for x in range(num)]
                        else:
                            if opt.label == "Seed":
                                start = int(random.randrange(SEED_MAX)) if p.seed is None or p.seed == '' or p.seed == -1 else int(p.seed)
                            else:
                                start = 0
                            step = 1
                            end = int(start + step * num * direction)
                            if end < 0:
                                direction *= -1
                                end = int(start + step * num * direction)
                            dprint(f"mr_oly val:{val}, start:{start} end:{end} step:{step} num:{num} direction:{direction}")
                            valslist_ext += [ int(start + step * x * direction) for x in range(num) ]
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type == float:
                valslist_ext = []

                for val in valslist:
                    m = re_range_float.fullmatch(val)
                    mc = re_range_count_float.fullmatch(val)
                    if m is not None:
                        start = float(m.group(1))
                        end = float(m.group(2))
                        step = float(m.group(3)) if m.group(3) is not None else 1

                        valslist_ext += np.arange(start, end + step, step).tolist()
                    elif mc is not None:
                        start = float(mc.group(1))
                        end   = float(mc.group(2))
                        num   = int(mc.group(3)) if mc.group(3) is not None else 1

                        valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
                    else:
                        valslist_ext.append(val)

                valslist = valslist_ext
            elif opt.type == str_permutations:
                valslist = list(permutations(valslist))

            valslist = [opt.type(x) for x in valslist]

            # Confirm options are valid before starting
            if opt.confirm:
                opt.confirm(p, valslist)

            return valslist

        x_opt = axis_options[x_type]
        xs = process_axis(x_opt, x_values)

        y_opt = axis_options[y_type]
        ys = process_axis(y_opt, y_values)

        # 'Checkpoint name' or 'Checkpoint Dropdown'
        def get_axis_val_and_label(p, axis_opt, _s):
            if axis_opt.label == 'Checkpoint name' or axis_opt.label == 'Checkpoint Dropdown':
                _vals = [_.split('#')[0].strip() if '#' in _ else _ for _ in _s]
                _labels = [_.split('#')[1].strip() if '#' in _ else _ for _ in _s]
            else:
                _vals = _s
                _labels = _s
            return [_vals, _labels]

        xs, x_labels = get_axis_val_and_label(p, x_opt, xs)
        ys, y_labels = get_axis_val_and_label(p, y_opt, ys)

        # 'Seed','Var. seed'
        def fix_axis_seeds(axis_opt, axis_list):
            if axis_opt.label in ['Seed','Var. seed']:
                if len(axis_list) > 0:
                    return [int(random.randrange(SEED_MAX)) if val is None or val == '' or val == -1 else val for val in axis_list]
                else:
                    return [int(random.randrange(SEED_MAX))]
            else:
                return axis_list

        if not no_fixed_seeds:
            xs = fix_axis_seeds(x_opt, xs)
            ys = fix_axis_seeds(y_opt, ys)

        # 'Steps'
        if x_opt.label == 'Steps':
            total_steps = sum(xs) * len(ys)
        elif y_opt.label == 'Steps':
            total_steps = sum(ys) * len(xs)
        else:
            total_steps = p.steps * len(xs) * len(ys)

        if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
            total_steps *= 2

        print(f"{TITLE} will create {len(xs) * len(ys) * p.n_iter} images on a {len(xs)}x{len(ys)} grid.")
        print(f"(Total steps to process: {total_steps * p.n_iter})")
        shared.total_tqdm.updateTotal(total_steps * p.n_iter)

        def cell(x, y):
            pc = copy(p)
            x_opt.apply(pc, x, xs)
            y_opt.apply(pc, y, ys)

            return process_images(pc)

        with SharedSettingsStackHelper():
            processed = draw_xy_grid(
                p,
                xs=xs,
                ys=ys,
                x_labels=[x_opt.format_value(p, x_opt, x) for x in x_labels],
                y_labels=[y_opt.format_value(p, y_opt, y) for y in y_labels],
                cell=cell,
                draw_legend=draw_legend,
                include_lone_images=include_lone_images,
                start_from=start_from
            )

        if opts.grid_save and processed:
            if save_info_in_grid:
                # workaround for p.all_negative_prompts == None
                if not p.all_negative_prompts:
                    p.all_negative_prompts = processed.all_negative_prompts
                infotext = processed.infotext(p, 0)
            else:
                infotext = None
            images.save_image(processed.images[0], p.outpath_grids, GRID_FILENAME_HEADER, info=infotext, extension=opts.grid_format, prompt=p.prompt, seed=processed.seed, grid=True, p=p)

        # restore checkpoint in case it was changed by axes
        if restore_checkpoint and not state.interrupted:
            modules.sd_models.reload_model_weights(shared.sd_model)

        if state.interrupted:
            state.interrupted = False

        return processed
