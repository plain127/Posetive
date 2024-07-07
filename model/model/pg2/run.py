#!/usr/bin/env python3

"""
run.py is used to run engine.
In fact we can run each implementation(engine) separately,
but run each implementation in a unified way can avoid some duplicate code,
such as prepare gpu, read and save config file, and so on.
"""
import collections
from argparse import ArgumentParser
from importlib import import_module
from os import path, makedirs
from pprint import pprint

import sys
import toml
import torch
# import generate

# python 3.8+ compatibility
try:
    collectionsAbc = collections.abc
except:
    collectionsAbc = collections


def update_config(config, config_update):
    for k, v in config_update.items():
        dv = config.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            config[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            config[k] = update_config(dv, v)
        else:
            config[k] = v
    return config


# each line represent a engine: key is name, value is the import path for this engine.
IMPLEMENTED_ENGINE = {
    "PG2-1": "implementations.PG2.train1",
    "PG2-2": "implementations.PG2.train2",
    "PG2-Generator": "generate"
}

#수정3. dafault 값 변경
def parse_argument(generation_id):
    parser = ArgumentParser("Train")
    parser.add_argument("implementation", type=str, default="PG2-Generator", choices=IMPLEMENTED_ENGINE.keys(), help="run which?")
    parser.add_argument("-g", '--gpu_id', default=0, type=int, help='gpu_id: e.g. 0', required=True)
    parser.add_argument("-c", "--config", type=str, default="/model/pg2/implementations/PG2/stage2.toml", help="config file path", required=True)
    parser.add_argument("-o", "--output", default=f"/model/generation/{generation_id}/result/" ,type=str, help="output path", required=True)
    parser.add_argument("-t", "--toml", action="append", type=str, help="overwrite toml config use cli arg")
    options = parser.parse_args()

    return options


def prepare_gpu(gpu_ids):
    torch.cuda.set_device(gpu_ids)
    torch.backends.cudnn.benchmark = True


def load_config(config_path, overwrite_tomls, generation_id):
    print("reading config from <{}>\n".format(path.abspath(config_path)))
    try:
        #수정2. config path를 동적으로 받는다.
        with open(config_path, "+r") as f:
            config = toml.load(f)
            config["dataset"]["path"]["test"]["image"] = f"/model/generation/{generation_id}/"
            config["dataset"]["path"]["test"]["bone"] = f"/model/generation/{generation_id}/pose_map_image/"
            config["dataset"]["path"]["test"]["mask"] = f"/model/generation/{generation_id}/pose_mask_image/"
            config["dataset"]["path"]["test"]["pair"] = f"/model/generation/{generation_id}/pairs.csv"
            config["dataset"]["path"]["test"]["annotation"] = f"/model/generation/{generation_id}/annotation.csv"
            f.seek(0) 
            toml.dump(config, f) 
            f.truncate()
            if overwrite_tomls is not None:
                config_update = toml.loads("\n".join(overwrite_tomls))
                print(config_update)
                config = update_config(config, config_update)
            return config
    except FileNotFoundError as e:
        print("can not find config file")
        raise e


def save_config(config, output_folder):
    if not path.exists(output_folder):
        makedirs(output_folder)
    with open(path.join(output_folder, "train.toml"), "w") as f:
        toml.dump(config, f)

#수정1. 
#add generation_id attribute
def main(generation_id):
    options_config = "/model/pg2/implementations/PG2/stage2.toml"
    options_gpu_id = 0
    options_implementation = "PG2-Generator"
    options_output = f"/model/generation/{generation_id}/result/"
    options_toml = None

    prepare_gpu(options_gpu_id)

    config = load_config(options_config, options_toml, generation_id)
    config["output"] = options_output
    pprint(config)
    save_config(config, config["output"])

    sys.path.append("../model/pg2")
    engine = import_module(IMPLEMENTED_ENGINE[options_implementation])
    if IMPLEMENTED_ENGINE[options_implementation] == "generate":
        config["engine"] = options_implementation
    else:
        save_config(config, config["output"])

    print("#" * 80, "\n")

    engine.run(config)

if __name__ == '__main__':
    main(generation_id)
