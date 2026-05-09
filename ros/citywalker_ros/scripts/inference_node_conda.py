#!/usr/bin/env python3
import os
import subprocess
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_script = os.path.join(script_dir, "inference_node.py")
    conda_setup = os.environ.get(
        "CITYWALKER_CONDA_SETUP",
        "/media/isee324/2a90eb70-2d62-4af4-b04a-0fcdde4122a5/anaconda3/etc/profile.d/conda.sh",
    )
    conda_env = os.environ.get("CITYWALKER_CONDA_ENV", "dino")
    ros_python = "/opt/ros/noetic/lib/python3/dist-packages"
    system_python = "/usr/lib/python3/dist-packages"
    trt_python = "/usr/lib/python3.8/dist-packages"
    existing_pythonpath = os.environ.get("PYTHONPATH", "")

    command = (
        "source '{conda_setup}' && "
        "conda activate '{conda_env}' && "
        "export PYTHONPATH='{ros_python}:{system_python}:{trt_python}:$PYTHONPATH:{existing_pythonpath}' && "
        "exec python '{inference_script}'"
    ).format(
        conda_setup=conda_setup,
        conda_env=conda_env,
        ros_python=ros_python,
        system_python=system_python,
        trt_python=trt_python,
        existing_pythonpath=existing_pythonpath,
        inference_script=inference_script,
    )

    os.execvp("bash", ["bash", "-lc", command])


if __name__ == "__main__":
    main()
