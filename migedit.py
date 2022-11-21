"""MIG Editor. CLI/importable module to automatically delete and create MIG gpu/cpu instances."""

__version__ = "0.1.0"

import argparse
import os
import time

from subprocess import Popen, PIPE


MIG_OPTIONS = {
    "1": "1g.5gb",
    "2": "2g.10gb",
    "3": "3g.20gb",
    "4": "4g.20gb",
    "7": "7g.40gb",
}


def execute_command(cmd, shell=False, vars={}):
    """
    Executes a (non-run) command.
    :param cmd -> Command to run.

    :return result -> Text printed by the command.
    """
    if isinstance(cmd, str):
        cmd = cmd.split()

    env = os.environ.copy()
    for k, v in vars.items():
        env[k] = str(v)

    result = []
    with Popen(
        cmd, stdout=PIPE, bufsize=1, universal_newlines=True, shell=shell, env=env
    ) as p:
        result.extend(p.stdout)

        if p.returncode != 0:
            pass

    return result


def get_dcgmi_instance_id(device, gpu_instance):
    for line in execute_command("dcgmi discovery -c"):
        if "EntityID" not in line:
            continue

        if line.count("GPU") != 1:
            continue

        if f"{device}/{gpu_instance}" in line:
            entity_id = int(line.split("EntityID:")[1].split(")")[0].strip())
            return f"i:{entity_id}"
    return None


def get_mig_ids(device):
    gpu = None
    ids = set()
    for line in execute_command("nvidia-smi -L"):
        if "UUID: GPU" in line:
            gpu = int(line.split("GPU")[1].split(":")[0].strip())
            continue
        elif "UUID: MIG" in line:
            if gpu == int(device):
                ids.add(line.split("UUID:")[1].split(")")[0].strip())

    return ids


def make_mig_devices(gpu: str, profiles: list, verbose=False) -> tuple:

    # Remove old instances
    result = "".join(execute_command(f"sudo nvidia-smi mig -dci")).lower()
    if "failed" in result or "unable" in result:
        raise ValueError(result)

    result = "".join(execute_command(f"sudo nvidia-smi mig -dgi")).lower()
    if "failed" in result or "unable" in result:
        raise ValueError(result)

    time.sleep(1)

    devicetemp = []

    # Create new instances
    for index, profile in enumerate(profiles):
        instance = profile.lower()

        if instance in MIG_OPTIONS:
            instance = MIG_OPTIONS[instance]

        if instance in MIG_OPTIONS.values():
            other_mig_ids = get_mig_ids(gpu)

            result = "".join(
                execute_command(
                    f"sudo nvidia-smi mig -i {int(device)} -cgi {instance} -C"
                )
            ).lower()

            gpu_instance = result.split("gpu instance id")[1].split("on gpu")[0].strip()

            if "failed" in result or "unable" in result:
                raise ValueError(result)

            mig_id = list(get_mig_ids(device) - other_mig_ids)[0]

            devicetemp.append((index, instance, mig_id, device, gpu_instance))

        time.sleep(
            2.0
        )  # Wait for the rather slow DCGMI discovery to yield a GPU entity id

    results = []

    for (index, device, instance, gpu_instance, mig_id) in devicetemp:

        while (entity_id := get_dcgmi_instance_id(device, gpu_instance)) is None:
            time.sleep(
                0.5
            )  # Wait for the rather slow DCGMI discovery to yield a GPU entity id

        print(
            f"Device {device}: Allocated {instance} ({gpu_instance}) with entity id ({entity_id}) and UUID ({mig_id})"
        )
        results.append((device, instance, gpu_instance, entity_id, mig_id))

    return results


def cli():
    parser = argparse.ArgumentParser(
        description="MIG Editor. CLI/importable module to automatically delete and create MIG gpu/cpu instances.",
    )
    parser.add_argument(
        "-i",
        "--instance",
        metavar="GPU",
        help="ID instance number of the GPU to manage MIG on.",
    )
    parser.add_argument(
        "-p",
        "--profiles",
        metavar="PROFILES",
        nargs="+",
        help="""Space seperated list of MIG profiles to use. Options: "1" or "1g.5gb", "2" or "2g.10gb", "3" or "3g.20gb", "4" or "4g.20gb", "7" or "7g.40gb".""",
        required=True,
    )

    args = parser.parse_args()

    make_mig_devices(args.instance, args.profiles)


if __name__ == "__main__":
    cli()
