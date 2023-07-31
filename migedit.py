"""MIG Editor. CLI/importable module to automatically delete and create MIG gpu/cpu instances."""

__version__ = "0.3.0"

import argparse
import os
import re
import time

from subprocess import Popen, PIPE

RE_OPTIONS = re.compile("((\S*)g\.\S*)")
NEWTAB = "\n\t"


def _execute_command(cmd: str | list, shell: bool = False, vars: dict = {}) -> list:
    """Executes a command

    Args:
        cmd (str | list): Command to run
        shell (bool, optional): Whether to run with shell enabled. Defaults to False.
        vars (dict, optional): Environment variables to pass on. Defaults to {}.

    Returns:
        list: Command output
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


def _get_dcgmi_instance_id(device: int, gpu_instance: int) -> str | None:
    """Get entity ID of GPU instance

    Args:
        device (int): GPU device to search
        gpu_instance (int): GPU instance to search for

    Returns:
        str | None: DCGMI instance id
    """

    for line in _execute_command("dcgmi discovery -c"):
        if "EntityID" not in line:
            continue

        if line.count("GPU") != 1:
            continue

        if f"{device}/{gpu_instance}" in line:
            entity_id = int(line.split("EntityID:")[1].split(")")[0].strip())
            return f"i:{entity_id}"
    return None


def _get_mig_ids(device: int) -> set:
    """Get NVIDIA SMI ids for all MIG compute instances on the device

    Args:
        device (int): GPU device to get instances of

    Returns:
        set: Set of id strings of all instances
    """

    gpu = None
    ids = set()
    for line in _execute_command("nvidia-smi -L"):
        if "UUID: GPU" in line:
            gpu = int(line.split("GPU")[1].split(":")[0].strip())
            continue
        elif "UUID: MIG" in line:
            if gpu == device:
                ids.add(line.split("UUID:")[1].split(")")[0].strip())

    return ids


def _list_transpose(input_list: list) -> list:
    """Transpose a list containing iterables

    Args:
        input_list (list): List of iterables to transpose

    Returns:
        list: Transposed list
    """
    return [list(i) for i in zip(*input_list)]


def make_mig_devices(
    gpu: int, profiles: list, available_mig: list = [], available_smig: list = []
) -> list:
    """Remove any old MIG instances and create MIG instances on gpu following profiles

    Args:
        gpu (int): GPU instance to create on
        profiles (list): List of profiles to use
        available_mig (list, optional): Available MIG profiles. Automatically retrieves profiles if empty
        available_smig (list, optional): Available shared MIG profiles. Automatically retrieves profiles if empty

    Raises:
        ValueError: Command failed

    Returns:
        list: Allocated MIG instances

    """

    # Retrieve missing profile availability data
    if (not available_mig) or (not available_smig):
        options_mig, options_smig = get_mig_profiles()

        if not available_mig:
            available_mig = options_mig
        if not available_smig:
            available_smig = options_smig

    # Remove old instances
    result = "".join(_execute_command(f"sudo nvidia-smi mig -dci")).lower()
    if "failed" in result or "unable" in result:
        raise ValueError(result)

    result = "".join(_execute_command(f"sudo nvidia-smi mig -dgi")).lower()
    if "failed" in result or "unable" in result:
        raise ValueError(result)

    time.sleep(1)

    devicetemp = []

    # TODO: remove this temp code
    shared_mig_gpui = {}

    # This supports Multi-MIG, which doesn't actually work right now
    # Tested on PyTorch (and failed), perhaps works for different frameworks
    for index, profile in enumerate(profiles):
        instance = profile.lower().strip()

        if instance in available_mig:
            instance = available_mig[instance]
        elif instance in available_smig:
            instance = available_smig[instance]

        # Normal MIG
        if instance in available_mig.values():
            other_mig_ids = _get_mig_ids(gpu)

            result = "".join(
                _execute_command(
                    f"sudo nvidia-smi mig -i {int(gpu)} -cgi {instance} -C"
                )
            ).lower()

            gpu_instance = result.split("gpu instance id")[1].split("on gpu")[0].strip()

            if "failed" in result or "unable" in result:
                raise ValueError(result)

            mig_id = _get_mig_ids(gpu) - other_mig_ids

            devicetemp.append((index, gpu, instance, gpu_instance, mig_id))

        elif instance in available_smig.values():
            # Shared MIG, TODO: make more flexible
            if gpu not in shared_mig_gpui:
                result = "".join(
                    _execute_command(
                        f"sudo nvidia-smi mig -i {gpu} -cgi {list(available_mig.values())[-1]}"
                    )
                ).lower()
                instance_id = result.split("gpu instance id")[1].split("on")[0].strip()
                shared_mig_gpui[gpu] = instance_id

            other_mig_ids = _get_mig_ids(gpu)

            gpu_instance = shared_mig_gpui[gpu]

            result = "".join(
                _execute_command(
                    f"sudo nvidia-smi mig -gi {gpu_instance} -cci {list(available_smig.values()).index(instance)}"
                )
            ).lower()

            if "failed" in result or "unable" in result:
                raise ValueError(result)

            mig_id = _get_mig_ids(gpu) - other_mig_ids

            devicetemp.append((index, gpu, instance, gpu_instance, mig_id))

        time.sleep(
            2.0
        )  # Wait for the rather slow DCGMI discovery to yield a GPU entity id

    results = []

    for index, device, instance, gpu_instance, mig_id in devicetemp:
        while (entity_id := _get_dcgmi_instance_id(device, gpu_instance)) is None:
            time.sleep(
                0.5
            )  # Wait for the rather slow DCGMI discovery to yield a GPU entity id

        print(
            f"Device {device}: Allocated {instance} ({gpu_instance}) with entity id ({entity_id}) and UUID ({mig_id})"
        )
        results.append((device, instance, gpu_instance, entity_id, mig_id))

    return results


def get_mig_profiles() -> tuple[dict, dict]:
    """Retrieve all available mig profiles

    Returns:
        tuple[dict, dict]: All available MIG and shared MIG profiles
    """

    # Get profile lists
    list_mig, scales = _list_transpose(
        [
            x[0]
            for line in _execute_command("nvidia-smi mig -lgip")
            if (x := RE_OPTIONS.findall(line))
        ]
    )
    list_smig = [
        f"{compute}c.{list_mig[-1]}" for compute in sorted(list(set(scales)))[:-1]
    ]

    # Convert to dicts
    available_mig, available_smig = {}, {}
    for profile in list_mig:
        original_prefix = profile.split("g")[0]
        prefix = original_prefix

        i = 0
        while prefix in available_mig:
            i += 1
            prefix = f"{original_prefix}.{i}"

        available_mig[prefix] = profile

    for profile in list_smig:
        original_prefix = "s" + profile.split("c")[0]
        prefix = original_prefix

        i = 0
        while prefix in available_mig:
            i += 1
            prefix = f"{original_prefix}.{i}"

        available_smig[prefix] = profile

    return available_mig, available_smig


def _cli():
    available_mig, available_smig = get_mig_profiles()

    parser = argparse.ArgumentParser(
        description="MIG Editor. CLI/importable module to automatically delete and create MIG gpu/cpu instances.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--instance",
        type=int,
        metavar="GPU",
        default=0,
        help="ID instance number of the GPU to manage MIG on.",
    )
    parser.add_argument(
        "-p",
        "--profiles",
        metavar="PROFILES",
        nargs="+",
        help=f"""Space seperated list of MIG profiles to use.\nOptions:\n\t{f"{NEWTAB}".join([f'{k: <3} or {v}' for k,v in (available_mig | available_smig).items()])}""",
        required=True,
    )

    args = parser.parse_args()

    make_mig_devices(args.instance, args.profiles, available_mig, available_smig)


if __name__ == "__main__":
    _cli()
