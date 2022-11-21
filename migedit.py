import argparse
import os
import time

from subprocess import Popen, PIPE


MIG_OPTIONS = ["1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"]


def execute_command(cmd, shell=False, vars={}):
    """
    Executes a (non-run) command.
    :param cmd -> Command to run.

    :return result -> Text printed by the command.
    """
    if isinstance(cmd, str):
        cmd = cmd.split()

    if vars:
        print(f"Executing command - {cmd} - {vars}")
    else:
        print(f"Executing command - {cmd}")

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


def make_mig_devices(df_workload, dev_table):

    # Remove MPS
    result = "".join(
        execute_command(["echo quit | nvidia-cuda-mps-control"], shell=True)
    ).lower()

    # Remove old instances
    result = "".join(execute_command(f"sudo nvidia-smi mig -dci")).lower()
    if "failed" in result or "unable" in result:
        raise ValueError(result)

    result = "".join(execute_command(f"sudo nvidia-smi mig -dgi")).lower()
    if "failed" in result or "unable" in result:
        raise ValueError(result)

    time.sleep(1)

    mig_table = dev_table.copy()
    entity_table = dev_table.copy()

    devicetemp = []

    # Create new instances
    for index, (device, profile) in df_workload[["Devices", "Collocation"]].iterrows():
        instance = profile.lower()

        if profile == "mps":
            # TODO: ACTIVATE MPS
            pass

        elif instance in MIG_OPTIONS:
            other_mig_ids = get_mig_ids(device)

            result = "".join(
                execute_command(
                    f"sudo nvidia-smi mig -i {int(device)} -cgi {profile} -C"
                )
            ).lower()

            gpu_instance = result.split("gpu instance id")[1].split("on gpu")[0].strip()

            if "failed" in result or "unable" in result:
                raise ValueError(result)

            mig_id = get_mig_ids(device) - other_mig_ids

            devicetemp.append((index, mig_id, device, gpu_instance))

        time.sleep(
            2.0
        )  # Wait for the rather slow DCGMI discovery to yield a GPU entity id

    for (index, mig_id, device, gpu_instance) in devicetemp:

        while (entity_id := get_dcgmi_instance_id(device, gpu_instance)) is None:
            time.sleep(
                0.5
            )  # Wait for the rather slow DCGMI discovery to yield a GPU entity id

        mig_table[index] = frozenset(mig_id)
        entity_table[index] = frozenset((entity_id,))

    return mig_table, entity_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIG Editor")
    parser.add_argument("-i", "--instance", metavar="GPU", help="gpu ID")
    parser.add_argument(
        "-p",
        "--profiles",
        metavar="PROFILES",
        nargs="+",
        help="MiG profiles",
        required=True,
    )

    parser.parse_args()
