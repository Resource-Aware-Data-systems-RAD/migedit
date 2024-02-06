# migedit
MIG Editor python CLI tool and bindings for MIG device management.
Supports all MIG devices and supports both 1:1 gpu instance compute instances setups and setups where compute instances share gpu instances.

- [migedit on Github](https://github.com/Resource-Aware-Data-Systems-RAD/migedit)
- [migedit on PyPI](https://pypi.org/project/migedit/)

# Usage (CLI)

`migedit -i 2 -p 1 2 3`
to create three instances (1g.5gb, 2g.10gb, 3g.20gb) on GPU 2.

or

`migedit -i 2 -p s2 s2`
to create two shared memory instances (2c.7g.40gb, 2c.7g.40gb) on GPU 2.

# Usage (Python bindings)

`import migedit`
to import migedit in your project.

`migedit.get_mig_profiles()`
to get a list of all available MIG instance configurations.

`migedit.make_mig_devices(0, ["1g.10gb"])`
to remove old MIG instances and create new ones.

# Changelog:'
- 3.4: Resolve ValueError when no MIG configs are found.
- 3.3: Removed sudo dependency on nvidia-smi.
- 3.2: Added `remove_mig_devices()`. Empty command will now remove instances only. Added parsing of comma separated values.
- 3.1: Added `remove_old` flag.
- 3.0: Support for non-A100 devices (H100, A30, etc.) by dynamically grabbing available profiles.
- 2.0: Support for Shared Memory Mig Mode (with 7g.40gb instances)
- 1.1: Various bugfixes
- 1.0: Initial

## Supported platforms

- [x] Linux (Python 3.10 or higher)

## Contributors

- [Ties Robroek](https://github.com/sipondo)

Thank You!

Contributions are welcome. _(Please add yourself to the list)_
