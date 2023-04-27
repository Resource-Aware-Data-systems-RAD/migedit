# migedit
MIG Editor python CLI tool and bindings for MIG device management

Usage:

`migedit -i 2 -p 1 2 3`
to create three instances (1g.5gb, 2g.10gb, 3g.20gb) on GPU 2.

or

`migedit -i 2 -p s2 s2`
to create two shared memory instances (2c.7g.40gb, 2c.7g.40gb) on GPU 2.


# Changelog:

- 2.0: Support for Shared Memory Mig Mode (with 7g.40gb instances)
- 1.1: Various bugfixes
- 1.0: Initial
