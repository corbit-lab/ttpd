"""Support modules for running the solvers -- not part of the method.

``paths`` selects which solver stack to put on sys.path; ``hub`` resolves model
weights and benchmark data, downloading from the Hugging Face Hub when they are
not already on disk. Import them via the re-exports on the parent package:

    from ttpd import _paths
    from ttpd.hub import ensure_local
"""
