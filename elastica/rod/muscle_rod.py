__doc__ = """ Muscle rod base classes and implementation details that need to be hidden from the user"""
__all__ = ["MuscularRod"]

from elastica import IMPORT_NUMBA

if IMPORT_NUMBA:
    from elastica._elastica_numba._rod._muscular_rod import MuscularRod
else:
    from elastica._elastica_numpy._rod._muscular_rod import MuscularRod
