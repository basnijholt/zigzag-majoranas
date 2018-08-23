import utils
import dill
import hashlib
import types
import inspect
import functools


def hash_dict(dictionary):
    temp_dict = dict()
    _hash = 0

    for k, v in dictionary.items():
        hashed_k = consistent_hash(k)
        hashed_v = consistent_hash(v)

        temp_dict[hashed_k] = hashed_v

    for k, v in sorted(temp_dict.items()):
        _hash += consistent_hash(k + v + _hash)

    return _hash


def hash_tuple(obj):
    _hash = 0

    for item in obj:
        _hash += consistent_hash(str(consistent_hash(item) + _hash))

    return _hash


def hash_list(obj):
    return consistent_hash(tuple(obj))


def hash_int(obj):
    return hash(obj)


def hash_float(obj):
    return hash(obj)


def hash_str(obj):
    m = hashlib.md5()
    m.update(bytes(obj, 'utf32'))
    _hash = int.from_bytes(m.digest(), byteorder='little', signed=True)
    return _hash


def hash_function(obj):
    return consistent_hash(inspect.getsource(obj))


def hash_builtin_function(obj):
    return consistent_hash(obj.__name__)


def hash_partial_function(obj):
    return consistent_hash(obj.func)


def consistent_hash(obj):
    hash_funcs = {
        dict: hash_dict,
        tuple: hash_tuple,
        list: hash_list,
        int: hash_int,
        float: hash_float,
        str: hash_str,
        types.FunctionType: hash_function,
        types.BuiltinFunctionType: hash_builtin_function,
        bool: hash_int,
        functools.partial: hash_partial_function
    }
    return hash_funcs[type(obj)](obj)
