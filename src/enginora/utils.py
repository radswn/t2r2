import inspect
import functools

def ignore_unmatched_kwargs(f):
    """
    Make function ignore unmatched kwargs.
    
    If the function already has the catch all **kwargs, do nothing.
    """
    if contains_var_kwarg(f):
        return f
    
    @functools.wraps(f)
    def inner(*args, **kwargs):
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if is_kwarg_of(key, f)
        }
        return f(*args, **filtered_kwargs)
    return inner


def contains_var_kwarg(f):
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in inspect.signature(f).parameters.values()
    )

def is_kwarg_of(key, f):
    param = inspect.signature(f).parameters.get(key, False)
    return param and (
        param.kind is inspect.Parameter.KEYWORD_ONLY or
        param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )