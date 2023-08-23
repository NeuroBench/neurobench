from importlib import import_module

def _lazy_import(package_name, module_name, class_name):
    module = import_module(module_name, package=package_name)
    return getattr(module, class_name)
