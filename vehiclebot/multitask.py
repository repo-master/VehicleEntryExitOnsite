
import uuid
import typing

#TODO: Meta class that replaces all methods in derived classes with class method taking in instance id

class GlobalInstances:
    @staticmethod
    def init():
        global instances
        instances = {}
    @staticmethod
    def create_instance(obj = None) -> typing.Tuple[str, typing.Any]:
        i_id = str(uuid.uuid1())
        instances[i_id] = obj
        return i_id
    #TODO: Decorator
    @staticmethod
    def get_instance(instance : str):
        return instances.get(instance)


