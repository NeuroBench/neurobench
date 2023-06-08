class Dict2Class(object):
    def __init__(self, in_dict: dict):
        for key in in_dict:
            value = in_dict[key]

            if isinstance(value, dict):
                value = Dict2Class(value)
            elif isinstance(value, list):
                value = [Dict2Class(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, tuple):
                value = tuple([Dict2Class(item) if isinstance(item, dict) else item for item in value])
            
            setattr(self, key, value)

    def to_dict(self):
        out_dict = {key: getattr(self, key) for key in self.__dict__}

        for key in out_dict:
            if isinstance(out_dict[key], Dict2Class):
                out_dict[key] = out_dict[key].to_dict()

        return out_dict
    
    def __repr__(self):
        return str(self.to_dict())
    