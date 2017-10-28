class AttrReader():
    def __init__(self, obj):
        for k,v in obj.items():
            setattr(self, k, v)
