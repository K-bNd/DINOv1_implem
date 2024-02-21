class Config:
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])
