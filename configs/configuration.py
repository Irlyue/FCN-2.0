import json


class BaseConfig:
    """
    The base configuration class. It supports attribute retrieval using the more natural '.' approach.
    Basic usage:
        config = BaseConfig({'learning_rate': 1e-3, 'optimizer': 'adam'}, name='train')
        >> config.keys()           # supports attribute from the dict class
        >> config.learning_rate    # easy way to get the attribute
        >> config.state            # readonly private state
        >> print(config)           # better string representation
    """
    def __init__(self, state, name='base'):
        """
        :param state: dict, containing the actual configuration items
        :param name: str, used in the string representation
        """
        self.__state = state.copy()
        self.name = name

    def __getattr__(self, key):
        if hasattr(self.state, key):
            return getattr(self.state, key)
        return self.__state[key]

    def __repr__(self):
        return '{}Config: {}'.format(self.name.capitalize(), json.dumps(self.state, indent=2))

    @property
    def state(self):
        return self.__state


class Config(BaseConfig):
    def __init__(self, state, name='simple'):
        super().__init__(state, name)


if __name__ == '__main__':
    a = Config({'learning_rate': 1e-3, 'optimizer': 'adam'})
    print(a)
    print(a.state)
    print(a.values())
    print(a.learning_rate)
