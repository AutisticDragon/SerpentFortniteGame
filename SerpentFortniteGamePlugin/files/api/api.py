from serpent.game_api import GameAPI



class FortniteAPI(GameAPI):
    # A GameAPI is intended to contain functions and pieces of data that are applicable to the 
    # game and not agent or environment specific (e.g. Game inputs, Frame processing)

    def __init__(self, game=None):
        super().__init__(game=game)

    def my_api_function(self):
        pass

    class MyAPINamespace:

        @classmethod
        def my_namespaced_api_function(cls):
            api = FortniteAPI.instance
