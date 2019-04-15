from serpent.game import Game

from serpent.input_controller import InputControllers

from .api.api import FortniteAPI

from serpent.utilities import Singleton

class SerpentFortniteGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"

        kwargs["input_controller"] = InputControllers.NATIVE_WIN32  # <= Specify your InputController backend here

        kwargs["window_name"] = "Fortnite  " # Do Not Change

        kwargs["executable_path"] = "Your Fortnite EXE File"

        super().__init__(**kwargs)

        self.api_class = FortniteAPI
        self.api_instance = None

        self.environments = dict()
        self.environment_data = dict()

        self.frame_transformation_pipeline_string = "RESIZE:640x480|GRAYSCALE"

        self.frame_width = 640
        self.frame_height = 480
        self.frame_channels = 0
    @property
    def screen_regions(self):
        regions = {
            "SCORE_AREA": (416, 252, 428, 266),
            "HP_AREA": (428, 251, 441, 271)
        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets
