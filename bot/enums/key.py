from enum import Enum

class Key(str, Enum):
    
    def __str__(self):
        return str(self.value)

    KEY_SIMILARITY = 'KEY_SIMILARITY_{}'
    KEY_STYLE_BASE_IMG = 'KEY_STYLE_BASE_{}'
    KEY_FILTER = 'KEY_FILTER_{}'
    # only needed when quality check failed if users choose to continue
    KEY_SIMILAR_IMAGE_IMG = 'KEY_SIMILAR_IMAGE_{}'
    KEY_QUALITY_CONTINUE_PREVSTATE = 'KEY_QUALITY_CONTINUE_PREVSTATE_{}'