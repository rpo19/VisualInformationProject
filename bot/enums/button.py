from enum import Enum

class Button(str, Enum):
    
    def __str__(self):
        return str(self.value)

    BTN_SEARCH_SIMILAR = 'Retrieve similar products'
    BTN_STYLE_TRANSFER = 'Apply style to a product'
    BTN_APPLY_FILTER = 'Apply filter to image'
    BTN_FILTER_MOSAIC = 'Mosaic'
    BTN_FILTER_HANDDRAW = 'Hand drawing'
    BTN_SIMIL_COLOR = 'by Color'
    BTN_SIMIL_SHAPE = 'by Shape'
    BTN_SIMIL_NEURAL_EFFICIENT = 'by EfficientNet'
    BTN_SIMIL_NEURAL_RESNET = 'by ResNet50'
    BTN_GENERATE_GIF = 'Generate gif'
    BTN_STOP = '/stop'
    BTN_START = '/start'
    BTN_YES = 'Yes'
    BTN_NO = 'No'