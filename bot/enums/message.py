from enum import Enum

class Message(str, Enum):
    def __str__(self):
        return str(self.value)
        
    MSG_START = 'Hi, send /start to begin!'
    MSG_PROMPT_ACTION = 'What do you want to do?'
    MSG_SEND_FOR_SIMILAR = 'Send an image of a clothing'
    MSG_SEND_FOR_STYLE_BASE = 'Send an image of a clothing to which you want to apply a new style (a uniform background is preferrable)'
    MSG_SEND_FOR_STYLE_STYLE = 'Send a style texture'
    MSG_CHOOSE_SIMILARITY = 'Choose which kind of similarity to look for'
    MSG_QUALITY_CHECK_FAILED = "The image sent didn't pass the quality check due to {}. Continue anyway?"
    MSG_SEND_FOR_APPLY_FILTER_BASE = 'Which kind of filter you want to apply?'
    MSG_SEND_FOR_APPLY_FILTER = 'Send an image to apply the selected filter'
    MSG_FILTER_DONE = 'Hope you like it!'
    MSG_STYLE_TRANSFER_DONE = 'Hope you like the new style!'
    MSG_RETRIEVAL_NEW_STYLE_DONE = "Here some similar clothes to the one generated!"
    MSG_RETRIEVAL__DONE = "Here some similar clothes"
    MSG_UNKNOWN = "I cannot recognize the image, sorry!"
    MSG_SEND_FOR_GIF = 'Send an image to generate a gif'
    MSG_GIF_DONE = 'Enjoy your gif!'