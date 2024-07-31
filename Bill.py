class Bill:
    x = 0  # Text coordinate x
    y = 0  # Text coordinate y
    width = 0  # Text width
    height = 0  # Text height
    conf = 0  # Text confidence
    text = None  # Text Recognized
    identity = None  # Either header or body

    '''
    Initialize a Bill object with specific attributes.
    '''
    def __init__(self, x=0, y=0, width=0, height=0, conf=0, text=None, identity=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.conf = conf
        self.text = text
        self.identity = identity

    '''
    Assign the coordinate for each recognize text.
    @param df: A DataFrame containing information about recognized text from image.
    @return body: A list of Bill objects, each representing a recognized text with assigned coordinates.
    '''
    def assignCoordinate(self, df):
        body = []
        for index in df.index:
            self.x = df.loc[index, 'left']
            self.y = df.loc[index, 'top']
            self.width = df.loc[index, 'width']
            self.height = df.loc[index, 'height']
            self.conf = df.loc[index, 'conf']
            self.text = df.loc[index, 'text']
            bill = Bill(self.x, self.y, self.width, self.height, self.conf, self.text)
            body.append(bill)
        return body

    '''
    Allocate the content to respective cell.
    '''
    def allocateContent(self):
        pass

    def __str__(self):
        return f'x: {self.x}, y: {self.y}, width: {self.width}, height: {self.height}, conf: {self.conf}, ' \
               f'text: {self.text}, identity: {self.identity}'
