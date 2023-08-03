class Charming_Melody:

    def __init__(self, index, title, singer, degrees_and_pauses):
        self.index = index
        self.title = title
        self.singer = singer
        self.degrees_and_pauses = degrees_and_pauses

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.index,
                                         self.title,
                                         self.singer,
                                         self.degrees_and_pauses)

    def get_index(self):
        return self.index

    def get_title(self):
        return self.title

    def get_singer(self):
        return self.singer

    def get_degrees_and_pauses(self):
        return self.degrees_and_pauses
    def get_degrees_and_pauses(self):
        return self.degrees_and_pauses