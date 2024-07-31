# Heuristic rules process tabular data in a document


from Bill import Bill


class TabularRule:
    identity = None
    head = None  # Header condition
    data = []
    row_list = []
    col_range = []
    final_list = []

    def __init__(self, data, first):
        self.data = data
        self.head = first
        self.row_list = []
        self.final_list = [None] * 4

    '''
    Execution function
    '''
    def runner(self):
        self.identity = 'Head' if self.head else 'Body'
        self.tableRules()
        self.headerRules() if self.head else self.contentRules()

    '''
    Assemble raw text into tabular format based on defined rules.
    '''
    def tableRules(self):
        rule4 = False
        temp_x1 = None
        temp_x2 = None
        content = ''

        for index, row in enumerate(self.data):
            rule6 = False
            x1 = row.x
            w1 = row.width
            text = row.text
            previous_row = self.data[index - 1]
            x2 = previous_row.x
            w2 = previous_row.width
            distance = x1 - (x2 + w2)

            if TabularRule.rule1(self.data):
                print('Comply Rule 1')
                temp_x1 = x1
                temp_x2 = x1 + w1

                grouped_bill = Bill(x=temp_x1, width=temp_x2-temp_x1, text=text, identity=self.identity)
                self.row_list = [grouped_bill]

            elif TabularRule.rule2(index):
                print('Comply Rule 2')
                content = text
                temp_x1 = x1

            elif TabularRule.rule3(distance, index, self.data):
                print('Comply Rule 3')
                temp_x1 = temp_x1 if temp_x1 else x2
                temp_x2 = x1 + w1

                grouped_bill = Bill(x=temp_x1, width=temp_x2-temp_x1, text=content, identity=self.identity)

                self.row_list.append(grouped_bill)

            elif TabularRule.rule4(distance):
                print('Comply Rule 4')
                # print(distance)
                content += ' ' + text
                temp_x1 = x2 if not rule4 else temp_x1
                rule4 = True

            elif TabularRule.rule5(distance, index, self.data):
                print('Comply Rule 5')
                temp_x1 = x2
                temp_x2 = x2 + w2

                grouped_bill = Bill(x=temp_x1, width=temp_x2-temp_x1, text=content, identity=self.identity)

                self.row_list.append(grouped_bill)

                content = text
                temp_x1 = x1
                temp_x2 = x1 + w1

                grouped_bill = Bill(x=temp_x1, width=temp_x2-temp_x1, text=content, identity=self.identity)

                self.row_list.append(grouped_bill)

            elif TabularRule.rule6(distance):
                print('Comply Rule 6')
                rule4 = False
                rule6 = True
                temp_x1 = x2 if not temp_x1 else temp_x1
                temp_x2 = x2 + w2

                grouped_bill = Bill(x=temp_x1, width=temp_x2-temp_x1, text=content, identity=self.identity)

                self.row_list.append(grouped_bill)

                content = text

            temp_x1, temp_x2 = (0, 0) if rule6 else (temp_x1, temp_x2)

    '''
    Apply rules specific to the header row.
    '''
    def headerRules(self):
        midpoint = 0
        self.col_range.clear()

        if self.row_list is None:
            return

        for index, ele in enumerate(self.row_list):
            x1 = ele.x
            w1 = ele.width
            x2 = self.row_list[index + 1].x if not index == len(self.row_list) - 1 else 0

            if index == 0:  # First element of the list
                continue
            elif index == 1:
                midpoint = (x2 - (x1 + w1)) / 2
                ele.x = x1 - midpoint
                xw1 = ((x1 + w1) + midpoint)
                ele.width = xw1 - ele.x
                self.row_list[index - 1].width = ele.x
            elif index == len(self.row_list) - 1:  # Last element of the list
                ele.x = x1 - midpoint
            else:
                ele.x = x1 - midpoint
                midpoint = (x2 - (x1 + w1)) / 2
                xw1 = (x1 + w1) + midpoint
                ele.width = xw1 - ele.x

        self.col_range.extend((ele.x, ele.x + ele.width) for ele in self.row_list)

    '''
    Apply rules specific to content rows.
    '''
    def contentRules(self):
        self.final_list = [
            val.text
            for val in self.row_list
            for x1, x2 in self.col_range
            if x1 < val.x < x2 and x1 < val.x + val.width < x2
        ]

    '''
    Rule 1: Check if the row only has 1 element.
    @param data: A list of data rows.
    @return True if rule is applied, False otherwise.
    '''
    @staticmethod
    def rule1(data):
        return len(data) == 1

    '''
    Rule 2: Check if the row is the first element.
    @param counter: An integer representing the row index.
    @return True if rule is applied, False otherwise.
    '''
    @staticmethod
    def rule2(counter):
        return counter == 0

    '''
    Rule 3: Check if the distance is lower than 40 and the row is last element. 
    @param dist: A integer representing the distance.
    @param counter: An integer representing the row index.
    @param data: A list of data rows.
    @return True if rule is applied, False otherwise.
    '''
    @staticmethod
    def rule3(dist, counter, data):
        return dist <= 40 and counter == len(data) - 1

    '''
    Rule 4: Check if the distance is lower than 40.
    @param dist: A integer representing the distance.
    @return True if rule is applied, False otherwise.
    '''
    @staticmethod
    def rule4(dist):
        return dist <= 40

    '''
    Rule 5: Check if the distance is higher or equal to 40 and the row is last element.
    @param dist: A integer representing the distance.
    @param counter: An integer representing the row index.
    @param data: A list of data rows.
    @return True if rule is applied, False otherwise.
    '''
    @staticmethod
    def rule5(dist, counter, data):
        return dist > 40 and counter == len(data) - 1

    '''
    Rule 6: Check if the distance is higher or equal to 40.
    @param dist: A integer representing the distance.
    @return True if rule is applied, False otherwise.
    '''
    @staticmethod
    def rule6(dist):
        return dist > 40
