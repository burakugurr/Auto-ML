from tracemalloc import start
import pandas_datareader.data as web

class Data:
    def __init__(self, company,start,end):
        self.company = company
        self.start_date = start
        self.end_date = end

    def get_data(self):
        return web.DataReader(self.company,'yahoo' ,self.start_date, self.end_date)

    def tranform_data(self,data):
        data.reset_index(inplace=True)
        data.set_index('Date',inplace=True)
        
        return data



