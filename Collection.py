
import pandas_datareader.data as web

"""
    Data class for getting data from the yahoo finance server
    params:
        company: company name
        start: Scraping start date
        end: Scraping end date
"""
class Data:
    def __init__(self, company,start,end):
        self.company = company
        self.start_date = start
        self.end_date = end

    # Get data from yahoo finance and return a dataframe
    def get_data(self):
        return web.DataReader(self.company,'yahoo' ,self.start_date, self.end_date)
        
    #  Return a dataframe with reshape data
    def tranform_data(self,data):
        data.reset_index(inplace=True)
        data.set_index('Date',inplace=True)
        return data



