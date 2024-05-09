import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
import os
import datetime

# Define a class for formatting
class DataFormatter(Formatter):
    def __init__ (self, dates, date_format='%Y-%m-%d'):
        self.dates = dates
        self.date_format = date_format

    # Extract the value at time t at position ‘position’
    def __call__(self, t, position=0):
        index = int(round(t))
        if index >= len(self.dates) or index < 0:
            return ''

        # My sample date seems to have the date in desired way
        # return dateObject.strftime(self.date_format)
        return self.date_format

def csv2rec(filename):
    return np.recfromtxt(filename, dtype=None, delimiter=',', names=True, encoding='utf-8')

if __name__=='__main__':
    # Load csv file into numpy record array
    data = csv2rec( os.path.dirname(os.path.abspath(__name__)) + '/data_visualization/data/aapl.csv')

    # Take a subset for plotting
    data = data[-70:]

    # Create the date formatter object
    formatter = DataFormatter(data.Date)

    # X axis
    x_vals = np.arange(len(data))
    # Y axis values are the closing stock quotes
    y_vals = data.Close

    # Plot data
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(x_vals, y_vals, 'o-')
    fig.autofmt_xdate()
    plt.show()

