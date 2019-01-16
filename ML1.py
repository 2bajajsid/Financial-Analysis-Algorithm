import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
import quandl
style.use("ggplot")
quandl.ApiConfig.api_key = "pBwWxMdmGSyuyiDeFctG"

data = quandl.get_table('WIKI/PRICES', ticker = ['AAPL', 'MSFT', 'WMT'], 
                        qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, 
                        date = { 'gte': '2015-12-31', 'lte': '2016-12-31' }, 
                        paginate=True)
data.head()

STATISTICS = ['D/E Ratio',
					'Trailing P/E',
                      'Price/Sales',
                      'Price/Book',
                      'Profit Margin',
                      'Operating Margin',
                      'Return on Assets',
                      'Return on Equity',
                      'Revenue Per Share',
                      'Market Cap',
                        'Enterprise Value',
                        'Forward P/E',
                        'PEG Ratio',
                        'Enterprise Value/Revenue',
                        'Enterprise Value/EBITDA',
                        'Revenue',
                        'Gross Profit',
                        'EBITDA',
                        'Net Income Avl to Common ',
                        'Diluted EPS',
                        'Earnings Growth',
                        'Revenue Growth',
                        'Total Cash',
                        'Total Cash Per Share',
                        'Total Debt',
                        'Current Ratio',
                        'Book Value Per Share',
                        'Cash Flow',
                        'Beta',
                        'Held by Insiders',
                        'Held by Institutions',
                        'Shares Short (as of',
                        'Short Ratio']

def Build_Data_Set(features = STATISTICS):

	data_df = pd.read_csv("key_stats.csv")
	data_df = data_df.reindex(np.random.permutation(data_df.index))

	X = np.array(data_df[features].values)#.tolist())
	y = (data_df["Status"]
		.replace("underperform", 0)
	    .replace("outperform", 1)
	    .values.tolist())

	X = preprocessing.scale(X) 


	return X,y

def Analysis():

	test_size = 50
	X, y = Build_Data_Set()

	print(X[0])

	print(len(X))

	clf = svm.SVC(kernel="linear", C=1.0)
	clf.fit(X[:-test_size],y[:-test_size])

	correct_count = 0; 

	predictions = clf.predict(X)
	
	for x in range(1, test_size + 1):
		if (predictions[-x] == y[-x]):
			correct_count += 1

	print("Accuracy:", (correct_count/test_size)*100)

def randomizing():
	df = pd.DataFrame({"D1": range(5), "D2": range(5)})
	df2 = df.reindex(np.random.permutation(df.index))
	print(df2)

randomizing()

Analysis()