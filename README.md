# Stock-Price-Prediction
This project involves work on predicting Multiple Assets Simultaneously using Stock Market Indexes such as S&P500, Nasdaq and DJIA and the evaluation of respective performance of indexes.

## Source of Data
	1. WRDS (Intel & Amgen)
		Daily Data of Stock Prices
	2. Investing.Com
		Commodity
            Gold
            Crude Oil
	3. Index Historical Data
		S&P 500
		Nasdaq
		DJIA
    
## Inputs
  1. Daily data of Indexes (Open,Close, High , Low)
  2. Commodities Historical Data (Gold, Crude Oil)
  3. Indicators: SMA, MACD, Stochastic Oscillators, ROC, RSI, ADX

## Target
  Price Prediction of Amgen (AMGN) and Intel (INTC) stock

## Models used for comparison
  1. Linear Regression
  2. K-neighbors Regressor
  3. Support Vector Machine (SVM)
  4. Random Forest Regressor
  5. AdaBoost Regressor
  6. Gradient Boosting Regressor
  
  And comparison between CNN model and LSTM model.
  
## How to run
run 'Final_Project_CNNvsLSTM.ipynb' Neural networks comparison, and for other models run 'Final_Project_Model_Comparison.ipynb'

## Results

![image](https://github.com/ManthanKPatel/Stock-Price-Prediction/assets/90741568/2decbd55-497a-4fb8-a24e-a17261ebd4d4)
![image](https://github.com/ManthanKPatel/Stock-Price-Prediction/assets/90741568/27763b07-af90-471d-b635-dcb53153fd8e)

![image](https://github.com/ManthanKPatel/Stock-Price-Prediction/assets/90741568/3ac33cfb-437c-4bd4-9e4d-568ca803b668)

![image](https://github.com/ManthanKPatel/Stock-Price-Prediction/assets/90741568/196b234e-51e0-4296-9236-084f07aba8dd)

![image](https://github.com/ManthanKPatel/Stock-Price-Prediction/assets/90741568/9a2b6865-b0a7-4726-b968-b5d761342fc5)

## References
1. https://archive.ph/R3ZLM
2. https://wrds-www.wharton.upenn.edu/
3. https://www.ig.com/en/trading-strategies/10-trading-indicators-every-trader-should-know-190604
4. https://towardsdatascience.com/building-a-comprehensive-set-of-technical-indicators-in-python-for-quantitative-trading-8d98751b5fb
5. https://towardsdatascience.com/implementation-of-technical-indicators-into-a-machine-learning-framework-for-quantitative-trading-44a05be8e06
6. https://drive.google.com/file/d/1RSuUSw18lOAPlIUgtnbAAn3ij5IZhiBQ/view
7. https://www.investing.com/

