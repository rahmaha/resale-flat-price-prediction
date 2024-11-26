<h2 align="center">HDB Resale Price Prediction</h2>

<!-- ABOUT THE PROJECT -->

## About The Project

This is project for midterm in MLZoomcamp 2024. The project is about predict resale flat prices in Singapore. Source of the dataset:

[data.gov.sg](https://beta.data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view)

The dataset contain features such as 'month', 'town', 'flat_type', 'block', 'street_name', 'storey_range' and many mores. Also the target would be 'resale_price' features. You can find details description of each features in the link above. Also, the dataset already downloaded and you can found it at `data/resale_flat_price_jan2017-nov2024.csv`.

The aim of this project is to predict the price of flat based on some conditions like which town, what kind of model for this flat and etc. This will be useful for customers who plan to buy flat, so they can prepare an estimate for their money.

- For EDA and model training (and selection), can be found in `Notebook.ipynb`,
- Training of the final model is in the `train.py` script.
- And the deployment for model is in the `predict.py`. The model deployed as web service using Flask in a Docker container on AWS Elastic Beanstalk.

## Data Preprocessing (feature engineer, eda, etc)

So, because it has column that show date or time. It need to handle carefully as they are related to trend and etc. Actually, I'm still lack behind on this but I try my best to handle this and explain to everyone who read this.

First of all, I have 'month' features from the original dataset. I made it as datetime for it types and split it to 'sale_month', 'sale_year', 'sale_date'. Why? because I want to understand their trend and learn their pattern. Although it still confusing. next, I made 'month_sin', 'month_cos' and 'month_since_start'. Why i made it? It's for **cycle encoding** I learned that computer will think that december (12) to january (1) is far because if i don't applied it they think it just normal number. So, to make model understand better, I applied cycle encoding. Next, one is 'month_since_start' it's for finding trend over time because the dataset from 2017 to 2024 November (although only two date on november). When I'm doing feature importance it turn out sale_year havehigh correlation with 'month_since_start'. So, the one that i included from date types are 'month_sin', 'month_cos', 'month_since_start'.

Another numerical columns I use for model is remaining_lease instead of lease_commence_date. I think it's more gives a clear representation of how much time is left for the lease. although, lease_commence_date have high correlationship with resale_price but their different is not too much wide. So, i think it's okay.

Next one, is categorical one. Actually, I think all of them are importants but I want to reduce the complexity of model and time. So, a few columns that has too many unique values I dropped it like 'block', 'street_name'.

The last one is dealing with the target 'resale_price'. It has long tail, I mean skewnees. So, I need to deal it with **log1p**.

## Model Training and Selection

I did try multiple models from linear regression, tress model (random forest, decision tree, xgboost until lightgbm) with trying different parameter.

The model that I choose is **LightGBM** with results are Validation RMSE: 0.07363337695202878
Test RMSE: 0.07342504700520996


## How to run this project

1. First, you can clone the repo using the following command:

```sh
   git clone https://github.com/rahmaha/mlzoomcamp-midterm.git
```

or click at the `code button` and chose download zip

2. To run the project, you need to have python, jupyter notebook and pipenv in your computer/laptop. This project also has a few of packages that needed that detailed in Pipfile/Pipfile.lock. To install all these packages, you can use pipenv to create separete environtment for this project. Make sure you already installed it or if not you can run this command:

```sh
   pip install pipenv
```

Then open terminal and go to the path of folder which contain this project (Pipfile/Pipfile.lock), and then run this command to install all packages needed:

```sh
   pipenv install
```

3. Now that you already installed it, you can start using this new virtual environtment for this project, run this following command to activate it:

```sh
   pipenv shell
```

4. Next is run the scripts like you normally do.

<!-- How to submit request to the web service-->

## How to submit request to the web service

This project only deployed on docker, because of that you need to have docker installed on your laptop. 
- first build docker image
`docker build -t hdb-price-predictor .`
![first step]('..\resale-flat-price-prediction\images\picture_1.png')

- second run docker image
`docker run -p 9696:9696 hdb-price-predictor`
![second step]('..\resale-flat-price-prediction\images\picture_2.png')

- third open new terminal, run pipenv shell on the same path
![third step]('..\resale-flat-price-prediction\images\picture_3.png')

- fourth type `python test.py`. test.py has the data that I want to test.
![fourth step]('..\resale-flat-price-prediction\images\picture_4.png')

