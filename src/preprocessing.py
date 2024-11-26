import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self):
        pass

    def process_remaining_lease(self, lease):
        """Convert remaining_lease string of years and months to float."""
        try:
            if 'months' in lease:
                # Extract years and months
                years = float(lease.split('years')[0].strip())
                months = float(lease.split('years')[
                               1].split('months')[0].strip())
                return round(years + (months / 12), 2)
            else:
                # Only years
                years = float(lease.split('years')[0].strip())
                return years
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def process_month(self, df):
        if 'month' not in df.columns:
            raise KeyError("The 'month' column is missing.")
        else:
            # change month to sale date
            df['sale_date'] = pd.to_datetime(df['month'], format='%Y-%m')

            # separate and month
            df['sale_month'] = pd.to_datetime(df['month']).dt.month

            # drop the month column
            df.drop(columns=['month'], inplace=True)

            # cycle encoding
            df['month_sin'] = np.sin(2 * np.pi * df['sale_month']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['sale_month']/12)

            # create time trend
            df['month_since_start'] = (
                df['sale_date'] - df['sale_date'].min()).dt.total_seconds()/(60*60*24*30)

        return df

    def process_resale_price(self, df):
        df['resale_price_log'] = np.log1p(df['resale_price'])
        if df['resale_price'].le(0).any():
            raise ValueError("Some resale_price values are non-positive.")
        return df
    def optimize_data(self, df):
        categorical_columns = ['town', 'flat_type', 'flat_model', 'storey_range']
        numerical_columns = ['floor_area_sqm', 'remaining_lease','month_sin', 'month_cos', 'month_since_start', 'resale_price_log']
        df[categorical_columns] = df[categorical_columns].astype('category')
        for col in numerical_columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df

    def preprocess(self, df):
        """call of them in one"""

        df['remaining_lease'] = df['remaining_lease'].apply(
            self.process_remaining_lease)
        df = self.process_month(df)
        df = self.process_resale_price(df)
        df = self.optimize_data(df)
        return df
