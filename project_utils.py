import os
import measured
import pandas as pd
import s3fs
import concurrent.futures
from IPython.display import display, HTML
import time


class Project:
    def __init__(self,
                 local_dir='/home/joshua_wassink/repos/joshua_wassink/',
                 s3_dir='s3://airbnb-emr/teams/martech-ds/joshua-wassink/',
                 project_name=None):
        self.s3 = s3fs.S3FileSystem(anon=False)

        self.project_name = project_name
        self.local_dir = local_dir
        self.s3_dir = s3_dir
        self.data = measured.Data()
        
        pd.reset_option('display.float_format')
        pd.options.display.float_format = '{:.10f}'.format
        
        for category in ['results', 'plots', 'sql', 's3']:
            self.set_directory(category=category)

    
    def load_data(
            self,
            file_name: str,
            file_name_ext: str = '',
            config: dict ={},
            from_cache: bool = True,
            verbose: bool = False):
        """
        Loads a SQL query from a file, executes it, and returns the result as a DataFrame.
        Can run asynchronously to allow continued work in notebooks.

        :param file_name: Name of the SQL file containing the query.
        :param config: Dictionary of parameters to replace in the SQL query.
        :param from_cache: If True, tries to load from cache before executing query.
        :param verbose: If True, prints the SQL query before execution.
        :param async_load: If True, loads data asynchronously and returns a future.
        :return: Either a DataFrame or a Future that will resolve to a DataFrame.
        """
        file_path = f"{self.s3_path}{file_name}{file_name_ext}.parquet"
        query_path = f"{self.sql_path}{file_name}.sql"

        if from_cache and self.s3.exists(file_path):
            print(f"Loading data for query {query_path} from S3")
            return pd.read_parquet(file_path)

        # Define the data loading function
        def _load_data_task():
            start_time = time.time()
            # Read the SQL file
            query = self.read_sql(query_path, config, verbose)
            
            # Execute the SQL query
            print(f"Loading data for query {query_path} from warehouse")
            df = self.data.fetch_df(query)
            df.to_parquet(file_path, index=False)
            
            elapsed_time = time.time() - start_time
            print(f"✅ Data loading complete for {file_name} in {elapsed_time:.2f} seconds")
            print(f"Saved {file_path} to S3")
                            
            return df

        # Run synchronously
        return _load_data_task()

    def read_sql(self, query_path, config={}, verbose: bool = False):
        """
        Reads a SQL query from a file and returns it.

        :param file_path: Path to the SQL file containing the query.
        :return: A string containing the query.
        """
        # Read the SQL file
        with open(query_path, 'r') as file:
            query = file.read()
            for key in config:
                query = query.replace(key, config[key])
        if verbose:
            print(query)
        return query


    def set_directory(self, category: str):
        """
        Creates a directory if it doesn't exist and sets a class attribute to the path.
        
        :param dirpath: Base directory path
        :param dirname: Name of the directory to create
        :return: Full path to the directory (dirpath + dirname)
        """
        # Ensure dirpath ends with a slash
        if category == 's3':
            dirpath = f"{self.s3_dir}{self.project_name}/"
        else:
            dirpath = f"{self.local_dir}{category}/{self.project_name}/"
            if not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
                                
        setattr(self.__class__, f"{category}_path", dirpath)

    def save_output(self, output, file_name: str):
        """
        Saves a DataFrame to a CSV file in the results directory.

        :param df: DataFrame to save.
        :param file_name: Name of the CSV file (without extension).
        """
        if type(output) == pd.DataFrame:
            out_path = f"{self.results_path}{file_name}.csv"
            output.to_csv(out_path, index=False)
        else:
            out_path = f"{self.plots_path}{file_name}.png"
            output.savefig(out_path, bbox_inches='tight')
        print(f"Saved {file_name} to {out_path}")