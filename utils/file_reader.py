"""Module that contains functions to read the paper's data"""
from . import pd


def read_networks_csv(filename, along):
    """Function that read one or more network instances from a csv file

    Args:
        filename (str): The name of the file to be read
        along (str): Specify how instances are ordered on the file.
          The only accepted values are "rows", if different instances are on different rows,
          or "cols", if different instances are on different columns

    Returns:
        pd.Dataframe: a pandas dataframe containing network instances along rows and networks' arcs along columns
    """

    if along == "rows":
        networks_df = pd.read_csv(filename,
                                  decimal=",",
                                  header=[0, 1],
                                  index_col=0)

        # Convert column names from str to int and start counting nodes from 0 (easier)
        networks_df.columns = [networks_df.columns.get_level_values(0).astype(int) - 1,
                               networks_df.columns.get_level_values(1).astype(int) - 1]

        return networks_df

    elif along == "cols":
        networks_df = pd.read_csv(filename,
                                  index_col=[0, 1],
                                  decimal=",")

        # start conting nodes from 0 makes life easier
        networks_df.index = [networks_df.index.get_level_values(0) - 1,
                             networks_df.index.get_level_values(1) - 1]

        return networks_df.T  # transpose to have instances along rows
