from . import pd


def read_networks_csv(filename, along):

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
