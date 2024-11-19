from ctypes import c_int, byref, sizeof
from sofistik_daten_for_project import *
from dlls import myDLL


class SofistikCDBReader:
    def __init__(self, file_path):
        """
        Initialize the SofistikCDBReader with a file path and connect to the CDB.

        Parameters:
            file_path (str): Path to the .cdb file.
        """
        self.file_path = file_path
        self.index = c_int()
        self.cdb_index = 99
        self.connect_to_cdb()

    def connect_to_cdb(self):
        """Connect to the CDB and initialize the index."""
        self.index.value = myDLL.sof_cdb_init(self.file_path.encode('utf-8'), self.cdb_index)
        self.check_cdb_status()

    def check_cdb_status(self):
        """Check and print the status of the CDB connection."""
        cdb_stat = c_int()
        cdb_stat.value = myDLL.sof_cdb_status(self.index.value)
        
        # Log messages if necessary
        # if cdb_stat.value & 1:
        #     print("Database is active.")
        # if cdb_stat.value & 2:
        #     print("Database is connected to a file.")
        # if cdb_stat.value & 4:
        #     print("Database requires byte-swapping.")

    @staticmethod
    def interpret_cdb_get_status(return_value):
        """
        Interpret the return value of sof_cdb_get and return the corresponding status message.

        Parameters:
            return_value (int): The return value from sof_cdb_get.
        
        Returns:
            str: A message describing the return value.
        """
        if return_value == 0:
            return "No error: Record read successfully."
        elif return_value == 1:
            return "Error: Item is longer than allocated Data."
        elif return_value == 2:
            return "End of file reached: No more records to read."
        elif return_value == 3:
            return "Error: Key does not exist."
        else:
            return f"Unknown error code {return_value}: Check documentation for details."

    def get_data(self):
        """
        Retrieve input and output results from cdb (if existing)

        Returns:
            list: List of tuples, each containing x, y, z coordinates for each node.
        """

        # Initalise variables
        # Notes for retrieval of different data types: coordinates, results in displacements
        # Node coordinates  kwh, kwl, data_structure = 20, 0, CNODE [data entry, DONT KNOW, class sofistik_daten]
        # Node displacements kwh, kwl, data_structure = 24, 2, CN_DISP [data entry, load case, class sofistik_daten]
        # TODO create a dictionary for correct input from user
        kwh, kwl, data_structure = 24, 1, CN_DISP

        # Get record length and instance of the Class, and initialise all parameters
        rec_len = c_int(sizeof(data_structure))
        data_object = data_structure()
        ie = c_int(0)
        results = []

        # Collect data from calculated file
        # TODO hanle error if data entry exists
        while ie.value < 2:
            # Get data from cdb file
            ie.value = myDLL.sof_cdb_get(self.index, kwh, kwl, byref(data_object), byref(rec_len), 1)
            
            # Collect results or cdb INPUT data with dedicated methods (see sofistik_daten.py)
            if kwh == 24:
                results.append([data_object.m_nr, data_object.m_ux, data_object.m_uy, data_object.m_uz])
            elif kwh == 20:
                # Collect coordinates data from nodes
                results.append([v for v in data_object.m_xyz])

            # Reinitialize the record length
            rec_len = c_int(sizeof(data_structure))

        return results

    def close_cdb(self):
        """Close the CDB connection."""
        myDLL.sof_cdb_close(0)
        cdb_stat = c_int()
        cdb_stat.value = myDLL.sof_cdb_status(self.index.value)


if __name__ == "__main__":
    # Test for an existing cdb file
    file_path = r"C:\feom\path\to\file"
    cdb_reader = SofistikCDBReader(file_path)
    results = cdb_reader.get_data()
    min_z = min(results, key=lambda z: z[-1])
    print(min_z)
    cdb_reader.close_cdb()
