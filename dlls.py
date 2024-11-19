# +============================================================================+
# | Company:   SOFiSTiK AG                                                     |
# | Version:   SOFiSTIK 2024                                                   |
# +============================================================================+

import os               # for the environment variable necessary, this is a great tool
import platform         # checks the python platform
import string
from ctypes import *    # read the functions from the cdb

# # This example has been tested with Python 3.7 (64-bit)
# print ("The path variable=", os.environ["Path"])

# Check the python platform (32bit or 64bit)
# print ("Python architecture=", platform.architecture())
sofPlatform = str(platform.architecture())

# Get the DLLs (32bit or 64bit DLL)
if sofPlatform.find("32Bit") < 0:
    # # DEBUG: Using 64-bit DLLs
    # print("Hint: 64bit DLLs are used")

    # Add necessary DLL directories for 64-bit DLLs using os.add_dll_directory
    os.add_dll_directory(r"C:\Program Files\SOFiSTiK\2024\SOFiSTiK 2024\interfaces\64bit")
    os.add_dll_directory(r"C:\Program Files\SOFiSTiK\2024\SOFiSTiK 2024")

    # Load the DLL and define function references
    fullPathDll = r"C:\Program Files\SOFiSTiK\2024\SOFiSTiK 2024\interfaces\64bit\sof_cdb_w-2024.dll"
    myDLL = cdll.LoadLibrary(fullPathDll)
    py_sof_cdb_get = myDLL.sof_cdb_get
    py_sof_cdb_get.restype = c_int
    py_sof_cdb_kenq = myDLL.sof_cdb_kenq_ex
    py_sof_cdb_kexist = myDLL.sof_cdb_kexist
    
else:
     # Set environment variable for the DLL files
    print ("Hint: 32bit DLLs are used")
    path = os.environ["Path"]

    # 32bit DLLs
    dllPath = r"C:\sofistik_installation\trunk\SOFiSTiK trunk\interfaces\32bit"
    os.environ["Path"] = dllPath + ";" + path

    # Get the DLL functions
    myDLL = cdll.LoadLibrary("cdb_w31.dll")
    py_sof_cdb_get = cdll.LoadLibrary("cdb_w31.dll").sof_cdb_get
    py_sof_cdb_get.restype = c_int

    py_sof_cdb_kenq = cdll.LoadLibrary("cdb_w31.dll").sof_cdb_kenq_ex

if __name__ == "__main__":
    # Connect to CDB
    Index = c_int()
    cdbIndex = 99

    # input the cdb path here
    fileName = r"S:\test\test_file.cdb"

    # important: Unicode call!
    Index.value = myDLL.sof_cdb_init(fileName.encode('utf-8'), cdbIndex)

    # get the CDB status
    cdbStat = c_int()
    cdbStat.value = myDLL.sof_cdb_status(Index.value)

    # # Print the Status of the CDB
    # print ("CDB Status:", cdbStat.value)

    # Close the CDB, 0 - will close all the files
    myDLL.sof_cdb_close(0)

    # Print again the status of the CDB, if status = 0 -> CDB Closed successfully
    cdbStat.value = myDLL.sof_cdb_status(Index.value)
    # if cdbStat.value == 0:
    #     print ("CDB closed successfully, status = 0")
