import subprocess
import time
import os
from datwriter import DatWriter
from tempfile import TemporaryDirectory
import shutil
import random
from readresults import SofistikCDBReader

class RunDat:
    def __init__(self, trajectory):
        self.z_values = [z for _, z in trajectory]
        self.y_values = [y for y, _ in trajectory]
        self.sps_path = r'C:/Program Files/SOFiSTiK/2024/SOFiSTiK 2024/sps.exe'
        self.temp_dir = os.path.join(os.getcwd(), 'temp')
        self.displacements = None

    def generate_dat(self):
        n_elements = len(self.z_values)
        h_values = [0.1 for _ in range(n_elements)]
        b_values = [0.1 for _ in range(n_elements)]

        # Instantiate the class and generate the .dat file text
        dat_writer = DatWriter(self.y_values, self.z_values, h_values, b_values)
        return dat_writer.generate_text()

    def _run_test(self):
        # Use a temporary directory to save the .dat file
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        file_path = os.path.join(self.temp_dir, 'linebridge.dat')

        # Generate and write the .dat file content
        dat_text = self.generate_dat()
        with open(file_path, "w") as file:
            file.write(dat_text)
        print(f"File saved to temporary path: {file_path}")

        # Run the SOFiSTiK SPS executable in the temporary directory
        start = time.time()
        subprocess.run([self.sps_path, file_path], cwd=self.temp_dir)
        print('Total time:', time.time() - start)


    def run(self, dat_path_to_save=''):
        # Use TemporaryDirectory as a context manager
        with TemporaryDirectory() as temp_dir:
            dat_temp_path = os.path.join(temp_dir, 'linebridge.dat')

            # Generate and write the .dat file content abd run the SOFiSTiK SPS executable
            dat_text = self.generate_dat()
            with open(dat_temp_path, "w") as file:
                file.write(dat_text)
            subprocess.run([self.sps_path, dat_temp_path], cwd=temp_dir)

            # Save file if string with file path is specified
            if dat_path_to_save:
                shutil.copy(dat_temp_path, dat_path_to_save)

            # Update results after simulation
            cdb_reader = SofistikCDBReader(dat_temp_path.replace('.dat', '.cdb'))
            self.displacements = cdb_reader.get_data()
            cdb_reader.close_cdb()
            

if __name__ == "__main__":
    # Example Z and Y values
    n_elements = 16
    z_values = [0] + [random.randint(0, n_elements) for _ in range(n_elements - 2)] + [0]
    y_values = [0] + sorted(random.randint(0, n_elements) for _ in range(n_elements - 2)) + [n_elements]
    trajectory = [(y, z) for y, z in zip(y_values, z_values)]

    # Create an instance of the class and run it
    run_dat = RunDat(trajectory)
    run_dat._run_test()
    run_dat.run(dat_path_to_save=os.path.join(os.getcwd(), 'temp.dat'))
