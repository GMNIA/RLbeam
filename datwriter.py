import tempfile

class DatWriter:
    def __init__(self, y_values, z_values, h_values, b_values, output_filename="parametric_study.dat"):
        self.y_values = y_values
        self.z_values = z_values
        self.h_values = h_values
        self.b_values = b_values
        self.output_filename = output_filename
        self.temp_dir = tempfile.mkdtemp()  # Temporary directory for the output file

    def generate_text(self):
        # Define the content template with placeholders for Z, H, and B values
        content_template = """$SOFiSTiK Export Version 0.1.3
$ File for parametric study

$ Exported by AQUA Version 19.00-70
PROG AQUA urs:1
HEAD
UNIT 0
NORM 'BS' 'en199x-200x' COUN 44 SNOW '1' UNIT 0
CONC 1    C '25' TYPR    B TITL "=C 25/30 (EN 1992)"
CTRL
{SREC_BLOCKS}
CTRL
END

$ Exported by SOFIMSHA Version 24.01-70
PROG SOFIMSHA urs:2
HEAD
UNIT 0
SYST SPAC GDIV 10000 GDIR NEGZ
{NODE_BLOCKS}
{BEAM_BLOCKS}
END

$ Exported by SOFILOAD Version 17.20-70
PROG SOFILOAD urs:6
HEAD EXPORT FROM DATABASE
UNIT TYPE 0
LC 1 'NONE' FACD 1 TITL "Point load"
$ POIN NODE '{LOAD_NODE}' TYPE PG 10
END

+PROG ASE urs:7
HEAD Linear analysis
LC ALL
END
"""

        # Generate the SREC blocks (rectangular sections) with parameterized H and B
        srec_blocks = "\n".join(
            f"SREC {i+1} H {self.h_values[i]} B {self.b_values[i]} MNO 1 MRF 0 REF    C IT 100[o/o] AY 100[o/o] AZ 100[o/o] BCYZ '0' SPT 0 TITL \"B/H = {int(self.b_values[i]*1000)} / {int(self.h_values[i]*1000)} mm\""
            for i in range(len(self.h_values))
        )

        # Generate the NODE blocks with parameterized Z values
        lines = []
        max_y, max_z = max(self.y_values), max(self.y_values)
        for i, (y, z) in enumerate(zip(self.y_values, self.z_values), 1):
            # Define additional attributes for nodes (constraints)
            if i in [1, len(self.z_values)]:
                constraints = "FIX PPMY"
            else:
                constraints = ""

            # Create the line for each node coordinates in a 3d 10X10 m space
            # TODO generalise te 10X10m space into variable space
            lines.append(f"NODE {i} X 0.0 Y {10.0 * y / max_y} Z {10.0 * z / max_z} {constraints}")
        node_blocks = "\n".join(lines)

        # Generate the BEAM blocks with give nodes
        lines = []
        for i in range(len(self.z_values) - 1):
            lines.append(f"BEAM {1001 + i} {1 + i} {2 + i} NCS {1 + i}")
        beam_blocks = "\n".join(lines)

        # Format the template with the generated blocks
        content = content_template.format(
            SREC_BLOCKS=srec_blocks,
            NODE_BLOCKS=node_blocks,
            BEAM_BLOCKS=beam_blocks,
            LOAD_NODE = int(len(self.z_values) / 2) + 1,
        )
        return content
