import MDAnalysis as mda 
from solvation_analysis.solute import Solute 
from solvation_analysis.tests import datafiles

u = mda.Universe(datafiles.ea_fec_pdb, datafiles.ea_fec_dcd)

# define solute AtomGroup
li_atoms = u.atoms.select_atoms("element Li")

# define solvent AtomGroups
EA = u.residues[0:235].atoms                    # ethyl acetate
FEC = u.residues[235:600].atoms                 # fluorinated ethylene carbonate
PF6 = u.atoms.select_atoms("byres element P")   # hexafluorophosphate

# instantiate solution
solute = Solute.from_atoms(li_atoms,
                    {'EA': EA, 'FEC': FEC, 'PF6': PF6},
                    radii={'PF6': 2.6, 'FEC': 2.7})


code = ["EA = u.residues[0:235].atoms","FEC = u.residues[235:600].atoms "]

for i in code:
    exec(i)