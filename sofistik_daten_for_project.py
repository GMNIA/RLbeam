#+============================================================================
#| Company:   SOFiSTiK AG 
#|      sofistik_daten.py
#|      automatically generated header, do not modify!
#+============================================================================


from ctypes import *


class CNODE(Structure):            # 20/00  Nodes
   _fields_ = [
         ('m_nr', c_int),          #        node-number
         ('m_inr', c_int),         #        internal node-number
         ('m_kfix', c_int),        #        degree of freedoms
         ('m_ncod', c_int),        #        additional bit code
         ('m_xyz', c_float * 3)    # [1001] X-Y-Z-ordinates
      ]
cnode = CNODE()

class CN_DISP(Structure):          # 24/LC:+  Displacements and support forces of nodes
   _fields_ = [
         ('m_nr', c_int),          #        node-number
         ('m_ux', c_float),        # [1003] displacement
         ('m_uy', c_float),        # [1003] displacement
         ('m_uz', c_float),        # [1003] displacement
         ('m_urx', c_float),       # [1004] rotation
         ('m_ury', c_float),       # [1004] rotation
         ('m_urz', c_float),       # [1004] rotation
         ('m_urb', c_float),       # [1005] twisting
         ('m_px', c_float),        # [1151] nodal support
         ('m_py', c_float),        # [1151] nodal support
         ('m_pz', c_float),        # [1151] nodal support
         ('m_mx', c_float),        # [1152] support moment
         ('m_my', c_float),        # [1152] support moment
         ('m_mz', c_float),        # [1152] support moment
         ('m_mb', c_float)         # [1105] warping moment
      ]
cn_disp = CN_DISP()


