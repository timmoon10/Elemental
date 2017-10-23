# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  CIRC_CIRC.hpp
  MC_MR.hpp
  MC_STAR.hpp
  MD_STAR.hpp
  MR_MC.hpp
  MR_STAR.hpp
  STAR_MC.hpp
  STAR_MD.hpp
  STAR_MR.hpp
  STAR_STAR.hpp
  STAR_VC.hpp
  STAR_VR.hpp
  VC_STAR.hpp
  VR_STAR.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
