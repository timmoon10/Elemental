# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  AllGather.hpp
  ColAllGather.hpp
  ColAllToAllDemote.hpp
  ColAllToAllPromote.hpp
  ColFilter.hpp
  Exchange.hpp
  Filter.hpp
  Gather.hpp
  GeneralPurpose.hpp
  PartialColAllGather.hpp
  PartialColFilter.hpp
  PartialRowAllGather.hpp
  PartialRowFilter.hpp
  RowAllGather.hpp
  RowAllToAllDemote.hpp
  RowAllToAllPromote.hpp
  RowFilter.hpp
  Scatter.hpp
  Translate.hpp
  TranslateBetweenGrids.hpp
  TransposeDist.hpp
  internal_decl.hpp
  internal_impl.hpp
  util.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
