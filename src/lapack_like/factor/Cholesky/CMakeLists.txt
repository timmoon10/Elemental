# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  LowerMod.hpp
  LowerVariant2.hpp
  LowerVariant3.hpp
  PivotedLowerVariant3.hpp
  PivotedUpperVariant3.hpp
  ReverseLowerVariant3.hpp
  ReverseUpperVariant3.hpp
  SolveAfter.hpp
  UpperMod.hpp
  UpperVariant2.hpp
  UpperVariant3.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
