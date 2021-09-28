add_library(treebearddebug SHARED
  TreePrintRoutines.cpp    
)

install(TARGETS treebearddebug
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})