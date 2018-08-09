cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(openni2_viewer)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(HDF5)
if(HDF5_FOUND)

  find_package(FLANN)
  include_directories(${FLANN_INCLUDE_DIRS})

  include_directories(${HDF5_INCLUDE_DIR} /usr/local/include /usr/include/hdf5/)

  add_executable (openni2_viewer openni2_viewer.cpp)
  target_link_libraries(openni2_viewer ${PCL_LIBRARIES} 
                                        ${Boost_LIBRARIES} ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY} 
                                        )
endif(HDF5_FOUND)
