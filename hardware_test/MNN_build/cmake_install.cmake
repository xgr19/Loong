# Install script for directory: /home/fhb/MNN/MNN-master

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/MNN" TYPE FILE FILES
    "/home/fhb/MNN/MNN-master/include/MNN/MNNDefine.h"
    "/home/fhb/MNN/MNN-master/include/MNN/Interpreter.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/HalideRuntime.h"
    "/home/fhb/MNN/MNN-master/include/MNN/Tensor.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/ErrorCode.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/ImageProcess.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/Matrix.h"
    "/home/fhb/MNN/MNN-master/include/MNN/Rect.h"
    "/home/fhb/MNN/MNN-master/include/MNN/MNNForwardType.h"
    "/home/fhb/MNN/MNN-master/include/MNN/AutoTime.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/MNNSharedContext.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/MNN/expr" TYPE FILE FILES
    "/home/fhb/MNN/MNN-master/include/MNN/expr/Expr.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/expr/ExprCreator.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/expr/MathOp.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/expr/NeuralNetWorkOp.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/expr/Optimizer.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/expr/Executor.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/expr/Module.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/expr/NeuralNetWorkOp.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/expr/ExecutorScope.hpp"
    "/home/fhb/MNN/MNN-master/include/MNN/expr/Scope.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/fhb/MNN/MNN-master/build/libMNN.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/fhb/MNN/MNN-master/build/3rd_party/protobuf/cmake/cmake_install.cmake")
  include("/home/fhb/MNN/MNN-master/build/express/cmake_install.cmake")
  include("/home/fhb/MNN/MNN-master/build/tools/train/cmake_install.cmake")
  include("/home/fhb/MNN/MNN-master/build/tools/converter/cmake_install.cmake")
  include("/home/fhb/MNN/MNN-master/build/tools/cv/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/fhb/MNN/MNN-master/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
