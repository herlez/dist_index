# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/herlez/uni/ma/dist_index

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/herlez/uni/ma/dist_index/build

# Include any dependencies generated for this target.
include CMakeFiles/tdc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tdc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tdc.dir/flags.make

CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.o: CMakeFiles/tdc.dir/flags.make
CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.o: ../extlib/tdc/src/vec/int_vector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/herlez/uni/ma/dist_index/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.o -c /home/herlez/uni/ma/dist_index/extlib/tdc/src/vec/int_vector.cpp

CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/herlez/uni/ma/dist_index/extlib/tdc/src/vec/int_vector.cpp > CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.i

CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/herlez/uni/ma/dist_index/extlib/tdc/src/vec/int_vector.cpp -o CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.s

CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.o: CMakeFiles/tdc.dir/flags.make
CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.o: ../extlib/tdc/src/vec/allocate.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/herlez/uni/ma/dist_index/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.o -c /home/herlez/uni/ma/dist_index/extlib/tdc/src/vec/allocate.cpp

CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/herlez/uni/ma/dist_index/extlib/tdc/src/vec/allocate.cpp > CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.i

CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/herlez/uni/ma/dist_index/extlib/tdc/src/vec/allocate.cpp -o CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.s

# Object files for target tdc
tdc_OBJECTS = \
"CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.o" \
"CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.o"

# External object files for target tdc
tdc_EXTERNAL_OBJECTS =

libtdc.a: CMakeFiles/tdc.dir/extlib/tdc/src/vec/int_vector.cpp.o
libtdc.a: CMakeFiles/tdc.dir/extlib/tdc/src/vec/allocate.cpp.o
libtdc.a: CMakeFiles/tdc.dir/build.make
libtdc.a: CMakeFiles/tdc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/herlez/uni/ma/dist_index/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libtdc.a"
	$(CMAKE_COMMAND) -P CMakeFiles/tdc.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tdc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tdc.dir/build: libtdc.a

.PHONY : CMakeFiles/tdc.dir/build

CMakeFiles/tdc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tdc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tdc.dir/clean

CMakeFiles/tdc.dir/depend:
	cd /home/herlez/uni/ma/dist_index/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/herlez/uni/ma/dist_index /home/herlez/uni/ma/dist_index /home/herlez/uni/ma/dist_index/build /home/herlez/uni/ma/dist_index/build /home/herlez/uni/ma/dist_index/build/CMakeFiles/tdc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tdc.dir/depend

