cmake_minimum_required(VERSION 3.20)
project(TorquePowerTool_packager LANGUAGES NONE)

# You can change this name; the generated executable will use it
set(APP_NAME "TorquePowerTool")
set(ENTRY_SCRIPT "${CMAKE_SOURCE_DIR}/main.py")

# Find the system Python interpreter
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Allow users to toggle “single-file mode” and “no console window”
option(PYI_ONEFILE "Build as a single-file executable" ON)
option(PYI_WINDOWED "Hide console window for GUI apps" ON)

# Required third-party packages (tkinter ships with Python, usually no install needed)
set(PY_REQ
    pip
    setuptools
    wheel
    pyinstaller
    numpy
    scipy
    matplotlib
)

# PyInstaller arguments
set(PYI_ARGS --noconfirm --name ${APP_NAME})
if(PYI_ONEFILE)
  list(APPEND PYI_ARGS --onefile)
endif()
if(PYI_WINDOWED)
  # For GUI apps, remove console window on Windows/macOS
  list(APPEND PYI_ARGS --windowed)
endif()

# Some matplotlib/tk backends are handled by hooks;
# if you hit missing-module errors at runtime, add hidden imports here:
# list(APPEND PYI_ARGS --hidden-import matplotlib.backends.backend_tkagg)

# Work inside the build directory (avoid writing intermediates into source tree)
set(WORK_DIR "${CMAKE_BINARY_DIR}")

# Install dependencies and build with PyInstaller
add_custom_target(build_exe ALL
  COMMAND ${Python3_EXECUTABLE} -m pip install --upgrade ${PY_REQ}
  COMMAND ${Python3_EXECUTABLE} -m PyInstaller ${PYI_ARGS} "${ENTRY_SCRIPT}"
  WORKING_DIRECTORY "${WORK_DIR}"
  USES_TERMINAL
  COMMENT "Installing Python deps and building executable with PyInstaller..."
)

# Outputs are at: ${CMAKE_BINARY_DIR}/dist/<APP_NAME>[.exe]
# For convenience, copy it to a simple output folder
include(GNUInstallDirs)
if(WIN32)
  set(BUILT_EXE "${CMAKE_BINARY_DIR}/dist/${APP_NAME}.exe")
else()
  set(BUILT_EXE "${CMAKE_BINARY_DIR}/dist/${APP_NAME}")
endif()

add_custom_command(TARGET build_exe POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/out"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${BUILT_EXE}" "${CMAKE_BINARY_DIR}/out/"
  COMMENT "Copying built executable to ${CMAKE_BINARY_DIR}/out/"
)

install(PROGRAMS "${BUILT_EXE}" DESTINATION ${CMAKE_INSTALL_BINDIR})

# Convenience clean: remove PyInstaller artifacts
add_custom_target(clean_pyi
  COMMAND ${CMAKE_COMMAND} -E rm -rf
          "${CMAKE_BINARY_DIR}/build"
          "${CMAKE_BINARY_DIR}/dist"
          "${CMAKE_BINARY_DIR}/${APP_NAME}.spec"
          "${CMAKE_BINARY_DIR}/out"
  COMMENT "Removing PyInstaller artifacts..."
)
