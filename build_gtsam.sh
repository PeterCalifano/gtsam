#!/usr/bin/env bash
# Build helper for GTSAM (Linux)
# - Aligned with build_lib.sh features (Aug 2025)
# - Generator-agnostic build via `cmake --build`

set -Eeuo pipefail
IFS=$'\n\t' # Narrows word splitting to newlines and tabs (safe with spaces)

# --- Defaults ---
buildpath="build"
install_path="/usr/local"

jobs="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}"
jobs=$(( jobs < 6 ? jobs : 6 ))

rebuild_only=false
build_type="relwithdebinfo"   # debug|release|relwithdebinfo|minsizerel
run_tests=true
CXX_FLAGS="-Wno-error=array-bounds"
python_wrap=false
matlab_wrap=false
unstable_build=false
use_expmap=true
use_tangent_preintegr=true
use_tbb=true
use_march_native=false
install=false
use_ninja=false
no_optim=false
clean_first=false
toolchain_file=""
cmake_defines=()
python_exe="${PYTHON_EXE:-$HOME/miniconda3/envs/gtsam/bin/python3.12}"

usage() {
  cat <<'USAGE'
Usage: build_gtsam.sh [OPTIONS]

Options:
  -B, --buildpath <dir>         Build directory (default: ./build)
      --install-path <dir>      Install prefix (default: /usr/local)
  -j, --jobs <N>                Parallel build jobs (default: $(nproc or 4))
  -r, --rebuild-only            Skip CMake configure; build existing tree only
  -t, --type|--type-build <t>   Build type: debug|release|relwithdebinfo|minsizerel
  -c, --checks                  Run tests (on by default). Alias of --run-tests
      --skip-tests              Do not run tests
  -f, --flagsCXX <flags>        Extra C/C++ flags (quoted). Appended to
                                default "-Wno-error=array-bounds", plus
                                warnings for Debug/RelWithDebInfo/Release
  -D, --define <var[=val]>      Extra CMake cache definitions (repeatable)
  -p, --python-wrap             Build Python wrapper
  -m, --matlab-wrap             Build MATLAB wrapper
  -u, --unstable-build          Build unstable components
  -e, --exp-map-disabled        Disable Expmap (Pose3/Rot3)
  -o, --on-manifold-preintegr   Use on-manifold preintegration instead of tangent
  -i, --install                 Run "install" target after tests
  -N, --ninja-build             Use Ninja generator (requires `ninja`)
  -n, --no-optim                Set -DNO_OPTIMIZATION=ON in the CMake cache
      --toolchain <file>        Pass CMake toolchain file (-DCMAKE_TOOLCHAIN_FILE=<file>)
      --clean                   Delete build dir before configuring
  -h, --help                    Show this help and exit

Environment:
  PYTHON_EXE  Override Python executable for wrappers
              (default: $HOME/miniconda3/envs/gtsam/bin/python3.12)

Examples:
  # Configure + build (RelWithDebInfo) into ./build
  ./build_gtsam.sh

  # Debug build with warnings, 8 jobs, and Ninja
  ./build_gtsam.sh -t debug -j 8 -N

  # Custom build dir and flags, run tests then install
  ./build_gtsam.sh -B out/release -t release -f "-march=native" -i
USAGE
}

die()  { echo -e "\e[31mError:\e[0m $*" >&2; echo; usage; exit 2; }
info() { echo -e "\e[34m[INFO]\e[0m $*"; }
trap 'echo -e "\e[31mBuild failed (line $LINENO).\e[0m"' ERR

bool_to_cmake() {
  if [[ "$1" == true ]]; then
    echo ON
  else
    echo OFF
  fi
}

# --- argument parsing (GNU getopt) ---
if ! command -v getopt > /dev/null 2>&1; then
  die "GNU getopt is required. On macOS: brew install gnu-getopt and adjust PATH."
fi

OPTIONS=B:j:rt:c:f:D:pmueoiNnh
LONGOPTIONS=buildpath:,build_path:,build-path:,install-path:,jobs:,rebuild-only,rebuild,type:,type-build:,checks,flagsCXX:,define:,python-wrap,matlab-wrap,unstable-build,unstable_build,exp-map-disabled,exp_map_disabled,on-manifold-preintegr,on_manifold_preintegr,help,ninja-build,no-optim,skip-tests,no-checks,clean,install,toolchain:
PARSED=$(getopt -o "$OPTIONS" -l "$LONGOPTIONS" -- "$@") || { usage; exit 2; }
eval set -- "$PARSED"

while true; do
  case "$1" in
    -B|--buildpath|--build_path|--build-path) buildpath="$2"; shift 2 ;;
        --install-path) install_path="$2"; shift 2 ;;
    -j|--jobs)          jobs="$2";     shift 2 ;;
    -r|--rebuild-only|--rebuild) rebuild_only=true; shift ;;
    -t|--type|--type-build) build_type="$2"; shift 2 ;;
    -c|--checks)        run_tests=true;  shift ;;
        --skip-tests|--no-checks) run_tests=false; shift ;;
    -f|--flagsCXX)      CXX_FLAGS="${CXX_FLAGS:+$CXX_FLAGS }$2"; shift 2 ;;
    -D|--define)        cmake_defines+=( "-D$2" ); shift 2 ;;
    -p|--python-wrap)   python_wrap=true; shift ;;
    -m|--matlab-wrap)   matlab_wrap=true; use_tbb=false; shift ;;
    -u|--unstable-build|--unstable_build) unstable_build=true; shift ;;
    -e|--exp-map-disabled|--exp_map_disabled) use_expmap=false; shift ;;
    -o|--on-manifold-preintegr|--on_manifold_preintegr) use_tangent_preintegr=false; shift ;;
    -i|--install)       install=true;    shift ;;
    -N|--ninja-build)   use_ninja=true;  shift ;;
    -n|--no-optim)      no_optim=true;   shift ;;
        --toolchain)    toolchain_file="$2"; shift 2 ;;
        --clean)        clean_first=true; shift ;;
    -h|--help)          usage; exit 0 ;;
    --) shift; break ;;
     *) die "Unknown option: $1" ;;
  esac
done

# --- normalize & validate build type ---
bt="${build_type,,}"
case "$bt" in
  debug)          cmake_bt="Debug" ;;
  release)        cmake_bt="Release" ;;
  relwithdebinfo) cmake_bt="RelWithDebInfo" ;;
  minsizerel)     cmake_bt="MinSizeRel" ;;
  *) die "Invalid build type: $build_type" ;;
esac

# For common types, enforce warnings unless user already provided them
if [[ "$bt" =~ ^(debug|relwithdebinfo|release)$ ]]; then
  CXX_FLAGS="${CXX_FLAGS:+$CXX_FLAGS }-Wall -Wextra -Wpedantic"
fi

# Enforce tests for Release
if [[ "$cmake_bt" == "Release" ]]; then
  run_tests=true
fi

# Validate toolchain file if provided
if [[ -n "$toolchain_file" && ! -f "$toolchain_file" ]]; then
  die "Toolchain file not found: $toolchain_file"
fi

# Pre-build checks
command -v cmake >/dev/null 2>&1 || die "cmake not found"
if [[ "$use_ninja" == true ]]; then
  command -v ninja >/dev/null 2>&1 || die "Requested Ninja but 'ninja' not found"
fi

if [[ "$rebuild_only" == true && ! -d "$buildpath" ]]; then
  die "No existing build directory at '$buildpath' for --rebuild-only"
fi

# Print info
info "Buildpath          : $buildpath"
info "Install prefix     : $install_path"
info "Jobs               : $jobs"
info "Build Type         : $cmake_bt"
info "Extra CXX flags    : ${CXX_FLAGS:-<none>}"
if [[ "$CXX_FLAGS" == *"-Wno-error=array-bounds"* ]]; then
  info "Notice             : -Warray-bounds is downgraded to warning (Eigen/AVX false-positive guard)."
fi
info "Use -march=native  : $use_march_native (via GTSAM_BUILD_WITH_MARCH_NATIVE)"
info "Extra CMake defines: ${cmake_defines[*]:-<none>}"
info "Python wrapper     : $python_wrap"
if [[ "$python_wrap" == true ]]; then
  info "Python executable  : $python_exe"
fi
info "MATLAB wrapper     : $matlab_wrap"
info "Unstable build     : $unstable_build"
info "Use Expmap         : $use_expmap"
info "Tangent preintegr  : $use_tangent_preintegr"
info "Use TBB            : $use_tbb"
info "Generator          : $([[ "$use_ninja" == true ]] && echo Ninja || echo 'Unix Makefiles')"
info "Toolchain file     : ${toolchain_file:-<none>}"
info "Run tests          : $run_tests"
info "Install after build: $install"

sleep 0.2

# --- Configure ---
if [[ "$rebuild_only" == false ]]; then
  if [[ "$clean_first" == true && -d "$buildpath" ]]; then
    info "Removing existing build dir '$buildpath'"
    rm -rf -- "$buildpath"
  fi

  cmake_args=(
    -S .
    -B "$buildpath"
    "-DCMAKE_BUILD_TYPE=$cmake_bt"
    "-DCMAKE_CXX_FLAGS=$CXX_FLAGS"
    "-DCMAKE_C_FLAGS=$CXX_FLAGS"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    "-DGTSAM_BUILD_UNSTABLE=$(bool_to_cmake "$unstable_build")"
    "-DGTSAM_BUILD_PYTHON=$(bool_to_cmake "$python_wrap")"
    "-DGTSAM_INSTALL_MATLAB_TOOLBOX=$(bool_to_cmake "$matlab_wrap")"
    "-DGTSAM_WITH_TBB=$(bool_to_cmake "$use_tbb")"
    -DGTSAM_WITH_EIGEN_MKL=OFF
    # Share the system Eigen with ROS, Pinocchio and downstream packages.
    -DGTSAM_USE_SYSTEM_EIGEN=ON
    "-DGTSAM_UNSTABLE_BUILD_PYTHON=$(bool_to_cmake "$python_wrap")"
    "-DGTSAM_TANGENT_PREINTEGRATION=$(bool_to_cmake "$use_tangent_preintegr")"
    "-DGTSAM_POSE3_EXPMAP=$(bool_to_cmake "$use_expmap")"
    "-DGTSAM_ROT3_EXPMAP=$(bool_to_cmake "$use_expmap")"
    "-DGTSAM_BUILD_WITH_MARCH_NATIVE=$(bool_to_cmake "$use_march_native")"
    "-DCMAKE_INSTALL_PREFIX=$install_path"
    '-DCMAKE_INSTALL_RPATH=$ORIGIN'
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE
  )
  [[ "$use_ninja"  == true ]] && cmake_args+=( -G Ninja )
  [[ "$no_optim"   == true ]] && cmake_args+=( -DNO_OPTIMIZATION=ON )
  [[ -n "$toolchain_file" ]] && cmake_args+=( "-DCMAKE_TOOLCHAIN_FILE=$toolchain_file" )
  [[ ${#cmake_defines[@]} -gt 0 ]] && cmake_args+=( "${cmake_defines[@]}" )

  if [[ "$python_wrap" == true ]]; then
    cmake_args+=( "-DPYTHON_EXECUTABLE=$python_exe" )
    if [[ ! -x "$python_exe" ]]; then
      echo "Warning: PYTHON_EXE does not point to an executable: $python_exe" >&2
    fi
  fi

  info "Configuring with CMake...\n"
  cmake "${cmake_args[@]}"
elif [[ -n "$toolchain_file" ]]; then
  info "Toolchain file provided, but --rebuild-only skips configure."
fi

# --- Build ---
info "\nBuilding..."
cmake --build "$buildpath" --parallel "$jobs"

# --- Test ---
if [[ "$run_tests" == true || "$install" == true ]]; then
  info "\nRunning tests..."
  ctest --test-dir "$buildpath" --output-on-failure -j "$jobs"
fi

# --- Install ---
if [[ "$install" == true ]]; then
  info "Installing..."
  cmake --build "$buildpath" --parallel "$jobs" --target install
fi

info "Done."
