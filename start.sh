#!/bin/sh
export MAXMROOT=${PWD}
forceENV=true

# Add library directory to environment
arch=`uname -s`

# Add python libraries to PYTHONPATH
if [[ "$arch" == "Darwin" ]]
then # for mac
    if [[ -d "$MAXMROOT/build/Debug" ]] && [[ ":$DYLD_LIBRARY_PATH:" != *":$MAXMROOT/build/Debug:"* ]]; then
       DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:+"$DYLD_LIBRARY_PATH:"}$MAXMROOT/build/Debug"
    fi
    if [[ -d "$MAXMROOT/build" ]] && [[ ":$DYLD_LIBRARY_PATH:" != *":$MAXMROOT/build:"* ]]; then
       DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:+"$DYLD_LIBRARY_PATH:"}$MAXMROOT/build"
    fi
else # for linux
    if [[ -d "$MAXMROOT/build" ]] && [[ ":$LD_LIBRARY_PATH:" != *":$MAXMROOT/build:"* ]]; then
     LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+"$LD_LIBRARY_PATH:"}$MAXMROOT/build"
    fi
fi
# Pythonpath
if [[ -d "$MAXMROOT/python" ]] && [[ ":$PYTHONPATH:" != *":$MAXMROOT/python:"* ]]; then
    PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$MAXMROOT/python"
fi

if [ ! -d "$MAXMROOT/maxmEnv" ] || [[ "$forceENV" = true ]];
then
    python3 -m ensurepip
    python3 -m venv maxmEnv
    source maxmEnv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e python 

    # Get system architecture
    #arch=$(root-config --arch)

    # create 'build' directory if not existing
    if [ ! -d "build" ]; then
    mkdir "build"
    fi
    # go into build directory
    cd "build"

    # compilation
    cmake ..
    make
    # get back to project directory
    cd ..
else
    source maxmEnv/bin/activate
fi

echo All done!
