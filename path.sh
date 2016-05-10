#!/bin/bash
echo "export OPIE_DIR=$(pwd)" >> $HOME/.bashrc
echo '''export PYTHONPATH=$OPIE_DIR/src:$PYTHONPATH''' >> $HOME/.bashrc
echo '''export PYTHONPATH=$OPIE_DIR/src/my_package/relation:$PYTHONPATH''' >> $HOME/.bashrc
echo '''export PYTHONPATH=$OPIE_DIR/src/my_package/complex:$PYTHONPATH''' >> $HOME/.bashrc
echo '''export PYTHONPATH=$OPIE_DIR/src/my_package/utils:$PYTHONPATH''' >> $HOME/.bashrc
