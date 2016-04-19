#!/bin/bash
echo "export OPIE_DIR="$(pwd) >> $HOME/.bashrc
echo '''export PYTHONPATH=.:..:$PYTHONPATH/src''' >> $HOME/.bashrc
