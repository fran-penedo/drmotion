#!/bin/bash

DREALVERSION=3.16.02
DREAL=dReal-${DREALVERSION}-linux-shared-libs
DREALURL=https://github.com/dreal/dreal3/releases/download/v${DREALVERSION}/${DREAL}.tar.gz

mkdir lib

wget -O - $DREALURL | tar -xz -C lib
mv -f lib/$DREAL lib/dReal
