#!/bin/bash

TIKA_JAR_URL="http://search.maven.org/remotecontent?filepath=org/apache/tika/tika-server/1.19/tika-server-1.19.jar"
TIKA_WORKSPACE=$HOME/tika
TIKA_FILE_NAME="tika_server.jar"

if [ ! -f $TIKA_WORKSPACE/$TIKA_FILE_NAME ]; then
    echo -e "Downloading tika-server.jar"

    if [ ! -d "$TIKA_WORKSPACE" ]; then
        echo -e "making tika workspace"
        mkdir $TIKA_WORKSPACE
    fi

    wget -c $TIKA_JAR_URL -O $TIKA_WORKSPACE/$TIKA_FILE_NAME
fi
