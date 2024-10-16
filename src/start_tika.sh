#!/bin/bash

TIKA_PORT=9998
TIKA_HOST=localhost
TIKA_WORKSPACE=$HOME/tika
TIKA_FILE_NAME="tika_server.jar"

echo -e "## Setting environment vars"

export TIKA_SERVER_ENDPOINT="http://$TIKA_HOST:$TIKA_PORT"
echo -e "TIKA_SERVER_ENDPOINT to $TIKA_SERVER_ENDPOINT"

export TIKA_CLIENT_ONLY=True
echo -e "TIKA_CLIENT_ONLY to $TIKA_CLIENT_ONLY"

echo -e "## Starting tika server on: $TIKA_WORKSPACE"
cd $TIKA_WORKSPACE

java -jar $TIKA_FILE_NAME -h $TIKA_HOST &
