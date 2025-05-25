#!/bin/bash

echo "*****************************************"
echo "*                                       *"
echo "*        Running entrypoint.sh          *"
echo "*                                       *"
echo "*****************************************"


# ****************           SET VARS           *****************
# ***************************************************************
DB_HOST=db
SOCKFILE=/code/gunicorn_sock/gunicorn.sock
MIGRATE=False

WORKERS_DB=("citus_worker1" "citus_worker2") # Lista de nodos trabajadores
DB_PORT=5432
envfile='.env'

# Set = as delimiter
IFS='='
while read line; do
    #Read the pairs key/value into an array based on "=" delimiter
    read -a strarr <<< ${line}

    #Assign var values
    if [ "${strarr[0]}" = "IS_PRODUCTION" ]; then
        IS_PRODUCTION=${strarr[1]}
    fi
    if [ "${strarr[0]}" = "WORKERS" ]; then
        WORKERS=${strarr[1]}
    fi
    if [ "${strarr[0]}" = "WEB_PORT" ]; then
        WEB_PORT=${strarr[1]}
    fi
    # if [ "${strarr[0]}" = "DB_PORT" ]; then
    #     DB_PORT=${strarr[1]}
    # fi
    if [ "${strarr[0]}" = "POSTGRES_USER" ]; then
        POSTGRES_USER=${strarr[1]}
    fi
    if [ "${strarr[0]}" = "POSTGRES_DB" ]; then
        POSTGRES_DB=${strarr[1]}
    fi
    if [ "${strarr[0]}" = "POSTGRES_PASSWORD" ]; then
        POSTGRES_PASSWORD=${strarr[1]}
    fi
done < $envfile

if [ "${IS_PRODUCTION}" = "True" ]; then
    echo "Running in PRODUCTION mode"
    echo "Number of workers: ${WORKERS}"
else
    echo "Running in DEVELOPMENT mode"
fi

if [ "${MIGRATE}" = "True" ]; then
    echo "Migrations & fixtures: YES"
else
    echo "Migrations & fixtures: NO"
fi
echo ""

# ************          WAIT FOR DB TO START       **************
# ***************************************************************
echo "Waiting for psql to boot..."
while true; do
    PG_STATUS="$(pg_isready -h $DB_HOST -U $POSTGRES_USER)"
    PG_EXIT=$(echo $?)
    if [ "$PG_EXIT" = "0" ]; then
        echo "DB is ready!"
        break
    fi
    sleep 1
done

# ***********    ADD CITUS WORKERS TO COORDINATOR    ************
# ***************************************************************
echo "Adding Citus workers to the cluster if not already added..."
for worker_db in "${WORKERS_DB[@]}"; do
    echo "Checking worker: $worker_db"
    EXISTS=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h $DB_HOST -U $POSTGRES_USER -d $POSTGRES_DB -tAc "SELECT nodename FROM pg_dist_node WHERE nodename = '$worker_db'")
    if [ -z "$EXISTS" ]; then
        echo "Adding worker: $worker_db"
        PGPASSWORD="$POSTGRES_PASSWORD" psql -h $DB_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT master_add_node('$worker_db', $DB_PORT);"
    else
        echo "Worker $worker_db already exists in the cluster."
    fi
done

# **********        STARTING RABBITMQ & CELERY         **********
# ***************************************************************
#service rabbitmq-server start
nohup celery -A data_hub worker -l info --uid=celery &
#nohup celery -A data_hub beat -l info --uid=celery &

# **********          MIGRATIONS AND FIXTURES          **********
# ***************************************************************
if [ "${MIGRATE}" = "True" ]; then
    python3 manage.py makemigrations
    python3 manage.py migrate
    python3 manage.py loaddata infrastructure_fixture.json
    python3 manage.py loaddata sensor_fixture.json
fi



# *********         RUN DJANGO IN DEV/PROD MODE         *********
# ***************************************************************
if [ "${IS_PRODUCTION}" = "True" ]; then
    # Servidor de producción
    # ------------------------------------------
    python3 manage.py collectstatic --noinput &&
    gunicorn data_hub.asgi:application --timeout 600 --bind unix:$SOCKFILE -w $WORKERS -k uvicorn.workers.UvicornWorker 
    #gunicorn -k uvicorn.workers.UvicornWorker main:app
    # Options to run in TCP mode instead of socket mode
    #gunicorn oomur.wsgi:application --bind 0.0.0.0:$WEB_PORT
    #gunicorn oomur.asgi:application --bind 0.0.0.0:$WEB_PORT -w $WORKERS
else
    # Servidor de desarrollo
    # ----------------------------------------
    python3 manage.py runserver 0.0.0.0:8000
    #tail -f /dev/null

fi

