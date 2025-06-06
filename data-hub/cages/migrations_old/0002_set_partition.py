# Generated by Django 5.1.4 on 2025-01-13 13:41

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cages', '0001_initial'),
    ]

    operations = [
        migrations.RunSQL(
            """
            DROP TABLE IF EXISTS sensorload;
            """
        ),
        migrations.RunSQL(
            """
            CREATE TABLE sensorload (
                id BIGSERIAL,
                date_time TIMESTAMP NOT NULL,
                weight FLOAT,
                sensor_id BIGINT NOT NULL,
                CONSTRAINT fk_sensor FOREIGN KEY (sensor_id) REFERENCES cages_sensor(id)
            ) PARTITION BY RANGE (date_time);
            """,
            reverse_sql="DROP TABLE IF EXISTS sensorload;"
            #   PRIMARY KEY (sensor_id, date_time)
        ),
        migrations.RunSQL(
            """
            SELECT cron.schedule('create-sensorload-partitions', '0 0 1 * *', $$
              SELECT create_time_partitions(
                  table_name         := 'sensorload',
                  partition_interval := '1 month',
                  end_at             := now() + '12 months'
              )
            $$);
            """,
            reverse_sql="SELECT cron.unschedule('create-sensorload-partitions');"
        ),
    ]
