# Generated by Django 3.1.5 on 2021-04-28 12:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('orders', '0002_auto_20210426_2305'),
    ]

    operations = [
        migrations.CreateModel(
            name='Border',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('border_name', models.CharField(max_length=15)),
                ('border_rate', models.IntegerField(default=0)),
            ],
        ),
    ]
