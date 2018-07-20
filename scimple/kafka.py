import inspect
import math
import os
import time
import random
import re
from collections import Collection, Iterable
from shutil import copyfile, copytree
from subprocess import Popen, PIPE, TimeoutExpired
import logging

from plot import _get_scimple_data_path, sc_sqlc
# #####
# KAFKA
# #####


def _export_pyspark_kafka_jar():
    global _sc, _spark
    if 'SPARK_HOME' in os.environ:
        if not os.path.isfile(
                os.path.join(os.environ['SPARK_HOME'],'jars/spark-streaming-kafka-0-8-assembly_2.11-2.3.0.jar')):
            copyfile(_get_scimple_data_path('spark-streaming-kafka-0-8-assembly_2.11-2.3.0.jar'),
                     os.path.join(os.environ['SPARK_HOME'], 'jars/spark-streaming-kafka-0-8-assembly_2.11-2.3.0.jar'))

_export_pyspark_kafka_jar()

if 'KAFKA_HOME' not in os.environ:
    os.environ['KAFKA_HOME'] = _get_scimple_data_path('kafka')


"""
kafka tools for windows
KAFKA_HOME can be set (overwritten by explicit
"""
is_running = False
home_ = os.environ['KAFKA_HOME']
topics = dict()
zoo = None
kafka = None
time_ = None
window_ = 5
window_scc = 4
scc = None
dstream = None
is_running_scc = False

def set_listening_window(window):
    """
    :param window: listening window in seconds
    :return:
    """
    global is_running, home_, topics, zoo, kafka, time_, window_, window_scc, scc, dstream, is_running_scc
    window_scc = window


def create_dstream(topic):
    """
    create e sparkstreamingcontext linked to a subject and a dstream
    :param topic:  str
    :param window: int seconds for scc buffer window
    :return: dstream
    """
    if 'KAFKA_HOME' not in os.environ:
        os.environ['KAFKA_HOME'] = _get_scimple_data_path('kafka')
    global is_running, home_, topics, zoo, kafka, time_, window_, window_scc, scc, dstream, is_running_scc
    sc, _ = sc_sqlc()  # also used to check pyspark avaibilty
    if not scc:
        from pyspark.streaming import StreamingContext
        scc = StreamingContext(sc, window_scc)
    from pyspark.streaming.kafka import KafkaUtils
    dstream = KafkaUtils.createStream(scc, 'localhost:2181', 'sparkit', {topic: 1})
    return dstream


def start_server(home=None, window=5):
    """
    :param window : window in seconds
    :param home_: path to kafka home_ directory
    :return: None
    """
    global is_running, home_, topics, zoo, kafka, time_, window_, window_scc, scc, dstream, is_running_scc

    if is_running:
        raise Warning('Kafka is already running')
    else:
        window_ = window
        if home:
            home_ = home
        if not home_:
            raise ValueError('please set KAFKA_HOME or provide it as start method argument')
        # print([home_ + '/bin/windows/zookeeper-server-start.bat',
        #        home_ + '/config/zookeeper.properties'])
        zoo = Popen([home_ + '/bin/windows/zookeeper-server-start.bat',
                     home_ + '/config/zookeeper.properties'],
                          universal_newlines=True)
        time.sleep(3)
        kafka_well_started = False
        while not kafka_well_started:
            try:
                kafka = Popen([home_ + '/bin/windows/kafka-server-start.bat',
                               home_ + '/config/server.properties'],
                                    universal_newlines=True)
                kafka.wait(20)
                Popen([home_ + '/bin/windows/kafka-server-stop.bat',
                       home_ + '/config/server.properties']).wait()
            except TimeoutExpired:
                kafka_well_started = True
            # if kafka.poll() is None:  # process en cours bien lancé

        is_running = True
        time_ = time.time()
        print("Kafka server started")


def start_listening():
    global is_running, home_, topics, zoo, kafka, time_, window_, window_scc, scc, dstream, is_running_scc

    if scc:
        scc.start()
        is_running_scc = True
        print("StreamingContext started")
    else:
        raise RuntimeError('no dstreams registered')


def talk(topic, message):
    """
    :param message : str
    :param topic: str : existing topic
    :return: None
    """
    global is_running, home_, topics, zoo, kafka, time_, window_, window_scc, scc, dstream, is_running_scc

    if not is_running:
        raise ValueError('Kafka is not running yet')
    if topic not in topics:
        topics[topic] = [message]
    else:
        topics[topic].append(message)
    if time.time() - time_ > window_:
        time_ = time.time()
        flush()


def flush():
    """
    Throw talked messages
    :return:
    """
    global is_running, home_, topics, zoo, kafka, time_, window_, window_scc, scc, dstream, is_running_scc

    if not is_running:
        raise ValueError('Kafka is not running yet')
    for topic in topics:
        p = Popen([home_ + '/bin/windows/kafka-console-producer.bat',
                   '--broker-list', 'localhost:9092',
                   '--topic', topic], stdin=PIPE, shell=True, universal_newlines=True)
        # '--property', '"parse.key=true"', '--property', '"key.separator=:"',
        p.communicate(input='\n'.join(topics[topic]))
    topics = dict()


def stop_server():
    global is_running, home_, topics, zoo, kafka, time_, window_, window_scc, scc, dstream, is_running_scc

    if is_running:
        flush()
    Popen([home_ + '/bin/windows/kafka-server-stop.bat',
           home_ + '/config/server.properties']).wait()
    Popen([home_ + '/bin/windows/zookeeper-server-stop.bat',
           home_ + '/config/zookeeper.properties']).wait()
    if zoo:
        zoo.kill()
    if kafka:
        kafka.kill()
    for topic_producer in topics.values():
        topic_producer.kill()
    is_running = False
    print("Kafka server ended")


def stop_listening():
    """
    After that you need to reregister dstreams !! Bug if restarting without dstreams
    :return:
    """
    global is_running, home_, topics, zoo, kafka, time_, window_, window_scc, scc, dstream, is_running_scc

    if not is_running_scc:
        raise RuntimeError('not listening')
    scc.stop(stopSparkContext=False)
    scc.awaitTermination()
    is_running_scc = False
    print("StreamingContext closed")
    from pyspark.streaming import StreamingContext
    scc = StreamingContext(sc_sqlc()[0], window_scc)
