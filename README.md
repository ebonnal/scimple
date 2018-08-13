# Scimple Lib

##### Tools scimplifying Apache Kafka, Apache Spark, matplotlib

github.com/EnzoBnl/Scimple

enzobonnal@gmail.com

TEST:

Test package by running: 
`pip install scimple`

or:
`git clone https://github.com/EnzoBnl/Scimple`
and
`pip install .`

Then run the test notebook

Example :

```
from scimple import kafka, Plot
```

Plot square function as scimply as : `Plot().add(x=range(-10, 10), y=lambda i, x: x[i]**2)`

Start a kafka server and talk on it as scimply as : 
```
kafka.start_server()
kafka.talk(topic="about_cats", message="cats_are_cute")
```

Create a PySpark Streaming dstream listening on a Kafka topic as scimply as : 
```
dstream = kafka.create_dstream("about_cats")
```
