
# Scimple
Data Science Simple : Tools scimplifying Matplotlib, Apache Kafka, Apache PySpark
## Goal
Gain time on repetitive things
## Features
```
from scimple import kafka, Plot
```
Plot square function as scimply as : 
`Plot().add(x=range(-10, 10), y=lambda i, x: x[i]**2)`

Start a kafka server and talk on it as scimply as : 
```
kafka.start_server()
kafka.talk(topic="about_cats", message="cats_are_cute")
```

Create a PySpark Streaming dstream listening on a Kafka topic as scimply as : 
```
dstream = kafka.create_dstream(topic="about_cats")
```
## Test it

1. run  `pip install scimple` 
2. Run your favorite *notebook* 
3. Test `./notebook_example.ipynb`

Some outputs from the notebook:

![](https://github.com/EnzoBnl/Scimple/edit/master/screens/1.png)

![](https://github.com/EnzoBnl/Scimple/edit/master/screens/2.png)

![](https://github.com/EnzoBnl/Scimple/edit/master/screens/3.png)
