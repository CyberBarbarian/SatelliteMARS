# mission.py

规定了任务`mission`类，`mission`在实例化的时候就会随机生成经纬度等信息，如果需要修改对应信息，直接在`mission`类中修改即可；

# create_mission.py

创建大量随机任务，存储在`data/missions.csv`中，可以修改创建任务规模；

# compute_access.py

读取`data/missons.csv`中的任务，计算每个任务的可访问时段，存储在`data/access.csv`中；
需要连接到已经打开的 STK 11 场景,场景储存在`scenario/RLSTAR.sc`中;

# handle_csv.py

读取`data/missions.csv`和`data/access.csv`
，进行对齐处理，计算当任务出现时，哪些卫星可以观测到，并将对应结果存储在`data/MRL_data.csv`中；

# sort.py

读取`data/MRL_data.csv`，对统一批次中的数据按照到达时间进行排序，将结果存储在`data/MRL_data_sorted.csv`中；

# generate_data.py

整合了`create_mission.py` `compute_access.py`、`handle_csv.py`、`sort.py`，可以一键生成数据；

# augument_data.py

读取指定原始数据，对数据进行增强处理，可以指定从原始数据中抽取的数量和生成的批次；

# no_similate_train.py

读取`data/lab`中的实验数据，训练模型，储存最好的模型，并且储存每轮的reward到`data/reward`中

# plot_training_rewards.py

读取`data/reward`中的reward数据，并画图