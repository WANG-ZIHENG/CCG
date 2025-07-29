# 设定循环的次数，比如这里设置为10
count=1

# while循环通过检查计数器来控制循环次数
while [ $count -le 10 ]
do
    # 在这里可以添加要循环执行的命令
    python cycle_train.py

    # 将计数器加1
    count=$((count + 1))

    # 可以添加其他逻辑来控制循环的行为
done