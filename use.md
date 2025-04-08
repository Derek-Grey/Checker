# cal_backtest1.1版本使用说明：

0、所需包：Pandas、numpy、loguru

1、打开自己的项目后先进行挂载，将数据和引用文件等从数据机挂载过来。

    建议将其挂载在调用文件的同一目录下面。

    创建文件夹：sudo mkdir`<name1>`和`<name2>`

    挂载内容：

    数据：sudo mount -t nfs 192.168.1.237:/data`<name1>`

    引用文件：sudo mount -t nfs 192.168.1.237:/home/sdq/CodeSDQ/DaiShangdin/runonly`<name2>`

2、在使用代码开头中加入引用

    (1)数据库钥匙：from`<name1>` import D1_11_dtype, D1_11_numpy_dtype

    (2)回测模块引用：from`<name2>` import cal_backtestV1_1

3、引用说明：参考代码可以参考Backtesting_reference_example.py

tips： `<name1>`在示例代码中为data、 `<name2>`在示例代码中为ddsd
