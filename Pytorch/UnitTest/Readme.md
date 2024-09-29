## [教程Pytest 使用手册 ‒ learning-pytest 1.0 文档](https://learning-pytest.readthedocs.io/zh/latest/index.html)


[Usage and Invocations - pytest documentation](https://docs.pytest.org/en/6.2.x/usage.html)


Pytest不支持多线程，所以在train之前要在cfg中设置不使用多线程cfg.dataloader.num_workers = 0，并且update到args里
在类里组织多个test方法
  会找到所有以test_为前缀的方法并分别测试，测试时，类中每个方法各有一个实例，而不是共享同一个实例
  在类中组织test方法的优点
1. 易于组织
2. 只在指定的类中共享固件
3. 在类的层级上mark，隐式地应用到其中的每个test上