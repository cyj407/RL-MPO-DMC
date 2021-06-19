# RL MPO Algorithm Replication
## Prerequisites
- python 3.7
- [Mujoco 150]
    - [installization tutorial](https://zhuanlan.zhihu.com/p/358447406?fbclid=IwAR2dQ_1x1uSlCdrK3_MJj10pvQL-1RW29_96V6ufoPrDfHKZJ7wKoE3m3zg)
- [mujoco-py v1.50](https://github.com/openai/mujoco-py/releases/tag/1.50.1.0)
    - unzip and open the directory
    ```
    python3 setup.py install
    ```
    ```
    pip install -U 'mujoco-py<1.50.2,>=1.50.1'
    ```
- [dm_control 0.0.0](https://github.com/deepmind/dm_control/releases/tag/mujoco1.50) (support mujoco 150)
    - unzip and open the directory
    ```
    python3 setup.py install
    ```
- [dm_control2gym](https://github.com/martinseilair/dm_control2gym)
