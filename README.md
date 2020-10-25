# Logi ethical gaming world

2020 Mei-chu Hackathon - Team 726

## Introduction
This is our solution for ethical gaming world. We use machine learning model to
detect whether the user is using hack method for playing, boosting or substituting.

This model idea is from [`Intrusion Detection Using Mouse Dynamics`](https://arxiv.org/pdf/1810.04668.pdf), and we modifiy a bit things of its [`github`](https://github.com/margitantal68/mouse_dynamics_balabit_chaoshen_dfl)

## How to record data
1. Run :  `python3 monitor_with_data_process.py`
2. Switch to the game window
3. Press `s` to start recording 
4. Press `F8` to end the recording
5. The data will store as `new_mouse_log.txt`


## How to start
in the folder `ml`:
1. for training model (Need to change training file path in the code !!) : `python3 main.py`
2. for using model : `python3 test.py <log_file>`

## Contribution
Jack Liu: Mouse monitor
